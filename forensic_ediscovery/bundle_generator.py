"""
Evidence bundle generator for forensic e-discovery.
Creates complete evidence packages with manifests, timelines, and ZIP archives.
"""

import csv
import json
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import get_settings
from .models import (
    EmailRecord,
    AIExtraction,
    EvidenceManifest,
    TimelineEntry,
    Severity,
)
from .hasher import ManifestGenerator, compute_sha256

logger = logging.getLogger(__name__)


class BundleGenerator:
    """Generator for complete evidence bundles."""

    def __init__(
        self,
        output_dir: Path,
        bundle_name: str = "evidence_bundle",
    ):
        """
        Initialize bundle generator.

        Args:
            output_dir: Base output directory
            bundle_name: Name for the bundle
        """
        self.output_dir = output_dir
        self.bundle_name = bundle_name
        self.bundle_dir = output_dir / "bundle"
        self.settings = get_settings()

        # Create bundle structure
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        (self.bundle_dir / "emails").mkdir(exist_ok=True)
        (self.bundle_dir / "ai_extractions").mkdir(exist_ok=True)

        self.manifest_generator = ManifestGenerator(
            output_dir=self.bundle_dir,
        )

        self.email_records: List[EmailRecord] = []
        self.ai_extractions: List[AIExtraction] = []

    def add_email(self, email_record: EmailRecord):
        """Add an email record to the bundle."""
        self.email_records.append(email_record)
        self.manifest_generator.add_email_record(email_record)

    def add_extraction(self, extraction: AIExtraction, extraction_path: Optional[Path] = None):
        """Add an AI extraction to the bundle."""
        self.ai_extractions.append(extraction)
        if extraction_path:
            self.manifest_generator.add_ai_extraction_path(extraction.doc_id, extraction_path)

    def generate_timeline(self) -> Path:
        """
        Generate chronological timeline CSV.

        Returns:
            Path to timeline.csv
        """
        logger.info("Generating timeline...")

        # Build timeline entries
        entries: List[TimelineEntry] = []

        for record in self.email_records:
            # Determine direction
            from_addr = record.headers.from_address.lower()
            source_lower = record.source_account.lower()

            if from_addr == source_lower:
                event_type = "email_sent"
            else:
                event_type = "email_received"

            # Find matching extraction for threat level
            threat_level = None
            relevance = 0.0
            summary = None

            for extraction in self.ai_extractions:
                if extraction.doc_id == record.gmail_id:
                    relevance = extraction.relevance_score
                    summary = extraction.summary

                    # Get highest threat severity
                    if extraction.threat_indicators:
                        severities = [t.severity for t in extraction.threat_indicators]
                        if Severity.CRITICAL in severities:
                            threat_level = Severity.CRITICAL
                        elif Severity.HIGH in severities:
                            threat_level = Severity.HIGH
                        elif Severity.MEDIUM in severities:
                            threat_level = Severity.MEDIUM
                        elif Severity.LOW in severities:
                            threat_level = Severity.LOW
                    break

            entries.append(TimelineEntry(
                timestamp=record.internal_date,
                doc_id=record.gmail_id,
                event_type=event_type,
                subject=record.headers.subject,
                from_address=record.headers.from_address,
                to_addresses=record.headers.to_addresses,
                source_account=record.source_account,
                threat_level=threat_level,
                summary=summary,
                relevance_score=relevance,
            ))

        # Sort by timestamp
        entries.sort(key=lambda e: e.timestamp)

        # Write CSV
        timeline_path = self.bundle_dir / "timeline.csv"

        with open(timeline_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "doc_id", "event_type", "subject",
                "from", "to", "source_account", "threat_level",
                "relevance_score", "summary"
            ])

            for entry in entries:
                writer.writerow([
                    entry.timestamp.isoformat(),
                    entry.doc_id,
                    entry.event_type,
                    entry.subject or "",
                    entry.from_address,
                    ";".join(entry.to_addresses),
                    entry.source_account,
                    entry.threat_level.value if entry.threat_level else "",
                    f"{entry.relevance_score:.2f}",
                    entry.summary or "",
                ])

        logger.info(f"Generated timeline with {len(entries)} entries: {timeline_path}")
        return timeline_path

    def generate_index(self) -> Path:
        """
        Generate index.jsonl with all email metadata.

        Returns:
            Path to index.jsonl
        """
        logger.info("Generating index...")

        index_path = self.bundle_dir / "index.jsonl"

        with open(index_path, "w", encoding="utf-8") as f:
            for record in self.email_records:
                entry = {
                    "doc_id": record.gmail_id,
                    "thread_id": record.thread_id,
                    "timestamp": record.internal_date.isoformat(),
                    "from": record.headers.from_address,
                    "to": record.headers.to_addresses,
                    "cc": record.headers.cc_addresses,
                    "subject": record.headers.subject,
                    "message_id": record.headers.message_id,
                    "source_account": record.source_account,
                    "attachment_count": len(record.attachments),
                    "raw_eml_hash": record.raw_eml_hash,
                }
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Generated index with {len(self.email_records)} entries: {index_path}")
        return index_path

    def generate_readme(self) -> Path:
        """
        Generate README.md explaining the bundle.

        Returns:
            Path to README.md
        """
        readme_content = f"""# Evidence Bundle

## Bundle Information
- **Bundle ID**: {self.manifest_generator.bundle_id}
- **Created**: {datetime.now(timezone.utc).isoformat()}
- **Created By**: forensic_ediscovery v1.0.0

## Source Accounts
{chr(10).join(f'- {a}' for a in self.manifest_generator.source_accounts)}

## Target Correspondents
{chr(10).join(f'- {c}' for c in self.manifest_generator.target_correspondents)}

## Target Domains
{chr(10).join(f'- {d}' for d in self.manifest_generator.target_domains)}

## Contents

### Files
- `manifest.json` - Complete evidence manifest with file hashes
- `hashes.sha256` - SHA-256 checksums for all files
- `timeline.csv` - Chronological timeline of all communications
- `index.jsonl` - Searchable metadata index (JSON Lines format)
- `DEEP_RESEARCH_PROMPT.md` - Instructions for investigator AI analysis

### Directories
- `emails/` - Raw .eml files, parsed JSON, and attachments organized by account
- `ai_extractions/` - Gemini AI analysis results for each email

## Statistics
- **Total Emails**: {len(self.email_records)}
- **Total Attachments**: {sum(len(r.attachments) for r in self.email_records)}

## Verification

To verify file integrity, run:
```bash
sha256sum -c hashes.sha256
```

## Chain of Custody

All files in this bundle are cryptographically hashed. The `manifest.json` file
contains complete provenance information including:
- Original Gmail message IDs
- Processing timestamps
- SHA-256 hashes for all files
- Source account attribution

## Processing Log

See `manifest.json` > `processing_steps` for detailed processing history.

## Legal Notice

This evidence bundle was created for authorized investigative purposes.
All original files are preserved without modification.
"""

        readme_path = self.bundle_dir / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")

        logger.info(f"Generated README: {readme_path}")
        return readme_path

    def copy_deep_research_prompt(self, prompt_path: Optional[Path] = None) -> Path:
        """
        Copy or generate DEEP_RESEARCH_PROMPT.md.

        Args:
            prompt_path: Path to existing prompt file

        Returns:
            Path to prompt file in bundle
        """
        dest_path = self.bundle_dir / "DEEP_RESEARCH_PROMPT.md"

        if prompt_path and prompt_path.exists():
            shutil.copy(prompt_path, dest_path)
        else:
            # Generate default prompt
            dest_path.write_text(generate_deep_research_prompt(), encoding="utf-8")

        logger.info(f"Added DEEP_RESEARCH_PROMPT.md: {dest_path}")
        return dest_path

    def finalize_bundle(self) -> EvidenceManifest:
        """
        Finalize the bundle with all manifests and files.

        Returns:
            Final EvidenceManifest
        """
        logger.info("Finalizing evidence bundle...")

        # Generate all files
        self.generate_timeline()
        self.generate_index()
        self.generate_readme()
        self.copy_deep_research_prompt()

        # Log final step
        self.manifest_generator.log_step("bundle_finalized", details={
            "total_emails": len(self.email_records),
            "total_extractions": len(self.ai_extractions),
        })

        # Save manifest and hashes
        manifest = self.manifest_generator.generate_manifest()
        self.manifest_generator.save_manifest(manifest)
        self.manifest_generator.save_hash_file()

        logger.info("Bundle finalized successfully")
        return manifest

    def create_zip(self) -> Path:
        """
        Create ZIP archive of the bundle.

        Returns:
            Path to ZIP file
        """
        zip_path = self.output_dir / f"{self.bundle_name}.zip"

        logger.info(f"Creating ZIP archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.bundle_dir.rglob("*"):
                if file_path.is_file():
                    arc_name = file_path.relative_to(self.bundle_dir)
                    zf.write(file_path, arc_name)

        # Compute ZIP hash
        zip_hash = compute_sha256(zip_path)
        logger.info(f"Created ZIP archive: {zip_path} (SHA-256: {zip_hash})")

        # Save ZIP hash
        hash_file = self.output_dir / f"{self.bundle_name}.zip.sha256"
        hash_file.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="utf-8")

        return zip_path


def generate_deep_research_prompt() -> str:
    """Generate the DEEP_RESEARCH_PROMPT.md content."""
    return '''# Deep Research Prompt for Forensic Investigation

## Role
You are an investigator writing an evidence-based dossier for law enforcement review.

## Inputs
- `timeline.csv` - Chronological timeline of all communications
- `index.jsonl` - Metadata index for all emails
- `emails/` - Raw .eml files, parsed content, and attachments
- `ai_extractions/` - AI-generated analysis for each email

## Non-Negotiables

1. **Citation Required**: Do not assert anything as fact without citing at least one document ID from `index.jsonl`

2. **Evidence Format**: For every allegation, include:
   - (a) Date/time
   - (b) Who said/did what
   - (c) Direct quote excerpt(s)
   - (d) Document IDs
   - (e) Related attachments and their hashes

3. **Classification**: Separate clearly:
   - **Verified by evidence** - Claims directly supported by documents
   - **Alleged by complainant** - Claims made but not independently verified
   - **Inferred** - Logical conclusions drawn from evidence patterns

## Analysis Tasks

### 1. Timeline Reconstruction
Build a minute-resolution timeline around the key incident date (2025-12-20) and the days before/after:
- Map all communications chronologically
- Identify escalation patterns
- Note gaps in communication that may be significant

### 2. Threat/Coercion Identification
For each identified threat or coercion indicator:
- Classify threat type (blackmail, extortion, implicit threat, etc.)
- Extract exact demands
- Note deadlines or ultimatums
- Document escalation patterns
- Identify mentions of disclosure or reputational harm
- Flag investor pressure tactics

### 3. Financial Relationship Evidence
Document all financial evidence:
- Funding amounts and transfers
- Promises or obligations
- SAFE/notes/equity mentions
- Investor introductions and follow-up
- Attempts to leverage finances for pressure

### 4. Witness and Contact Identification
Identify potential witnesses:
- Business partner names/roles
- Other investor contacts referenced
- Third parties mentioned in communications
- Anyone who may have direct knowledge

## Required Outputs

### DOSSIER.md
A police-ready narrative summary (1-2 pages) including:
- Executive summary
- Timeline highlights
- Evidence index
- Key findings

### EVIDENCE_TABLE.csv
One row per evidence item:
| DocID | Date | Type | Description | Quote | Attachments | Hash |

### OPEN_QUESTIONS.md
- What evidence is missing
- Recommended next investigative steps
- Suggested subpoenas (phone logs, metadata requests)
- NOTE: State as suggestions, not legal advice

## Important Guidelines

- Use only evidence present in this bundle
- Do not fabricate or assume facts not in evidence
- Quote directly when possible
- Cross-reference with attachments when available
- Flag ambiguous or contradictory evidence
- Maintain objectivity - present evidence, not conclusions

## Output Format

All outputs should be Markdown formatted for readability.
Include document IDs in brackets [DOC_ID] for easy reference.
'''


def create_evidence_bundle(
    email_records: List[EmailRecord],
    ai_extractions: List[AIExtraction],
    output_dir: Path,
    bundle_name: str = "evidence_bundle",
) -> tuple[EvidenceManifest, Path]:
    """
    Convenience function to create a complete evidence bundle.

    Args:
        email_records: Parsed email records
        ai_extractions: AI extraction results
        output_dir: Output directory
        bundle_name: Name for the bundle

    Returns:
        Tuple of (manifest, zip_path)
    """
    generator = BundleGenerator(output_dir, bundle_name)

    for record in email_records:
        generator.add_email(record)

    # Match extractions to emails
    extraction_map = {e.doc_id: e for e in ai_extractions}
    for record in email_records:
        if record.gmail_id in extraction_map:
            extraction = extraction_map[record.gmail_id]
            extraction_path = output_dir / "bundle" / "ai_extractions" / f"{extraction.doc_id}_extraction.json"
            generator.add_extraction(extraction, extraction_path)

    manifest = generator.finalize_bundle()
    zip_path = generator.create_zip()

    return manifest, zip_path
