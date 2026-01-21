"""
SHA-256 hashing and manifest generation for chain of custody.
Provides cryptographic verification of evidence integrity.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from .models import (
    FileHashRecord,
    ManifestEntry,
    EvidenceManifest,
    EmailRecord,
)
from .config import get_settings

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hex-encoded SHA-256 hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.

    Args:
        data: Bytes to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(data).hexdigest()


def compute_sha256_string(text: str, encoding: str = "utf-8") -> str:
    """
    Compute SHA-256 hash of a string.

    Args:
        text: String to hash
        encoding: Text encoding

    Returns:
        Hex-encoded SHA-256 hash
    """
    return compute_sha256_bytes(text.encode(encoding))


class ManifestGenerator:
    """Generator for evidence manifests and hash records."""

    def __init__(
        self,
        output_dir: Path,
        source_accounts: Optional[List[str]] = None,
    ):
        """
        Initialize manifest generator.

        Args:
            output_dir: Directory for manifest files
            source_accounts: List of source email accounts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        settings = get_settings()
        self.source_accounts = source_accounts or settings.email_accounts
        self.target_correspondents = settings.get_all_target_emails()
        self.target_domains = settings.get_all_target_domains()

        self.bundle_id = str(uuid4())
        self.hash_records: List[FileHashRecord] = []
        self.manifest_entries: List[ManifestEntry] = []
        self.processing_steps: List[Dict[str, Any]] = []

    def log_step(
        self,
        step: str,
        status: str = "completed",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Log a processing step."""
        self.processing_steps.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "status": status,
            "details": details or {},
            "error": error,
        })
        logger.info(f"Processing step: {step} - {status}")

    def add_email_record(self, email_record: EmailRecord) -> ManifestEntry:
        """
        Add an email record to the manifest.

        Args:
            email_record: Parsed email record

        Returns:
            ManifestEntry for the email
        """
        files: List[FileHashRecord] = []

        # Add raw .eml file
        if email_record.raw_eml_path.exists():
            files.append(FileHashRecord(
                file_path=str(email_record.raw_eml_path),
                sha256_hash=email_record.raw_eml_hash,
                file_type="raw_eml",
                size_bytes=email_record.raw_eml_path.stat().st_size,
                doc_id=email_record.gmail_id,
            ))

        # Add body files
        if email_record.body.text_plain_path and email_record.body.text_plain_path.exists():
            files.append(FileHashRecord(
                file_path=str(email_record.body.text_plain_path),
                sha256_hash=email_record.body.text_plain_hash or "",
                file_type="body_text",
                size_bytes=email_record.body.text_plain_path.stat().st_size,
                doc_id=email_record.gmail_id,
            ))

        if email_record.body.text_html_path and email_record.body.text_html_path.exists():
            files.append(FileHashRecord(
                file_path=str(email_record.body.text_html_path),
                sha256_hash=email_record.body.text_html_hash or "",
                file_type="body_html",
                size_bytes=email_record.body.text_html_path.stat().st_size,
                doc_id=email_record.gmail_id,
            ))

        # Add attachments
        for attachment in email_record.attachments:
            if attachment.file_path.exists():
                files.append(FileHashRecord(
                    file_path=str(attachment.file_path),
                    sha256_hash=attachment.sha256_hash,
                    file_type="attachment",
                    size_bytes=attachment.size_bytes,
                    doc_id=email_record.gmail_id,
                ))

        # Add metadata JSON
        if email_record.metadata_json_path.exists():
            files.append(FileHashRecord(
                file_path=str(email_record.metadata_json_path),
                sha256_hash=compute_sha256(email_record.metadata_json_path),
                file_type="metadata",
                size_bytes=email_record.metadata_json_path.stat().st_size,
                doc_id=email_record.gmail_id,
            ))

        # Create manifest entry
        entry = ManifestEntry(
            doc_id=email_record.gmail_id,
            gmail_id=email_record.gmail_id,
            thread_id=email_record.thread_id,
            subject=email_record.headers.subject,
            from_address=email_record.headers.from_address,
            to_addresses=email_record.headers.to_addresses,
            date=email_record.internal_date,
            source_account=email_record.source_account,
            files=files,
        )

        self.manifest_entries.append(entry)
        self.hash_records.extend(files)

        logger.debug(f"Added email {email_record.gmail_id} to manifest ({len(files)} files)")
        return entry

    def add_ai_extraction_path(self, doc_id: str, extraction_path: Path):
        """Add AI extraction file path to a manifest entry."""
        for entry in self.manifest_entries:
            if entry.doc_id == doc_id:
                entry.ai_extraction_path = str(extraction_path)

                # Add to hash records
                if extraction_path.exists():
                    self.hash_records.append(FileHashRecord(
                        file_path=str(extraction_path),
                        sha256_hash=compute_sha256(extraction_path),
                        file_type="ai_extraction",
                        size_bytes=extraction_path.stat().st_size,
                        doc_id=doc_id,
                    ))
                break

    def generate_manifest(self) -> EvidenceManifest:
        """
        Generate the complete evidence manifest.

        Returns:
            EvidenceManifest object
        """
        # Calculate statistics
        total_emails = len(self.manifest_entries)
        total_attachments = sum(
            len([f for f in e.files if f.file_type == "attachment"])
            for e in self.manifest_entries
        )
        total_files = len(self.hash_records)

        # Get date range
        dates = [e.date for e in self.manifest_entries if e.date]
        date_range_start = min(dates) if dates else None
        date_range_end = max(dates) if dates else None

        manifest = EvidenceManifest(
            bundle_id=self.bundle_id,
            source_accounts=self.source_accounts,
            target_correspondents=self.target_correspondents,
            target_domains=self.target_domains,
            total_emails=total_emails,
            total_attachments=total_attachments,
            total_files=total_files,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            entries=self.manifest_entries,
            processing_steps=self.processing_steps,
        )

        return manifest

    def save_manifest(self, manifest: Optional[EvidenceManifest] = None) -> Path:
        """
        Save manifest to JSON file.

        Args:
            manifest: Manifest to save (generates if not provided)

        Returns:
            Path to saved manifest file
        """
        manifest = manifest or self.generate_manifest()

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved manifest to {manifest_path}")
        return manifest_path

    def save_hash_file(self) -> Path:
        """
        Save SHA-256 hash file in standard format.

        Returns:
            Path to saved hash file
        """
        hash_path = self.output_dir / "hashes.sha256"

        lines = []
        for record in self.hash_records:
            # Standard sha256sum format: hash  filename
            lines.append(f"{record.sha256_hash}  {record.file_path}")

        hash_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info(f"Saved hash file to {hash_path} ({len(lines)} entries)")
        return hash_path

    def verify_hashes(self) -> List[Dict[str, Any]]:
        """
        Verify all recorded hashes against files.

        Returns:
            List of verification results
        """
        results = []

        for record in self.hash_records:
            file_path = Path(record.file_path)
            result = {
                "file_path": record.file_path,
                "recorded_hash": record.sha256_hash,
                "doc_id": record.doc_id,
            }

            if not file_path.exists():
                result["status"] = "missing"
                result["current_hash"] = None
            else:
                current_hash = compute_sha256(file_path)
                result["current_hash"] = current_hash
                result["status"] = "verified" if current_hash == record.sha256_hash else "modified"

            results.append(result)

        verified = sum(1 for r in results if r["status"] == "verified")
        logger.info(f"Hash verification: {verified}/{len(results)} files verified")

        return results
