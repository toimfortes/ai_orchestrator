"""
Gemini AI structured extraction for forensic email analysis.
Uses google-genai SDK for threat detection and evidence extraction.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Union

from pydantic import BaseModel, Field

from .config import get_settings, GeminiConfig
from .models import (
    AIExtraction,
    ThreatIndicator,
    FinancialReference,
    FactualClaim,
    ThreatType,
    Severity,
    EmailRecord,
)

logger = logging.getLogger(__name__)

# Gemini extraction schema for structured outputs
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "threat_indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "threat_type": {
                        "type": "string",
                        "enum": [
                            "explicit_threat", "implicit_threat", "coercion",
                            "blackmail", "extortion", "reputational_harm",
                            "financial_pressure", "deadline_pressure",
                            "disclosure_threat", "none"
                        ]
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"]
                    },
                    "description": {"type": "string"},
                    "quote_excerpt": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["threat_type", "severity", "description", "quote_excerpt", "confidence"]
            }
        },
        "financial_references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "amount": {"type": "string"},
                    "description": {"type": "string"},
                    "quote_excerpt": {"type": "string"},
                    "parties_involved": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["category", "description", "quote_excerpt"]
            }
        },
        "factual_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "speaker": {"type": "string"},
                    "quote_excerpt": {"type": "string"},
                    "supporting_attachments": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "verification_status": {
                        "type": "string",
                        "enum": ["unverified", "verified", "alleged", "contradicted"]
                    }
                },
                "required": ["claim", "speaker", "quote_excerpt"]
            }
        },
        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
        "requires_human_review": {"type": "boolean"},
        "review_reasons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "threat_indicators", "financial_references", "factual_claims",
        "relevance_score", "requires_human_review", "summary", "key_points"
    ]
}


class GeminiExtractor:
    """Extractor using Gemini AI for structured evidence analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_reasoning_model: bool = False,
    ):
        """
        Initialize Gemini extractor.

        Args:
            api_key: Gemini API key (default from env)
            model: Model to use (default from settings)
            use_reasoning_model: Use deeper reasoning model
        """
        self.settings = get_settings()
        self.gemini_config = self.settings.gemini

        # Get API key
        self.api_key = api_key or self.settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Select model
        if model:
            self.model_name = model
        elif use_reasoning_model:
            self.model_name = self.gemini_config.reasoning_model
        else:
            self.model_name = self.gemini_config.default_model

        self._client = None

    @property
    def client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini client with model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def _build_extraction_prompt(
        self,
        email_record: EmailRecord,
        context: Optional[str] = None,
    ) -> str:
        """Build the extraction prompt for an email."""
        # Get email content
        body_text = email_record.body.text_plain or ""
        if not body_text and email_record.body.text_html:
            # Strip HTML tags for basic text extraction
            import re
            body_text = re.sub(r'<[^>]+>', '', email_record.body.text_html)

        # Build attachment list
        attachment_list = ""
        if email_record.attachments:
            attachment_list = "\n\nAttachments:\n" + "\n".join(
                f"- {a.filename} ({a.content_type}, {a.size_bytes} bytes)"
                for a in email_record.attachments
            )

        prompt = f"""Analyze this email for a forensic investigation related to alleged blackmail/extortion.

DOCUMENT ID: {email_record.gmail_id}
DATE: {email_record.internal_date.isoformat()}
FROM: {email_record.headers.from_address}
TO: {', '.join(email_record.headers.to_addresses)}
CC: {', '.join(email_record.headers.cc_addresses) if email_record.headers.cc_addresses else 'None'}
SUBJECT: {email_record.headers.subject or '(no subject)'}
{attachment_list}

EMAIL BODY:
{body_text[:10000]}  # Limit body length

---

ANALYSIS INSTRUCTIONS:
1. Identify any threats, coercion, or blackmail indicators (explicit or implicit)
2. Note any financial references (funding, investment, obligations, transfers)
3. Extract factual claims with direct quote excerpts
4. Flag if human review is required (ambiguous threats, sensitive content)
5. Provide a relevance score (0-1) for the investigation

CONTEXT:
{context or "This email is part of an investigation into alleged blackmail where the suspect allegedly threatened to disclose personal conversations to business partners if demands were not met."}

Return a structured JSON response with your analysis.
"""
        return prompt

    def extract_from_email(
        self,
        email_record: EmailRecord,
        context: Optional[str] = None,
        attachment_files: Optional[List[Path]] = None,
    ) -> AIExtraction:
        """
        Extract structured information from an email.

        Args:
            email_record: Parsed email record
            context: Additional context for analysis
            attachment_files: Attachment files to include in analysis

        Returns:
            AIExtraction with structured findings
        """
        logger.info(f"Extracting from email {email_record.gmail_id}")

        prompt = self._build_extraction_prompt(email_record, context)

        try:
            # Build content parts
            contents = [prompt]

            # Add attachments if supported and within size limits
            if attachment_files:
                for file_path in attachment_files:
                    if self._can_include_file(file_path):
                        contents.append(self._create_file_part(file_path))

            # Call Gemini with structured output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": EXTRACTION_SCHEMA,
                    "temperature": self.gemini_config.temperature,
                    "max_output_tokens": self.gemini_config.max_output_tokens,
                }
            )

            # Parse response
            result = json.loads(response.text)

            # Convert to AIExtraction model
            extraction = AIExtraction(
                doc_id=email_record.gmail_id,
                model_used=self.model_name,
                threat_indicators=[
                    ThreatIndicator(
                        threat_type=ThreatType(t.get("threat_type", "none")),
                        severity=Severity(t.get("severity", "info")),
                        description=t.get("description", ""),
                        quote_excerpt=t.get("quote_excerpt", ""),
                        confidence=t.get("confidence", 0.5),
                    )
                    for t in result.get("threat_indicators", [])
                ],
                financial_references=[
                    FinancialReference(
                        category=f.get("category", ""),
                        amount=f.get("amount"),
                        description=f.get("description", ""),
                        quote_excerpt=f.get("quote_excerpt", ""),
                        parties_involved=f.get("parties_involved", []),
                    )
                    for f in result.get("financial_references", [])
                ],
                factual_claims=[
                    FactualClaim(
                        claim=c.get("claim", ""),
                        speaker=c.get("speaker", ""),
                        quote_excerpt=c.get("quote_excerpt", ""),
                        supporting_attachments=c.get("supporting_attachments", []),
                        verification_status=c.get("verification_status", "unverified"),
                    )
                    for c in result.get("factual_claims", [])
                ],
                relevance_score=result.get("relevance_score", 0.0),
                requires_human_review=result.get("requires_human_review", False),
                review_reasons=result.get("review_reasons", []),
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
            )

            logger.info(
                f"Extracted from {email_record.gmail_id}: "
                f"{len(extraction.threat_indicators)} threats, "
                f"{len(extraction.financial_references)} financial refs, "
                f"relevance={extraction.relevance_score:.2f}"
            )

            return extraction

        except Exception as e:
            logger.error(f"Extraction failed for {email_record.gmail_id}: {e}", exc_info=True)
            # Return empty extraction with error flag
            return AIExtraction(
                doc_id=email_record.gmail_id,
                model_used=self.model_name,
                requires_human_review=True,
                review_reasons=[f"Extraction failed: {str(e)}"],
                summary=f"Extraction error: {str(e)}",
            )

    def _can_include_file(self, file_path: Path) -> bool:
        """Check if a file can be included in Gemini request."""
        if not file_path.exists():
            return False

        size = file_path.stat().st_size
        if size > self.gemini_config.max_file_size_bytes:
            logger.debug(f"File too large for inline: {file_path} ({size} bytes)")
            return False

        # Check MIME type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in self.gemini_config.supported_mime_types:
            return True

        return False

    def _create_file_part(self, file_path: Path) -> dict:
        """Create a file part for Gemini request."""
        import mimetypes
        import base64

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        data = file_path.read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")

        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": encoded,
            }
        }

    def save_extraction(
        self,
        extraction: AIExtraction,
        output_dir: Path,
    ) -> Path:
        """
        Save extraction to JSON file.

        Args:
            extraction: AIExtraction to save
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{extraction.doc_id}_extraction.json"
        output_path.write_text(
            extraction.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.debug(f"Saved extraction to {output_path}")
        return output_path


def batch_extract(
    email_records: List[EmailRecord],
    output_dir: Path,
    api_key: Optional[str] = None,
    use_reasoning_model: bool = False,
    progress_callback=None,
) -> List[AIExtraction]:
    """
    Batch extract from multiple emails.

    Args:
        email_records: List of email records to process
        output_dir: Directory for extraction outputs
        api_key: Gemini API key
        use_reasoning_model: Use deeper reasoning model
        progress_callback: Callback(current, total) for progress

    Returns:
        List of AIExtraction results
    """
    extractor = GeminiExtractor(
        api_key=api_key,
        use_reasoning_model=use_reasoning_model,
    )

    extractions = []
    total = len(email_records)

    for idx, record in enumerate(email_records):
        try:
            extraction = extractor.extract_from_email(record)
            extractor.save_extraction(extraction, output_dir)
            extractions.append(extraction)

            if progress_callback:
                progress_callback(idx + 1, total)

        except Exception as e:
            logger.error(f"Failed to extract from {record.gmail_id}: {e}")
            # Add placeholder extraction
            extractions.append(AIExtraction(
                doc_id=record.gmail_id,
                model_used=extractor.model_name,
                requires_human_review=True,
                review_reasons=[f"Processing failed: {str(e)}"],
            ))

    return extractions
