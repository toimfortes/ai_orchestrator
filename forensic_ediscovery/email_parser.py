"""
Email parser for extracting content from raw RFC 2822 emails.
Handles MIME parsing, body extraction, and attachment processing.
"""

import email
import email.policy
import hashlib
import logging
import re
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, List, Tuple

from .models import EmailRecord, EmailHeader, EmailBody, AttachmentInfo
from .hasher import compute_sha256, compute_sha256_bytes

logger = logging.getLogger(__name__)


class EmailParser:
    """Parser for RFC 2822 email messages."""

    def __init__(self, output_dir: Path):
        """
        Initialize email parser.

        Args:
            output_dir: Base directory for saving extracted content
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_raw_email(
        self,
        raw_bytes: bytes,
        gmail_id: str,
        thread_id: str,
        internal_date: int,
        source_account: str,
    ) -> EmailRecord:
        """
        Parse raw email bytes into EmailRecord.

        Args:
            raw_bytes: Raw email in RFC 2822 format
            gmail_id: Gmail message ID
            thread_id: Gmail thread ID
            internal_date: Internal date timestamp (ms)
            source_account: Source email account

        Returns:
            Parsed EmailRecord
        """
        logger.debug(f"Parsing email {gmail_id}")

        # Parse email message
        msg = email.message_from_bytes(raw_bytes, policy=email.policy.default)

        # Create account-specific directory
        account_dir = self.output_dir / source_account.replace("@", "_at_").replace(".", "_")
        email_dir = account_dir / gmail_id
        email_dir.mkdir(parents=True, exist_ok=True)

        # Save raw .eml file
        eml_path = email_dir / f"{gmail_id}.eml"
        eml_path.write_bytes(raw_bytes)
        eml_hash = compute_sha256_bytes(raw_bytes)

        # Parse headers
        headers = self._parse_headers(msg)

        # Extract body content
        body = self._extract_body(msg, email_dir, gmail_id)

        # Extract attachments
        attachments = self._extract_attachments(msg, email_dir, gmail_id)

        # Convert internal date
        internal_datetime = datetime.fromtimestamp(
            internal_date / 1000,  # Gmail uses milliseconds
            tz=timezone.utc,
        )

        # Save metadata JSON path
        metadata_path = email_dir / f"{gmail_id}_metadata.json"

        record = EmailRecord(
            gmail_id=gmail_id,
            thread_id=thread_id,
            internal_date=internal_datetime,
            headers=headers,
            body=body,
            attachments=attachments,
            raw_eml_path=eml_path,
            raw_eml_hash=eml_hash,
            metadata_json_path=metadata_path,
            source_account=source_account,
        )

        # Save metadata
        metadata_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

        logger.info(f"Parsed email {gmail_id}: {headers.subject or '(no subject)'}")
        return record

    def _parse_headers(self, msg: EmailMessage) -> EmailHeader:
        """Parse email headers into structured format."""
        def get_addresses(header_value: Optional[str]) -> List[str]:
            if not header_value:
                return []
            # Simple parsing - could use email.utils.parseaddr for more robustness
            addresses = []
            for part in header_value.split(","):
                part = part.strip()
                # Extract email from "Name <email>" format
                match = re.search(r'<([^>]+)>', part)
                if match:
                    addresses.append(match.group(1).lower())
                elif "@" in part:
                    addresses.append(part.lower())
            return addresses

        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            if not date_str:
                return None
            try:
                return email.utils.parsedate_to_datetime(date_str)
            except Exception:
                return None

        def get_references(ref_str: Optional[str]) -> List[str]:
            if not ref_str:
                return []
            return [r.strip() for r in re.findall(r'<[^>]+>', ref_str)]

        from_addr = msg.get("From", "")
        from_match = re.search(r'<([^>]+)>', from_addr)
        from_email = from_match.group(1) if from_match else from_addr

        return EmailHeader(
            **{
                "from": from_email.lower() if from_email else "",
                "to": get_addresses(msg.get("To")),
                "cc": get_addresses(msg.get("Cc")),
                "bcc": get_addresses(msg.get("Bcc")),
            },
            subject=msg.get("Subject"),
            date=parse_date(msg.get("Date")),
            message_id=msg.get("Message-ID"),
            in_reply_to=msg.get("In-Reply-To"),
            references=get_references(msg.get("References")),
        )

    def _extract_body(
        self,
        msg: EmailMessage,
        email_dir: Path,
        gmail_id: str,
    ) -> EmailBody:
        """Extract text/plain and text/html bodies."""
        text_plain = None
        text_html = None
        text_plain_path = None
        text_html_path = None
        text_plain_hash = None
        text_html_hash = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                if content_type == "text/plain" and text_plain is None:
                    text_plain = self._decode_payload(part)
                elif content_type == "text/html" and text_html is None:
                    text_html = self._decode_payload(part)
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                text_plain = self._decode_payload(msg)
            elif content_type == "text/html":
                text_html = self._decode_payload(msg)

        # Save body files
        if text_plain:
            text_plain_path = email_dir / f"{gmail_id}_body.txt"
            text_plain_path.write_text(text_plain, encoding="utf-8")
            text_plain_hash = compute_sha256(text_plain_path)

        if text_html:
            text_html_path = email_dir / f"{gmail_id}_body.html"
            text_html_path.write_text(text_html, encoding="utf-8")
            text_html_hash = compute_sha256(text_html_path)

        return EmailBody(
            text_plain=text_plain,
            text_html=text_html,
            text_plain_path=text_plain_path,
            text_html_path=text_html_path,
            text_plain_hash=text_plain_hash,
            text_html_hash=text_html_hash,
        )

    def _decode_payload(self, part: EmailMessage) -> Optional[str]:
        """Decode email part payload to string."""
        try:
            payload = part.get_payload(decode=True)
            if payload is None:
                return None

            # Try to detect charset
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset)
            except (UnicodeDecodeError, LookupError):
                # Fallback to utf-8 with error handling
                return payload.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to decode payload: {e}")
            return None

    def _extract_attachments(
        self,
        msg: EmailMessage,
        email_dir: Path,
        gmail_id: str,
    ) -> List[AttachmentInfo]:
        """Extract all attachments from email."""
        attachments = []
        attachment_dir = email_dir / "attachments"

        if not msg.is_multipart():
            return attachments

        for idx, part in enumerate(msg.walk()):
            content_disposition = str(part.get("Content-Disposition", ""))
            content_type = part.get_content_type()

            # Check if it's an attachment or inline content
            is_attachment = "attachment" in content_disposition
            is_inline = "inline" in content_disposition
            content_id = part.get("Content-ID")

            # Skip text bodies unless they're explicit attachments
            if content_type in ("text/plain", "text/html") and not is_attachment:
                continue

            # Skip multipart containers
            if content_type.startswith("multipart/"):
                continue

            # Get filename
            filename = part.get_filename()
            if not filename:
                if is_inline and content_id:
                    # Use content ID for inline images
                    ext = content_type.split("/")[-1] if "/" in content_type else "bin"
                    filename = f"inline_{idx}.{ext}"
                else:
                    continue  # Skip parts without filename

            # Sanitize filename
            filename = self._sanitize_filename(filename)

            # Create attachment directory
            attachment_dir.mkdir(parents=True, exist_ok=True)

            # Handle duplicate filenames
            file_path = attachment_dir / filename
            if file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while file_path.exists():
                    file_path = attachment_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            # Save attachment
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    file_path.write_bytes(payload)
                    file_hash = compute_sha256(file_path)

                    attachments.append(AttachmentInfo(
                        filename=filename,
                        content_type=content_type,
                        size_bytes=len(payload),
                        sha256_hash=file_hash,
                        file_path=file_path,
                        is_inline=is_inline,
                        content_id=content_id,
                    ))
                    logger.debug(f"Saved attachment: {filename} ({len(payload)} bytes)")

            except Exception as e:
                logger.warning(f"Failed to save attachment {filename}: {e}")

        return attachments

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, "_")

        # Limit length
        if len(filename) > 200:
            stem = filename[:190]
            ext = filename[-10:] if "." in filename[-10:] else ""
            filename = stem + ext

        return filename.strip()


def parse_emails_from_export(
    export_dir: Path,
    output_dir: Path,
    source_account: str,
) -> List[EmailRecord]:
    """
    Parse all exported emails from a directory.

    Args:
        export_dir: Directory containing exported .eml files
        output_dir: Directory for parsed output
        source_account: Source email account

    Returns:
        List of parsed EmailRecords
    """
    parser = EmailParser(output_dir)
    records = []

    for eml_file in export_dir.glob("*.eml"):
        try:
            raw_bytes = eml_file.read_bytes()
            gmail_id = eml_file.stem  # Use filename as ID

            record = parser.parse_raw_email(
                raw_bytes=raw_bytes,
                gmail_id=gmail_id,
                thread_id="unknown",
                internal_date=int(eml_file.stat().st_mtime * 1000),
                source_account=source_account,
            )
            records.append(record)

        except Exception as e:
            logger.error(f"Failed to parse {eml_file}: {e}")

    return records
