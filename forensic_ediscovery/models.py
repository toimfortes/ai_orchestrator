"""
Pydantic models for forensic e-discovery data structures.
All data models for emails, evidence, extractions, and manifests.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ThreatType(str, Enum):
    """Types of threats/coercion indicators."""
    EXPLICIT_THREAT = "explicit_threat"
    IMPLICIT_THREAT = "implicit_threat"
    COERCION = "coercion"
    BLACKMAIL = "blackmail"
    EXTORTION = "extortion"
    REPUTATIONAL_HARM = "reputational_harm"
    FINANCIAL_PRESSURE = "financial_pressure"
    DEADLINE_PRESSURE = "deadline_pressure"
    DISCLOSURE_THREAT = "disclosure_threat"
    NONE = "none"


class Severity(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EmailHeader(BaseModel):
    """Parsed email headers."""
    from_address: str = Field(alias="from")
    to_addresses: List[str] = Field(default_factory=list, alias="to")
    cc_addresses: List[str] = Field(default_factory=list, alias="cc")
    bcc_addresses: List[str] = Field(default_factory=list, alias="bcc")
    subject: Optional[str] = None
    date: Optional[datetime] = None
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class AttachmentInfo(BaseModel):
    """Information about an email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    sha256_hash: str
    file_path: Path
    is_inline: bool = False
    content_id: Optional[str] = None


class EmailBody(BaseModel):
    """Email body content."""
    text_plain: Optional[str] = None
    text_html: Optional[str] = None
    text_plain_path: Optional[Path] = None
    text_html_path: Optional[Path] = None
    text_plain_hash: Optional[str] = None
    text_html_hash: Optional[str] = None


class EmailRecord(BaseModel):
    """Complete email record with metadata."""
    # Gmail API identifiers
    gmail_id: str
    thread_id: str
    internal_date: datetime

    # Parsed headers
    headers: EmailHeader

    # Body content
    body: EmailBody

    # Attachments
    attachments: List[AttachmentInfo] = Field(default_factory=list)

    # File paths
    raw_eml_path: Path
    raw_eml_hash: str
    metadata_json_path: Path

    # Processing metadata
    source_account: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # Matching info
    matched_correspondent: Optional[str] = None
    direction: Optional[str] = None  # "inbound", "outbound", "cc"


class ThreatIndicator(BaseModel):
    """A detected threat or coercion indicator."""
    threat_type: ThreatType
    severity: Severity
    description: str
    quote_excerpt: str
    confidence: float = Field(ge=0.0, le=1.0)


class FinancialReference(BaseModel):
    """A reference to financial matters."""
    category: str  # "funding", "investment", "transfer", "obligation", "safe_note"
    amount: Optional[str] = None
    description: str
    quote_excerpt: str
    parties_involved: List[str] = Field(default_factory=list)


class FactualClaim(BaseModel):
    """A factual claim extracted from email."""
    claim: str
    speaker: str
    quote_excerpt: str
    supporting_attachments: List[str] = Field(default_factory=list)
    verification_status: str = "unverified"  # "verified", "alleged", "contradicted"


class AIExtraction(BaseModel):
    """Gemini AI extraction results for an email."""
    doc_id: str  # Reference to EmailRecord.gmail_id
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str

    # Extracted data
    threat_indicators: List[ThreatIndicator] = Field(default_factory=list)
    financial_references: List[FinancialReference] = Field(default_factory=list)
    factual_claims: List[FactualClaim] = Field(default_factory=list)

    # Scoring
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)

    # Summary
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)


class FileHashRecord(BaseModel):
    """Record of a file and its hash for chain of custody."""
    file_path: str
    sha256_hash: str
    file_type: str  # "raw_eml", "body_text", "body_html", "attachment", "metadata"
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    doc_id: Optional[str] = None  # Related email doc ID


class ManifestEntry(BaseModel):
    """Entry in the evidence manifest."""
    doc_id: str
    gmail_id: str
    thread_id: str
    subject: Optional[str] = None
    from_address: str
    to_addresses: List[str]
    date: datetime
    source_account: str
    files: List[FileHashRecord]
    ai_extraction_path: Optional[str] = None


class EvidenceManifest(BaseModel):
    """Complete evidence manifest for the bundle."""
    bundle_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "forensic_ediscovery"
    version: str = "1.0.0"

    # Source configuration
    source_accounts: List[str]
    target_correspondents: List[str]
    target_domains: List[str]

    # Statistics
    total_emails: int = 0
    total_attachments: int = 0
    total_files: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None

    # Entries
    entries: List[ManifestEntry] = Field(default_factory=list)

    # Processing log
    processing_steps: List[Dict[str, Any]] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """Entry for the chronological timeline."""
    timestamp: datetime
    doc_id: str
    event_type: str  # "email_sent", "email_received", "attachment"
    subject: Optional[str] = None
    from_address: str
    to_addresses: List[str]
    source_account: str
    threat_level: Optional[Severity] = None
    summary: Optional[str] = None
    relevance_score: float = 0.0


class ProcessingLog(BaseModel):
    """Log entry for processing steps."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    step: str
    status: str  # "started", "completed", "failed"
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
