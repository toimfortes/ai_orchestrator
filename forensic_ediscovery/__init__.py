"""
Forensic E-Discovery Email Processing System

A reproducible tool for exporting, normalizing, and summarizing email evidence
for legal investigations. Produces evidence-safe bundles with cryptographic
hashing, chain of custody manifests, and AI-powered extraction.

Usage:
    # Full pipeline
    python -m forensic_ediscovery full -o ./evidence_output -c credentials.json

    # Individual steps
    python -m forensic_ediscovery export -o ./evidence_output -c credentials.json
    python -m forensic_ediscovery analyze -i ./evidence_output
    python -m forensic_ediscovery bundle -i ./evidence_output
"""

__version__ = "1.0.0"
__author__ = "Forensic E-Discovery Team"

from .config import get_settings, Settings
from .models import (
    EmailRecord,
    EmailHeader,
    EmailBody,
    AttachmentInfo,
    AIExtraction,
    ThreatIndicator,
    FinancialReference,
    FactualClaim,
    EvidenceManifest,
    TimelineEntry,
    ThreatType,
    Severity,
)
from .gmail_adapter import GmailAdapter, create_gmail_adapters
from .email_parser import EmailParser
from .hasher import compute_sha256, ManifestGenerator
from .gemini_extractor import GeminiExtractor, batch_extract
from .bundle_generator import BundleGenerator, create_evidence_bundle

__all__ = [
    # Config
    "get_settings",
    "Settings",
    # Models
    "EmailRecord",
    "EmailHeader",
    "EmailBody",
    "AttachmentInfo",
    "AIExtraction",
    "ThreatIndicator",
    "FinancialReference",
    "FactualClaim",
    "EvidenceManifest",
    "TimelineEntry",
    "ThreatType",
    "Severity",
    # Adapters
    "GmailAdapter",
    "create_gmail_adapters",
    # Parsers
    "EmailParser",
    # Hashing
    "compute_sha256",
    "ManifestGenerator",
    # Extraction
    "GeminiExtractor",
    "batch_extract",
    # Bundle
    "BundleGenerator",
    "create_evidence_bundle",
]
