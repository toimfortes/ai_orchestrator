"""
Configuration management for Forensic E-Discovery system.
All configurable values centralized here - no hardcoding elsewhere.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class TargetCorrespondent(BaseModel):
    """A target correspondent to match in email searches."""
    name: str
    emails: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)


class GmailConfig(BaseModel):
    """Gmail API configuration."""
    scopes: List[str] = Field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.readonly"
    ])
    credentials_file: str = "credentials.json"
    token_file_pattern: str = "token_{account}.json"
    max_results_per_query: int = 500
    batch_size: int = 100


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    default_model: str = "gemini-2.5-flash"
    reasoning_model: str = "gemini-2.5-pro"
    max_file_size_bytes: int = 20 * 1024 * 1024  # 20MB for inline
    supported_mime_types: List[str] = Field(default_factory=lambda: [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "audio/mp3",
        "audio/wav",
        "audio/mpeg",
        "video/mp4",
        "video/mpeg",
        "text/plain",
        "text/html",
        "text/csv",
    ])
    temperature: float = 0.1  # Low for factual extraction
    max_output_tokens: int = 8192


class HashingConfig(BaseModel):
    """File hashing configuration."""
    algorithm: str = "sha256"
    chunk_size: int = 65536  # 64KB chunks for large files


class OutputConfig(BaseModel):
    """Output bundle configuration."""
    bundle_dir: str = "bundle"
    emails_subdir: str = "emails"
    ai_extractions_subdir: str = "ai_extractions"
    zip_filename: str = "evidence_bundle.zip"


class Settings(BaseSettings):
    """Main settings for the forensic e-discovery system."""

    # Environment variable prefix
    model_config = {"env_prefix": "FORENSIC_", "env_file": ".env", "extra": "ignore"}

    # API Keys (from environment)
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")

    # Email accounts to process
    email_accounts: List[str] = Field(default_factory=lambda: [
        "cortexcerebral@gmail.com",
        "toimusa@gmail.com",
        "antonioforteslegal@gmail.com",
    ])

    # Target correspondents
    target_correspondents: List[TargetCorrespondent] = Field(default_factory=lambda: [
        TargetCorrespondent(
            name="Aslan",
            emails=["aslan@planingo.ai"]
        ),
        TargetCorrespondent(
            name="Toofan",
            emails=["toofans@aslancloud.com"]
        ),
        TargetCorrespondent(
            name="EMM Legal",
            emails=["NatalieTenorioBernal@emmlegal.com", "andrewmarshall@emmlegal.com"],
            domains=["emmlegal.com"]
        ),
    ])

    # Key incident date (for timeline anchoring)
    incident_date: str = "2025-12-20"
    incident_description: str = (
        "Aslan allegedly called business partner and disclosed personal "
        "conversations after refusal to concede to blackmail."
    )

    # Sub-configurations
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    hashing: HashingConfig = Field(default_factory=HashingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Processing options
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "evidence_output")
    verbose: bool = False
    dry_run: bool = False

    def get_all_target_emails(self) -> List[str]:
        """Get flat list of all target email addresses."""
        emails = []
        for correspondent in self.target_correspondents:
            emails.extend(correspondent.emails)
        return emails

    def get_all_target_domains(self) -> List[str]:
        """Get flat list of all target domains."""
        domains = []
        for correspondent in self.target_correspondents:
            domains.extend(correspondent.domains)
        return domains

    def build_gmail_query(self) -> str:
        """Build Gmail search query for target correspondents."""
        conditions = []

        # Add email conditions
        for email in self.get_all_target_emails():
            conditions.append(f"from:{email}")
            conditions.append(f"to:{email}")

        # Add domain conditions
        for domain in self.get_all_target_domains():
            conditions.append(f"from:@{domain}")
            conditions.append(f"to:@{domain}")

        # Join with OR
        return " OR ".join(conditions)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
