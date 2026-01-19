"""Data models for research findings and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Finding severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResearchFocus(str, Enum):
    """Focus areas for research."""

    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    RELIABILITY = "reliability"


class ResearchTier(str, Enum):
    """Research tier levels."""

    QUICK = "quick"  # Single provider, fast
    STANDARD = "standard"  # 2 providers, comparison
    COUNCIL = "council"  # 4 providers, full consensus


@dataclass
class Finding:
    """A single finding from research."""

    severity: Severity
    location: str  # file.py:line or general area
    issue: str  # Description of the issue
    evidence: str | None = None  # Code snippet or quote
    fix: str | None = None  # Suggested fix
    confidence: str = "HIGH"  # HIGH, MEDIUM, LOW
    source: str | None = None  # Provider that found it (for disagreements)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "location": self.location,
            "issue": self.issue,
            "evidence": self.evidence,
            "fix": self.fix,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Create from dictionary."""
        return cls(
            severity=Severity(data.get("severity", "medium")),
            location=data.get("location", "unknown"),
            issue=data.get("issue", ""),
            evidence=data.get("evidence"),
            fix=data.get("fix"),
            confidence=data.get("confidence", "HIGH"),
            source=data.get("source"),
        )


@dataclass
class ProviderResult:
    """Result from a single research provider."""

    provider: str
    raw_output: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    duration_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "raw_output": self.raw_output,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ComparisonResult:
    """Result of comparing findings from multiple providers."""

    agreed_findings: list[Finding] = field(default_factory=list)
    disagreed_findings: list[Finding] = field(default_factory=list)
    confidence_score: float = 1.0  # 0-1, based on agreement ratio

    @property
    def total_findings(self) -> int:
        """Total number of unique findings."""
        return len(self.agreed_findings) + len(self.disagreed_findings)

    @property
    def agreement_percentage(self) -> float:
        """Percentage of findings that both providers agreed on."""
        if self.total_findings == 0:
            return 100.0
        return (len(self.agreed_findings) / self.total_findings) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agreed_findings": [f.to_dict() for f in self.agreed_findings],
            "disagreed_findings": [f.to_dict() for f in self.disagreed_findings],
            "confidence_score": self.confidence_score,
            "total_findings": self.total_findings,
            "agreement_percentage": self.agreement_percentage,
        }


@dataclass
class StandardResearchResult:
    """Result of standard (2-provider) research."""

    question: str
    focus: ResearchFocus
    provider_1: ProviderResult
    provider_2: ProviderResult
    comparison: ComparisonResult
    total_duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def confidence(self) -> str:
        """Overall confidence level based on agreement."""
        if self.comparison.confidence_score >= 0.8:
            return "HIGH"
        elif self.comparison.confidence_score >= 0.5:
            return "MEDIUM"
        return "LOW"

    def format_report(self) -> str:
        """Format as human-readable report."""
        lines = [
            "=" * 70,
            "  STANDARD RESEARCH RESULTS",
            f"  Providers: {self.provider_1.provider}, {self.provider_2.provider} | "
            f"Duration: {self.total_duration_seconds:.0f}s | "
            f"Confidence: {self.comparison.agreement_percentage:.0f}%",
            "=" * 70,
        ]

        # Consensus findings
        if self.comparison.agreed_findings:
            lines.append("")
            lines.append("  CONSENSUS (Both providers agree)")
            lines.append("-" * 70)
            for f in self.comparison.agreed_findings:
                lines.append(f"  [{f.severity.value.upper()}] {f.location}")
                lines.append(f"  {f.issue}")
                if f.fix:
                    lines.append(f"  Fix: {f.fix}")
                lines.append("")

        # Disagreements
        if self.comparison.disagreed_findings:
            lines.append("")
            lines.append("  DISAGREEMENTS (Needs human review)")
            lines.append("-" * 70)
            for f in self.comparison.disagreed_findings:
                source_tag = f" ({f.source})" if f.source else ""
                lines.append(f"  [{f.severity.value.upper()}] {f.location}{source_tag}")
                lines.append(f"  {f.issue}")
                lines.append("")

        # Summary
        lines.append("")
        lines.append("  SUMMARY")
        lines.append("-" * 70)
        lines.append(
            f"  {len(self.comparison.agreed_findings)} agreed issues, "
            f"{len(self.comparison.disagreed_findings)} disagreements"
        )

        critical = sum(
            1 for f in self.comparison.agreed_findings if f.severity == Severity.CRITICAL
        )
        high = sum(
            1 for f in self.comparison.agreed_findings if f.severity == Severity.HIGH
        )
        if critical or high:
            lines.append(f"  Priority: {critical} critical, {high} high")

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "focus": self.focus.value,
            "provider_1": self.provider_1.to_dict(),
            "provider_2": self.provider_2.to_dict(),
            "comparison": self.comparison.to_dict(),
            "total_duration_seconds": self.total_duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
        }
