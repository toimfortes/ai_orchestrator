"""Finding extraction and comparison logic for multi-provider research."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from ai_orchestrator.research.models import (
    ComparisonResult,
    Finding,
    Severity,
)

logger = logging.getLogger(__name__)


def extract_findings(raw_output: str) -> list[Finding]:
    """
    Extract structured findings from raw LLM output.

    Parses markdown-formatted output looking for finding patterns.
    Handles variations in formatting from different providers.

    Args:
        raw_output: Raw text output from research provider.

    Returns:
        List of extracted Finding objects.
    """
    findings: list[Finding] = []

    # Pattern for finding headers: **[SEVERITY]** file.py:123 - title
    # or variations like [CRITICAL] file.py:123
    finding_pattern = re.compile(
        r"\*?\*?\[?(CRITICAL|HIGH|MEDIUM|LOW)\]?\*?\*?\s+"
        r"([^\s\-:]+(?::\d+)?)\s*[-:]\s*(.+?)(?:\n|$)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Split into sections by finding headers
    sections = re.split(r"(?=\*?\*?\[?(?:CRITICAL|HIGH|MEDIUM|LOW)\]?\*?\*?)", raw_output)

    for section in sections:
        if not section.strip():
            continue

        match = finding_pattern.search(section)
        if not match:
            continue

        severity_str = match.group(1).upper()
        location = match.group(2).strip()
        title = match.group(3).strip()

        # Map severity string to enum
        try:
            severity = Severity(severity_str.lower())
        except ValueError:
            severity = Severity.MEDIUM

        # Extract issue description (look for "Issue:" or just use title)
        issue_match = re.search(r"Issue:\s*(.+?)(?:\n\s*-|\n\s*Evidence:|\n\s*Fix:|$)", section, re.DOTALL)
        issue = issue_match.group(1).strip() if issue_match else title

        # Extract evidence
        evidence_match = re.search(r"Evidence:\s*`?(.+?)`?(?:\n\s*-|\n\s*Fix:|$)", section, re.DOTALL)
        evidence = evidence_match.group(1).strip() if evidence_match else None

        # Extract fix suggestion
        fix_match = re.search(r"Fix:\s*(.+?)(?:\n\s*-|\n\*\*|$)", section, re.DOTALL)
        fix = fix_match.group(1).strip() if fix_match else None

        findings.append(Finding(
            severity=severity,
            location=location,
            issue=issue,
            evidence=evidence,
            fix=fix,
        ))

    # Fallback: if no structured findings found, try simpler patterns
    if not findings:
        findings = _extract_findings_simple(raw_output)

    logger.debug("Extracted %d findings from output", len(findings))
    return findings


def _extract_findings_simple(raw_output: str) -> list[Finding]:
    """
    Simple fallback extraction for less structured output.

    Looks for common patterns like:
    - Bullet points with severity keywords
    - Numbered lists with file references
    """
    findings: list[Finding] = []

    # Pattern: lines containing severity + file reference
    simple_pattern = re.compile(
        r"[-*\d.]\s*(critical|high|medium|low)[:\s]+(.+?)(?:\n|$)",
        re.IGNORECASE,
    )

    for match in simple_pattern.finditer(raw_output):
        severity_str = match.group(1).upper()
        content = match.group(2).strip()

        # Try to extract file:line from content
        location_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*\.py(?::\d+)?)", content)
        location = location_match.group(1) if location_match else "unknown"

        try:
            severity = Severity(severity_str.lower())
        except ValueError:
            severity = Severity.MEDIUM

        findings.append(Finding(
            severity=severity,
            location=location,
            issue=content,
        ))

    return findings


def find_similar_finding(
    target: Finding,
    candidates: list[Finding],
    location_threshold: int = 10,
    text_threshold: float = 0.5,
) -> Finding | None:
    """
    Find a similar finding in the candidates list.

    Matching heuristics:
    1. Same file AND within N lines = likely same issue
    2. High text similarity in issue description = likely same issue

    Args:
        target: The finding to match.
        candidates: List of findings to search.
        location_threshold: Max line difference for location match.
        text_threshold: Min similarity ratio for text match.

    Returns:
        Matching Finding if found, None otherwise.
    """
    target_file = _extract_file(target.location)
    target_line = _extract_line(target.location)

    for candidate in candidates:
        # Location-based matching
        cand_file = _extract_file(candidate.location)
        cand_line = _extract_line(candidate.location)

        if target_file and cand_file and target_file == cand_file:
            if target_line and cand_line:
                if abs(target_line - cand_line) <= location_threshold:
                    return candidate

        # Text similarity matching (fallback)
        similarity = _text_similarity(target.issue, candidate.issue)
        if similarity >= text_threshold:
            return candidate

    return None


def _extract_file(location: str) -> str | None:
    """Extract file name from location string."""
    match = re.match(r"([^:]+\.py)", location)
    return match.group(1) if match else None


def _extract_line(location: str) -> int | None:
    """Extract line number from location string."""
    match = re.search(r":(\d+)", location)
    return int(match.group(1)) if match else None


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using SequenceMatcher."""
    # Normalize texts
    t1 = text1.lower().split()
    t2 = text2.lower().split()

    # Use SequenceMatcher for similarity
    return SequenceMatcher(None, t1, t2).ratio()


def merge_findings(f1: Finding, f2: Finding) -> Finding:
    """
    Merge two similar findings into one.

    Prefers more specific information from either source.
    """
    # Use higher severity
    if _severity_rank(f1.severity) >= _severity_rank(f2.severity):
        severity = f1.severity
    else:
        severity = f2.severity

    # Prefer more specific location
    location = f1.location if ":" in f1.location else f2.location

    # Combine issues (prefer longer/more detailed)
    issue = f1.issue if len(f1.issue) >= len(f2.issue) else f2.issue

    # Prefer available evidence
    evidence = f1.evidence or f2.evidence

    # Prefer available fix
    fix = f1.fix or f2.fix

    return Finding(
        severity=severity,
        location=location,
        issue=issue,
        evidence=evidence,
        fix=fix,
        confidence="HIGH",  # Both providers agreed
        source="consensus",
    )


def _severity_rank(severity: Severity) -> int:
    """Get numeric rank for severity comparison."""
    return {
        Severity.CRITICAL: 4,
        Severity.HIGH: 3,
        Severity.MEDIUM: 2,
        Severity.LOW: 1,
    }.get(severity, 0)


def compare_findings(
    findings_1: list[Finding],
    findings_2: list[Finding],
    provider_1_name: str = "provider_1",
    provider_2_name: str = "provider_2",
) -> ComparisonResult:
    """
    Compare findings from two providers to identify consensus and disagreements.

    Args:
        findings_1: Findings from first provider.
        findings_2: Findings from second provider.
        provider_1_name: Name of first provider (for tagging).
        provider_2_name: Name of second provider (for tagging).

    Returns:
        ComparisonResult with agreed and disagreed findings.
    """
    agreed: list[Finding] = []
    disagreed: list[Finding] = []
    matched_from_2: set[int] = set()

    # Find matches from provider 1's perspective
    for f1 in findings_1:
        match = find_similar_finding(f1, findings_2)
        if match:
            # Both providers found this
            merged = merge_findings(f1, match)
            agreed.append(merged)
            # Track which findings from provider 2 were matched
            for i, f2 in enumerate(findings_2):
                if f2 is match:
                    matched_from_2.add(i)
                    break
        else:
            # Only provider 1 found this
            f1.source = f"{provider_1_name}_only"
            disagreed.append(f1)

    # Add unmatched findings from provider 2
    for i, f2 in enumerate(findings_2):
        if i not in matched_from_2:
            f2.source = f"{provider_2_name}_only"
            disagreed.append(f2)

    # Calculate confidence score
    total = len(agreed) + len(disagreed)
    confidence_score = len(agreed) / total if total > 0 else 1.0

    return ComparisonResult(
        agreed_findings=agreed,
        disagreed_findings=disagreed,
        confidence_score=confidence_score,
    )


def check_no_issues_response(raw_output: str) -> bool:
    """
    Check if the response indicates no issues were found.

    Args:
        raw_output: Raw text output from provider.

    Returns:
        True if the response indicates no significant issues.
    """
    no_issue_patterns = [
        r"no\s+significant\s+issues?\s+found",
        r"no\s+critical\s+issues?",
        r"code\s+looks\s+good",
        r"lgtm",
        r"no\s+major\s+concerns?",
        r"well[\s-]structured",
        r"no\s+issues?\s+(?:identified|detected|found)",
    ]

    lower_output = raw_output.lower()
    for pattern in no_issue_patterns:
        if re.search(pattern, lower_output):
            return True

    return False
