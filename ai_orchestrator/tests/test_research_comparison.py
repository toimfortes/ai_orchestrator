"""Tests for research finding extraction and comparison logic."""

import pytest

from ai_orchestrator.research.comparison import (
    extract_findings,
    find_similar_finding,
    compare_findings,
    check_no_issues_response,
    merge_findings,
)
from ai_orchestrator.research.models import Finding, Severity


class TestExtractFindings:
    """Tests for extract_findings function."""

    def test_extracts_structured_finding(self):
        """Test extraction of well-formatted finding."""
        raw_output = """
### Findings

- **[CRITICAL]** orchestrator.py:1569 - Race condition in gather
  - Issue: asyncio.gather swallows CancelledError silently
  - Evidence: `results = await asyncio.gather(*tasks, return_exceptions=True)`
  - Fix: Use TaskGroup or handle CancelledError explicitly
"""
        findings = extract_findings(raw_output)

        assert len(findings) == 1
        assert findings[0].severity == Severity.CRITICAL
        assert "orchestrator.py" in findings[0].location
        assert "gather" in findings[0].issue.lower() or "race" in findings[0].issue.lower()

    def test_extracts_multiple_findings(self):
        """Test extraction of multiple findings."""
        raw_output = """
### Findings

- **[CRITICAL]** state_manager.py:71 - Directory sync missing
  - Issue: No fsync on Windows
  - Fix: Add Windows equivalent

- **[HIGH]** retry_utils.py:98 - Jitter too narrow
  - Issue: Jitter range may not prevent thundering herd
  - Fix: Increase jitter factor

- **[MEDIUM]** orchestrator.py:200 - Magic number
  - Issue: Hardcoded timeout value
  - Fix: Move to configuration
"""
        findings = extract_findings(raw_output)

        assert len(findings) == 3
        assert findings[0].severity == Severity.CRITICAL
        assert findings[1].severity == Severity.HIGH
        assert findings[2].severity == Severity.MEDIUM

    def test_extracts_without_markdown_bold(self):
        """Test extraction without markdown formatting."""
        raw_output = """
[CRITICAL] orchestrator.py:100 - Security issue
Issue: Command injection possible
Fix: Use subprocess with list args
"""
        findings = extract_findings(raw_output)

        assert len(findings) >= 1
        assert findings[0].severity == Severity.CRITICAL

    def test_handles_empty_output(self):
        """Test handling of empty output."""
        findings = extract_findings("")
        assert len(findings) == 0

    def test_handles_no_findings_text(self):
        """Test handling of 'no issues found' text."""
        raw_output = "The code looks good. No significant issues found."
        findings = extract_findings(raw_output)
        assert len(findings) == 0


class TestCheckNoIssuesResponse:
    """Tests for check_no_issues_response function."""

    def test_detects_no_issues_found(self):
        """Test detection of 'no issues found' responses."""
        assert check_no_issues_response("No significant issues found")
        assert check_no_issues_response("LGTM")
        assert check_no_issues_response("The code looks good")
        assert check_no_issues_response("No major concerns identified")

    def test_does_not_flag_actual_findings(self):
        """Test that actual findings are not flagged as clean."""
        assert not check_no_issues_response("[CRITICAL] security issue found")
        assert not check_no_issues_response("There is a race condition")


class TestFindSimilarFinding:
    """Tests for find_similar_finding function."""

    def test_matches_same_file_nearby_line(self):
        """Test matching by file and nearby line number."""
        target = Finding(
            severity=Severity.HIGH,
            location="orchestrator.py:100",
            issue="Race condition in async code",
        )
        candidates = [
            Finding(
                severity=Severity.CRITICAL,
                location="orchestrator.py:105",
                issue="Concurrent access issue",
            ),
            Finding(
                severity=Severity.LOW,
                location="other.py:100",
                issue="Different file issue",
            ),
        ]

        match = find_similar_finding(target, candidates)

        assert match is not None
        assert "orchestrator.py" in match.location

    def test_matches_by_text_similarity(self):
        """Test matching by similar issue text."""
        target = Finding(
            severity=Severity.HIGH,
            location="unknown",
            issue="asyncio.gather swallows CancelledError",
        )
        candidates = [
            Finding(
                severity=Severity.CRITICAL,
                location="file.py:50",
                issue="gather() silently swallows CancelledError exceptions",
            ),
        ]

        match = find_similar_finding(target, candidates)

        assert match is not None

    def test_no_match_for_different_issues(self):
        """Test that unrelated issues don't match."""
        target = Finding(
            severity=Severity.HIGH,
            location="file_a.py:100",
            issue="Memory leak in cache",
        )
        candidates = [
            Finding(
                severity=Severity.HIGH,
                location="file_b.py:200",
                issue="SQL injection vulnerability",
            ),
        ]

        match = find_similar_finding(target, candidates)

        assert match is None


class TestMergeFindings:
    """Tests for merge_findings function."""

    def test_uses_higher_severity(self):
        """Test that merged finding uses higher severity."""
        f1 = Finding(severity=Severity.HIGH, location="file.py:100", issue="Issue A")
        f2 = Finding(severity=Severity.CRITICAL, location="file.py:105", issue="Issue B")

        merged = merge_findings(f1, f2)

        assert merged.severity == Severity.CRITICAL

    def test_prefers_specific_location(self):
        """Test that merged finding uses more specific location."""
        f1 = Finding(severity=Severity.HIGH, location="file.py:100", issue="Issue")
        f2 = Finding(severity=Severity.HIGH, location="file.py", issue="Issue")

        merged = merge_findings(f1, f2)

        assert merged.location == "file.py:100"

    def test_combines_available_details(self):
        """Test that merged finding combines evidence and fix."""
        f1 = Finding(
            severity=Severity.HIGH,
            location="file.py:100",
            issue="Issue",
            evidence="code snippet",
            fix=None,
        )
        f2 = Finding(
            severity=Severity.HIGH,
            location="file.py:100",
            issue="Issue",
            evidence=None,
            fix="suggested fix",
        )

        merged = merge_findings(f1, f2)

        assert merged.evidence == "code snippet"
        assert merged.fix == "suggested fix"


class TestCompareFindings:
    """Tests for compare_findings function."""

    def test_identifies_agreed_findings(self):
        """Test that matching findings are marked as agreed."""
        findings_1 = [
            Finding(severity=Severity.CRITICAL, location="file.py:100", issue="Security issue"),
        ]
        findings_2 = [
            Finding(severity=Severity.CRITICAL, location="file.py:105", issue="Security vulnerability"),
        ]

        result = compare_findings(findings_1, findings_2, "provider1", "provider2")

        assert len(result.agreed_findings) == 1
        assert len(result.disagreed_findings) == 0
        assert result.confidence_score == 1.0

    def test_identifies_disagreed_findings(self):
        """Test that non-matching findings are marked as disagreed."""
        findings_1 = [
            Finding(severity=Severity.HIGH, location="file_a.py:100", issue="Memory leak"),
        ]
        findings_2 = [
            Finding(severity=Severity.HIGH, location="file_b.py:200", issue="SQL injection"),
        ]

        result = compare_findings(findings_1, findings_2, "gemini", "openai")

        assert len(result.agreed_findings) == 0
        assert len(result.disagreed_findings) == 2
        assert result.confidence_score == 0.0

        # Check source tagging
        sources = [f.source for f in result.disagreed_findings]
        assert "gemini_only" in sources
        assert "openai_only" in sources

    def test_mixed_agreement_disagreement(self):
        """Test scenario with both agreed and disagreed findings."""
        findings_1 = [
            Finding(severity=Severity.CRITICAL, location="file.py:100", issue="Race condition in asyncio gather"),
            Finding(severity=Severity.HIGH, location="other.py:50", issue="Memory leak in cache invalidation"),
        ]
        findings_2 = [
            Finding(severity=Severity.CRITICAL, location="file.py:105", issue="Race condition in asyncio code"),
            Finding(severity=Severity.MEDIUM, location="different.py:75", issue="SQL injection vulnerability in query builder"),
        ]

        result = compare_findings(findings_1, findings_2, "p1", "p2")

        assert len(result.agreed_findings) == 1
        assert len(result.disagreed_findings) == 2
        assert 0 < result.confidence_score < 1

    def test_empty_findings(self):
        """Test comparison with empty findings lists."""
        result = compare_findings([], [], "p1", "p2")

        assert len(result.agreed_findings) == 0
        assert len(result.disagreed_findings) == 0
        assert result.confidence_score == 1.0  # Full agreement (nothing to disagree on)

    def test_confidence_score_calculation(self):
        """Test that confidence score is calculated correctly."""
        # 2 agreed, 1 disagreed = 2/3 = 0.67
        findings_1 = [
            Finding(severity=Severity.HIGH, location="a.py:1", issue="Issue A"),
            Finding(severity=Severity.HIGH, location="b.py:2", issue="Issue B"),
            Finding(severity=Severity.LOW, location="c.py:3", issue="Only in 1"),
        ]
        findings_2 = [
            Finding(severity=Severity.HIGH, location="a.py:5", issue="Issue A found"),
            Finding(severity=Severity.HIGH, location="b.py:8", issue="Issue B found"),
        ]

        result = compare_findings(findings_1, findings_2, "p1", "p2")

        assert len(result.agreed_findings) == 2
        assert len(result.disagreed_findings) == 1
        assert result.confidence_score == pytest.approx(2/3, rel=0.01)
