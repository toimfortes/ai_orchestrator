"""Tests for incremental reviewer functionality."""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_orchestrator.reviewing.incremental_reviewer import (
    ChangeDetector,
    ChangeSet,
    FileChange,
    IncrementalReviewer,
    IssueSeverity,
    QuickReviewResult,
    ReviewGranularity,
    ReviewIssue,
)


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_total_changes(self):
        """Test total changes calculation."""
        change = FileChange(
            path="test.py",
            change_type="modified",
            additions=10,
            deletions=5,
        )
        assert change.total_changes == 15


class TestChangeSet:
    """Tests for ChangeSet dataclass."""

    def test_has_changes_empty(self):
        """Test has_changes with no files."""
        cs = ChangeSet()
        assert cs.has_changes is False

    def test_has_changes_with_files(self):
        """Test has_changes with files."""
        cs = ChangeSet(files=[
            FileChange(path="test.py", change_type="modified")
        ])
        assert cs.has_changes is True

    def test_total_lines_changed(self):
        """Test total lines calculation."""
        cs = ChangeSet(files=[
            FileChange(path="a.py", change_type="modified", additions=10, deletions=5),
            FileChange(path="b.py", change_type="added", additions=20, deletions=0),
        ])
        assert cs.total_lines_changed == 35

    def test_has_significant_changes(self):
        """Test significance threshold."""
        cs = ChangeSet(files=[
            FileChange(path="a.py", change_type="modified", additions=5, deletions=3),
        ])
        assert cs.has_significant_changes(threshold=10) is False
        assert cs.has_significant_changes(threshold=5) is True

    def test_get_combined_diff(self):
        """Test combined diff generation."""
        cs = ChangeSet(files=[
            FileChange(
                path="a.py",
                change_type="modified",
                diff_content="+line1\n-line2",
            ),
            FileChange(
                path="b.py",
                change_type="added",
                diff_content="+new file",
            ),
        ])
        combined = cs.get_combined_diff()
        assert "a.py" in combined
        assert "b.py" in combined
        assert "+line1" in combined

    def test_summary(self):
        """Test summary generation."""
        cs = ChangeSet(files=[
            FileChange(
                path="test.py",
                change_type="modified",
                additions=10,
                deletions=5,
            )
        ])
        summary = cs.summary()
        assert "test.py" in summary
        assert "+10" in summary
        assert "-5" in summary


class TestQuickReviewResult:
    """Tests for QuickReviewResult dataclass."""

    def test_has_issues_empty(self):
        """Test has_issues with no issues."""
        result = QuickReviewResult()
        assert result.has_issues is False

    def test_has_issues_with_issues(self):
        """Test has_issues with issues."""
        result = QuickReviewResult(issues=[
            ReviewIssue(severity=IssueSeverity.LOW, message="Minor issue")
        ])
        assert result.has_issues is True

    def test_has_critical_issues(self):
        """Test critical issue detection."""
        result = QuickReviewResult(issues=[
            ReviewIssue(severity=IssueSeverity.MEDIUM, message="Medium issue"),
            ReviewIssue(severity=IssueSeverity.CRITICAL, message="Critical bug"),
        ])
        assert result.has_critical_issues is True

    def test_has_blocking_issues(self):
        """Test blocking issue detection."""
        # No blocking
        result1 = QuickReviewResult(issues=[
            ReviewIssue(severity=IssueSeverity.LOW, message="Low"),
            ReviewIssue(severity=IssueSeverity.MEDIUM, message="Medium"),
        ])
        assert result1.has_blocking_issues is False

        # Has blocking (HIGH)
        result2 = QuickReviewResult(issues=[
            ReviewIssue(severity=IssueSeverity.HIGH, message="High"),
        ])
        assert result2.has_blocking_issues is True

        # Has blocking (CRITICAL)
        result3 = QuickReviewResult(issues=[
            ReviewIssue(severity=IssueSeverity.CRITICAL, message="Critical"),
        ])
        assert result3.has_blocking_issues is True

    def test_get_feedback_for_agent_no_issues(self):
        """Test feedback with no issues."""
        result = QuickReviewResult()
        feedback = result.get_feedback_for_agent()
        assert "No issues" in feedback

    def test_get_feedback_for_agent_with_issues(self):
        """Test feedback with issues."""
        result = QuickReviewResult(issues=[
            ReviewIssue(
                severity=IssueSeverity.HIGH,
                message="SQL injection vulnerability",
                file_path="db.py",
                line_number=42,
                suggestion="Use parameterized queries",
            )
        ])
        feedback = result.get_feedback_for_agent()
        assert "HIGH" in feedback
        assert "db.py:42" in feedback
        assert "SQL injection" in feedback
        assert "parameterized queries" in feedback


class TestChangeDetector:
    """Tests for ChangeDetector class."""

    @pytest.fixture
    def detector(self, tmp_path):
        """Create detector with temp path."""
        return ChangeDetector(tmp_path)

    @pytest.mark.asyncio
    async def test_run_git_success(self, detector):
        """Test running git command."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
            mock_exec.return_value = mock_proc

            result = await detector._run_git("status")
            assert result == "output"

    @pytest.mark.asyncio
    async def test_run_git_failure(self, detector):
        """Test git command failure handling."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))
            mock_exec.return_value = mock_proc

            result = await detector._run_git("invalid")
            assert result == ""

    def test_parse_status(self, detector):
        """Test git status parsing."""
        assert detector._parse_status("A") == "added"
        assert detector._parse_status("?") == "added"
        assert detector._parse_status("D") == "deleted"
        assert detector._parse_status("R") == "renamed"
        assert detector._parse_status("M") == "modified"
        assert detector._parse_status("") == "modified"

    def test_count_diff_lines(self, detector):
        """Test diff line counting."""
        diff = """+new line
-removed line
+another new
 unchanged
+++ header
--- header"""
        additions, deletions = detector._count_diff_lines(diff)
        assert additions == 2
        assert deletions == 1


class TestIncrementalReviewer:
    """Tests for IncrementalReviewer class."""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock CLI adapter."""
        adapter = MagicMock()
        adapter.name = "gemini"
        adapter.invoke = AsyncMock()
        return adapter

    @pytest.fixture
    def reviewer(self, tmp_path, mock_adapter):
        """Create reviewer with mocks."""
        return IncrementalReviewer(
            project_path=tmp_path,
            review_agent="gemini",
            granularity=ReviewGranularity.FILE,
            min_lines_threshold=5,
            adapters={"gemini": mock_adapter},
        )

    @pytest.mark.asyncio
    async def test_quick_review_below_threshold(self, reviewer):
        """Test that small changes skip review."""
        changes = ChangeSet(files=[
            FileChange(path="test.py", change_type="modified", additions=2, deletions=1)
        ])

        result = await reviewer.quick_review(changes)

        # Should skip - below threshold
        assert not result.has_issues
        assert result.changes_reviewed == changes

    @pytest.mark.asyncio
    async def test_quick_review_adapter_not_available(self, tmp_path):
        """Test review when adapter not available."""
        reviewer = IncrementalReviewer(
            project_path=tmp_path,
            review_agent="nonexistent",
            adapters={},
        )

        changes = ChangeSet(files=[
            FileChange(path="test.py", change_type="modified", additions=20, deletions=10)
        ])

        result = await reviewer.quick_review(changes)
        assert not result.has_issues

    @pytest.mark.asyncio
    async def test_quick_review_success(self, reviewer, mock_adapter):
        """Test successful quick review."""
        from ai_orchestrator.cli_adapters.base import CLIResult, CLIStatus

        mock_adapter.invoke.return_value = CLIResult(
            cli_name="gemini",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="LGTM - no critical issues found",
            stderr="",
        )

        changes = ChangeSet(files=[
            FileChange(
                path="test.py",
                change_type="modified",
                additions=20,
                deletions=10,
                diff_content="+new code\n-old code",
            )
        ])

        result = await reviewer.quick_review(changes)

        assert result.reviewer == "gemini"
        assert not result.has_issues  # LGTM means no issues
        mock_adapter.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_quick_review_finds_issues(self, reviewer, mock_adapter):
        """Test review that finds issues."""
        from ai_orchestrator.cli_adapters.base import CLIResult, CLIStatus

        mock_adapter.invoke.return_value = CLIResult(
            cli_name="gemini",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="""CRITICAL: db.py:42 - SQL injection vulnerability
HIGH: auth.py:15 - Missing authentication check
MEDIUM: utils.py - Consider error handling""",
            stderr="",
        )

        changes = ChangeSet(files=[
            FileChange(path="db.py", change_type="modified", additions=20, deletions=5)
        ])

        result = await reviewer.quick_review(changes)

        assert result.has_issues
        assert result.has_critical_issues
        assert len(result.issues) >= 2

    @pytest.mark.asyncio
    async def test_review_and_feedback_no_changes(self, reviewer):
        """Test review_and_feedback with no changes."""
        with patch.object(reviewer.change_detector, "detect_changes") as mock_detect:
            mock_detect.return_value = ChangeSet()

            result, feedback = await reviewer.review_and_feedback()

            assert feedback is None

    @pytest.mark.asyncio
    async def test_review_and_feedback_with_blocking_issues(self, reviewer, mock_adapter):
        """Test review_and_feedback returns feedback for blocking issues."""
        from ai_orchestrator.cli_adapters.base import CLIResult, CLIStatus

        mock_adapter.invoke.return_value = CLIResult(
            cli_name="gemini",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="CRITICAL: security.py:10 - Hardcoded password",
            stderr="",
        )

        changes = ChangeSet(files=[
            FileChange(path="security.py", change_type="modified", additions=20, deletions=0)
        ])

        result, feedback = await reviewer.review_and_feedback(changes)

        assert result.has_blocking_issues
        assert feedback is not None
        assert "CRITICAL" in feedback

    def test_parse_review_output_lgtm(self, reviewer):
        """Test parsing LGTM output."""
        issues = reviewer._parse_review_output("LGTM - no critical issues found")
        assert len(issues) == 0

    def test_parse_review_output_with_issues(self, reviewer):
        """Test parsing output with issues."""
        output = """CRITICAL: auth.py:10 - Security vulnerability
HIGH: db.py:20 - Missing null check
Something about a bug in the code"""

        issues = reviewer._parse_review_output(output)

        assert len(issues) >= 2
        assert any(i.severity == IssueSeverity.CRITICAL for i in issues)
        assert any(i.severity == IssueSeverity.HIGH for i in issues)

    def test_get_stats(self, reviewer):
        """Test stats retrieval."""
        reviewer._review_count = 5
        reviewer._issues_found_total = 12

        stats = reviewer.get_stats()

        assert stats["total_reviews"] == 5
        assert stats["total_issues_found"] == 12
        assert stats["avg_issues_per_review"] == pytest.approx(2.4)


class TestReviewGranularity:
    """Tests for ReviewGranularity enum."""

    def test_values(self):
        """Test enum values."""
        assert ReviewGranularity.FILE.value == "file"
        assert ReviewGranularity.COMMIT.value == "commit"
        assert ReviewGranularity.HUNK.value == "hunk"
        assert ReviewGranularity.TIME.value == "time"

    def test_from_string(self):
        """Test creating from string."""
        assert ReviewGranularity("file") == ReviewGranularity.FILE
        assert ReviewGranularity("commit") == ReviewGranularity.COMMIT
