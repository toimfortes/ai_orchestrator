"""Incremental review system for micro-reviews during implementation.

This module enables any reviewing agent to review code changes at granular levels
(file, commit, function) as they are implemented, providing immediate feedback
to the coding agent.

Architecture:
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Coding  │────▶│ Detect  │────▶│ Review  │──┐
    │ Agent   │     │ Changes │     │ Agent   │  │
    └─────────┘     └─────────┘     └─────────┘  │
         ▲                                       │
         │              Feedback                 │
         └───────────────────────────────────────┘

Usage:
    reviewer = IncrementalReviewer(
        project_path=Path("/path/to/project"),
        review_agent="gemini",
        granularity="file",
    )

    # During implementation
    changes = await reviewer.detect_changes()
    if changes.has_significant_changes:
        feedback = await reviewer.quick_review(changes)
        if feedback.has_issues:
            # Feed back to coding agent
            ...
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_orchestrator.cli_adapters.base import CLIAdapter

logger = logging.getLogger(__name__)


class ReviewGranularity(str, Enum):
    """Granularity level for incremental reviews."""
    FILE = "file"           # Review after each file changes
    COMMIT = "commit"       # Review after each git commit
    HUNK = "hunk"           # Review each diff hunk (fine-grained)
    TIME = "time"           # Review every N seconds


class IssueSeverity(str, Enum):
    """Severity of issues found in review."""
    CRITICAL = "critical"   # Must fix immediately (security, crash)
    HIGH = "high"           # Should fix before continuing
    MEDIUM = "medium"       # Fix before final review
    LOW = "low"             # Nice to have
    INFO = "info"           # Informational only


@dataclass
class FileChange:
    """Represents a change to a single file."""
    path: str
    change_type: str  # added, modified, deleted, renamed
    additions: int = 0
    deletions: int = 0
    diff_content: str = ""

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions


@dataclass
class ChangeSet:
    """Collection of detected changes."""
    files: list[FileChange] = field(default_factory=list)
    commit_hash: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    base_ref: str = "HEAD"

    @property
    def has_changes(self) -> bool:
        return len(self.files) > 0

    @property
    def total_lines_changed(self) -> int:
        return sum(f.total_changes for f in self.files)

    def has_significant_changes(self, threshold: int = 10) -> bool:
        """Check if changes exceed significance threshold."""
        return self.total_lines_changed >= threshold

    def get_combined_diff(self) -> str:
        """Get combined diff content for all files."""
        parts = []
        for f in self.files:
            if f.diff_content:
                parts.append(f"### {f.path} ({f.change_type})\n{f.diff_content}")
        return "\n\n".join(parts)

    def summary(self) -> str:
        """Human-readable summary of changes."""
        if not self.files:
            return "No changes detected"

        lines = [f"Changes detected ({len(self.files)} files, {self.total_lines_changed} lines):"]
        for f in self.files:
            lines.append(f"  {f.change_type}: {f.path} (+{f.additions}/-{f.deletions})")
        return "\n".join(lines)


@dataclass
class ReviewIssue:
    """Single issue found during review."""
    severity: IssueSeverity
    message: str
    file_path: str | None = None
    line_number: int | None = None
    suggestion: str | None = None


@dataclass
class QuickReviewResult:
    """Result of a quick incremental review."""
    issues: list[ReviewIssue] = field(default_factory=list)
    reviewer: str = ""
    review_time_seconds: float = 0.0
    changes_reviewed: ChangeSet | None = None
    raw_output: str = ""

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def has_blocking_issues(self) -> bool:
        """Issues that should block further implementation."""
        blocking = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}
        return any(i.severity in blocking for i in self.issues)

    def get_feedback_for_agent(self) -> str:
        """Format issues as feedback for the coding agent."""
        if not self.issues:
            return "No issues found in the recent changes."

        lines = ["Issues found in recent changes that need immediate attention:"]
        for i, issue in enumerate(self.issues, 1):
            loc = ""
            if issue.file_path:
                loc = f" in {issue.file_path}"
                if issue.line_number:
                    loc += f":{issue.line_number}"

            lines.append(f"\n{i}. [{issue.severity.value.upper()}]{loc}")
            lines.append(f"   {issue.message}")
            if issue.suggestion:
                lines.append(f"   Suggestion: {issue.suggestion}")

        return "\n".join(lines)


class ChangeDetector:
    """Detects code changes using git."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._last_commit_hash: str | None = None
        self._baseline_set = False

    async def set_baseline(self) -> None:
        """Set current state as baseline for change detection."""
        self._last_commit_hash = await self._get_current_commit()
        # Stage all changes to establish baseline
        await self._run_git("add", "-A")
        self._baseline_set = True
        logger.info("Change detection baseline set at %s", self._last_commit_hash)

    async def detect_changes(
        self,
        granularity: ReviewGranularity = ReviewGranularity.FILE,
    ) -> ChangeSet:
        """
        Detect changes since last check.

        Args:
            granularity: Level of change detection.

        Returns:
            ChangeSet with detected changes.
        """
        if granularity == ReviewGranularity.COMMIT:
            return await self._detect_commit_changes()
        else:
            # FILE, HUNK, TIME all use working directory diff
            return await self._detect_working_changes()

    async def _detect_working_changes(self) -> ChangeSet:
        """Detect uncommitted changes in working directory."""
        # Get status of changed files
        status_output = await self._run_git(
            "status", "--porcelain", "-uall"
        )

        if not status_output.strip():
            return ChangeSet()

        files = []
        for line in status_output.strip().split("\n"):
            if not line:
                continue

            status = line[:2].strip()
            file_path = line[3:].strip()

            # Handle renamed files
            if " -> " in file_path:
                file_path = file_path.split(" -> ")[1]

            change_type = self._parse_status(status)

            # Get diff for this file
            diff_content = ""
            additions = 0
            deletions = 0

            if change_type != "deleted":
                try:
                    diff_output = await self._run_git(
                        "diff", "--", file_path
                    )
                    if not diff_output:
                        # File might be untracked
                        diff_output = await self._run_git(
                            "diff", "--no-index", "/dev/null", file_path,
                            check=False
                        )
                    diff_content = diff_output
                    additions, deletions = self._count_diff_lines(diff_output)
                except Exception as e:
                    logger.debug("Could not get diff for %s: %s", file_path, e)

            files.append(FileChange(
                path=file_path,
                change_type=change_type,
                additions=additions,
                deletions=deletions,
                diff_content=diff_content,
            ))

        return ChangeSet(files=files)

    async def _detect_commit_changes(self) -> ChangeSet:
        """Detect changes in new commits since baseline."""
        current_commit = await self._get_current_commit()

        if current_commit == self._last_commit_hash:
            return ChangeSet()

        # Get diff between commits
        diff_output = await self._run_git(
            "diff", "--stat",
            self._last_commit_hash or "HEAD~1",
            current_commit,
        )

        # Parse diff stat
        files = []
        for line in diff_output.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                file_path = parts[0].strip()
                stats = parts[1].strip() if len(parts) > 1 else ""

                # Parse stats like "10 +" or "5 -" or "3 +, 2 -"
                additions = stats.count("+")
                deletions = stats.count("-")

                files.append(FileChange(
                    path=file_path,
                    change_type="modified",
                    additions=additions,
                    deletions=deletions,
                ))

        # Update baseline
        self._last_commit_hash = current_commit

        return ChangeSet(
            files=files,
            commit_hash=current_commit,
            base_ref=self._last_commit_hash or "HEAD~1",
        )

    def _parse_status(self, status: str) -> str:
        """Parse git status code to change type."""
        if "A" in status or "?" in status:
            return "added"
        elif "D" in status:
            return "deleted"
        elif "R" in status:
            return "renamed"
        else:
            return "modified"

    def _count_diff_lines(self, diff: str) -> tuple[int, int]:
        """Count additions and deletions in diff."""
        additions = 0
        deletions = 0
        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions

    async def _get_current_commit(self) -> str:
        """Get current commit hash."""
        return (await self._run_git("rev-parse", "HEAD")).strip()

    async def _run_git(self, *args: str, check: bool = True) -> str:
        """Run a git command and return output."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if check and proc.returncode != 0:
                logger.warning("Git command failed: git %s", " ".join(args))
                return ""

            return stdout.decode("utf-8", errors="replace")
        except Exception as e:
            logger.error("Git command error: %s", e, exc_info=True)
            return ""


# Quick review prompts
QUICK_REVIEW_PROMPT = """You are reviewing a small code change. Be concise and focus only on significant issues.

## Recent Code Changes
{diff}

## Review Instructions
1. Look for CRITICAL issues only:
   - Security vulnerabilities (injection, auth bypass, data exposure)
   - Definite bugs (null pointer, infinite loop, resource leak)
   - Breaking changes

2. Look for HIGH issues:
   - Logic errors
   - Missing error handling for likely failures
   - Performance problems (N+1 queries, blocking calls in async)

3. IGNORE for now (will be caught in full review):
   - Style/formatting
   - Minor naming issues
   - Documentation
   - Test coverage

## Output Format
If issues found, list them as:
SEVERITY: file:line - description

If no significant issues:
LGTM - no critical issues found

Be brief. This is a quick check, not a full review."""


class IncrementalReviewer:
    """
    Performs incremental micro-reviews during implementation.

    Uses any available reviewing agent to check code changes at granular
    levels, providing immediate feedback to the coding agent.
    """

    def __init__(
        self,
        project_path: Path,
        review_agent: str = "gemini",
        granularity: ReviewGranularity = ReviewGranularity.FILE,
        min_lines_threshold: int = 10,
        adapters: dict[str, "CLIAdapter"] | None = None,
    ):
        self.project_path = project_path
        self.review_agent = review_agent
        self.granularity = granularity
        self.min_lines_threshold = min_lines_threshold
        self.adapters = adapters or {}

        self.change_detector = ChangeDetector(project_path)
        self._review_count = 0
        self._issues_found_total = 0

    async def initialize(self) -> None:
        """Initialize the reviewer and set baseline."""
        await self.change_detector.set_baseline()

    async def detect_changes(self) -> ChangeSet:
        """Detect changes since last review."""
        return await self.change_detector.detect_changes(self.granularity)

    async def quick_review(
        self,
        changes: ChangeSet,
        reviewer_override: str | None = None,
    ) -> QuickReviewResult:
        """
        Perform a quick review of the changes.

        Args:
            changes: Changes to review.
            reviewer_override: Override the default review agent.

        Returns:
            QuickReviewResult with any issues found.
        """
        reviewer = reviewer_override or self.review_agent

        if not changes.has_significant_changes(self.min_lines_threshold):
            logger.debug(
                "Skipping review - changes below threshold (%d < %d lines)",
                changes.total_lines_changed,
                self.min_lines_threshold,
            )
            return QuickReviewResult(
                reviewer=reviewer,
                changes_reviewed=changes,
            )

        adapter = self.adapters.get(reviewer)
        if not adapter:
            logger.warning("Review agent %s not available", reviewer)
            return QuickReviewResult(
                reviewer=reviewer,
                changes_reviewed=changes,
            )

        # Build review prompt
        diff = changes.get_combined_diff()
        prompt = QUICK_REVIEW_PROMPT.format(diff=diff[:50000])  # Limit diff size

        start_time = datetime.now(UTC)

        try:
            result = await adapter.invoke(
                prompt,
                timeout_seconds=120,  # Quick review should be fast
                working_dir=str(self.project_path),
            )

            duration = (datetime.now(UTC) - start_time).total_seconds()

            if result.success:
                issues = self._parse_review_output(result.output)
                self._review_count += 1
                self._issues_found_total += len(issues)

                return QuickReviewResult(
                    issues=issues,
                    reviewer=reviewer,
                    review_time_seconds=duration,
                    changes_reviewed=changes,
                    raw_output=result.output,
                )
            else:
                logger.warning("Quick review failed: %s", result.stderr)
                return QuickReviewResult(
                    reviewer=reviewer,
                    review_time_seconds=duration,
                    changes_reviewed=changes,
                )

        except Exception as e:
            logger.error("Quick review error: %s", e, exc_info=True)
            return QuickReviewResult(
                reviewer=reviewer,
                changes_reviewed=changes,
            )

    async def review_and_feedback(
        self,
        changes: ChangeSet | None = None,
    ) -> tuple[QuickReviewResult, str | None]:
        """
        Detect changes, review them, and generate feedback if needed.

        Returns:
            Tuple of (review_result, feedback_for_agent or None)
        """
        if changes is None:
            changes = await self.detect_changes()

        if not changes.has_changes:
            return QuickReviewResult(), None

        result = await self.quick_review(changes)

        if result.has_blocking_issues:
            feedback = result.get_feedback_for_agent()
            return result, feedback

        return result, None

    def _parse_review_output(self, output: str) -> list[ReviewIssue]:
        """Parse review output into structured issues."""
        issues: list[ReviewIssue] = []

        # Check for LGTM
        if "LGTM" in output.upper() or "no critical issues" in output.lower():
            return issues

        # Parse issue lines
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse "SEVERITY: file:line - description" format
            severity = None
            for sev in IssueSeverity:
                if line.upper().startswith(sev.value.upper()):
                    severity = sev
                    line = line[len(sev.value):].strip()
                    if line.startswith(":"):
                        line = line[1:].strip()
                    break

            if severity is None:
                # Try to infer severity from keywords
                if any(kw in line.lower() for kw in ["security", "injection", "vulnerability", "crash"]):
                    severity = IssueSeverity.CRITICAL
                elif any(kw in line.lower() for kw in ["bug", "error", "wrong", "incorrect"]):
                    severity = IssueSeverity.HIGH
                elif any(kw in line.lower() for kw in ["should", "consider", "might"]):
                    severity = IssueSeverity.MEDIUM
                else:
                    continue  # Skip lines we can't parse

            # Try to extract file:line
            file_path = None
            line_number = None

            if " - " in line:
                location, message = line.split(" - ", 1)
                if ":" in location:
                    parts = location.split(":")
                    file_path = parts[0].strip()
                    if len(parts) > 1 and parts[1].isdigit():
                        line_number = int(parts[1])
            else:
                message = line

            issues.append(ReviewIssue(
                severity=severity,
                message=message.strip(),
                file_path=file_path,
                line_number=line_number,
            ))

        return issues

    def get_stats(self) -> dict[str, int | float]:
        """Get review statistics."""
        return {
            "total_reviews": self._review_count,
            "total_issues_found": self._issues_found_total,
            "avg_issues_per_review": (
                self._issues_found_total / self._review_count
                if self._review_count > 0 else 0
            ),
        }
