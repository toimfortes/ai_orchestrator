"""Feedback classifier for structured review analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Severity level of identified issues."""

    CRITICAL = "critical"  # Security vulnerability, guaranteed crash, data loss
    HIGH = "high"  # Significant bug, performance issue, maintainability problem
    MEDIUM = "medium"  # Code smell, minor bug, suboptimal pattern
    LOW = "low"  # Suggestion, style issue, nice-to-have improvement


class IssueCategory(str, Enum):
    """Category of identified issues."""

    SECURITY = "security"  # Auth, injection, data exposure
    PERFORMANCE = "performance"  # Speed, memory, scalability
    CORRECTNESS = "correctness"  # Bugs, logic errors, edge cases
    MAINTAINABILITY = "maintainability"  # Code quality, readability
    ARCHITECTURE = "architecture"  # Design patterns, structure
    TESTING = "testing"  # Test coverage, test quality
    DOCUMENTATION = "documentation"  # Comments, docstrings, README
    STYLE = "style"  # Formatting, naming conventions
    OTHER = "other"  # Uncategorized


class Actionability(str, Enum):
    """How actionable the feedback is."""

    IMMEDIATE = "immediate"  # Must fix before merge
    SHOULD_FIX = "should_fix"  # Should fix, but not blocking
    NICE_TO_HAVE = "nice_to_have"  # Optional improvement


@dataclass
class ClassifiedFeedback:
    """A single piece of classified feedback."""

    original_text: str
    severity: IssueSeverity
    category: IssueCategory
    actionability: Actionability
    is_blocker: bool
    file_path: str | None = None
    line_number: int | None = None
    fix_suggestion: str | None = None
    confidence: float = 1.0  # Classifier confidence (0-1)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def location(self) -> str | None:
        """Get formatted location string."""
        if self.file_path and self.line_number:
            return f"{self.file_path}:{self.line_number}"
        elif self.file_path:
            return self.file_path
        return None


@dataclass
class ClassificationResult:
    """Result of classifying all feedback from a review."""

    feedback_items: list[ClassifiedFeedback]
    reviewer_name: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for f in self.feedback_items if f.severity == IssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high severity issues."""
        return sum(1 for f in self.feedback_items if f.severity == IssueSeverity.HIGH)

    @property
    def blocker_count(self) -> int:
        """Count of blocking issues."""
        return sum(1 for f in self.feedback_items if f.is_blocker)

    @property
    def has_blockers(self) -> bool:
        """Whether there are any blocking issues."""
        return self.blocker_count > 0

    def get_by_severity(self, severity: IssueSeverity) -> list[ClassifiedFeedback]:
        """Get all feedback items of a specific severity."""
        return [f for f in self.feedback_items if f.severity == severity]

    def get_by_category(self, category: IssueCategory) -> list[ClassifiedFeedback]:
        """Get all feedback items of a specific category."""
        return [f for f in self.feedback_items if f.category == category]


class FeedbackClassifier:
    """
    Classify raw review feedback into structured, actionable categories.

    Uses pattern matching and heuristics to classify feedback without
    requiring additional LLM calls. For more accurate classification,
    consider using LLM-based classification.
    """

    # Keywords for severity detection
    CRITICAL_KEYWORDS = frozenset([
        "vulnerability", "injection", "xss", "csrf", "sql injection",
        "security hole", "data leak", "crash", "data loss", "exploit",
        "remote code execution", "rce", "authentication bypass",
        "critical", "severe", "dangerous", "unsafe"
    ])

    HIGH_KEYWORDS = frozenset([
        "bug", "error", "broken", "incorrect", "wrong", "fail",
        "performance issue", "memory leak", "race condition",
        "high priority", "important", "significant"
    ])

    MEDIUM_KEYWORDS = frozenset([
        "should", "consider", "improve", "refactor", "cleanup",
        "code smell", "anti-pattern", "suboptimal", "inefficient"
    ])

    LOW_KEYWORDS = frozenset([
        "suggestion", "nitpick", "nit", "style", "minor",
        "optional", "nice to have", "could", "might"
    ])

    # Keywords for category detection
    SECURITY_KEYWORDS = frozenset([
        "security", "auth", "permission", "access", "token",
        "injection", "xss", "csrf", "vulnerability", "secret",
        "password", "credential", "encrypt", "sanitize"
    ])

    PERFORMANCE_KEYWORDS = frozenset([
        "performance", "slow", "fast", "speed", "latency",
        "memory", "cpu", "optimize", "cache", "efficient",
        "n+1", "query", "batch", "async"
    ])

    CORRECTNESS_KEYWORDS = frozenset([
        "bug", "error", "incorrect", "wrong", "logic",
        "edge case", "null", "exception", "crash", "fail"
    ])

    TESTING_KEYWORDS = frozenset([
        "test", "coverage", "mock", "fixture", "assert",
        "unit test", "integration", "e2e"
    ])

    ARCHITECTURE_KEYWORDS = frozenset([
        "architecture", "design", "pattern", "structure",
        "dependency", "coupling", "cohesion", "solid"
    ])

    # Location extraction patterns
    LOCATION_PATTERNS = [
        re.compile(r"(?:in|at|file)\s+[`'\"]?([a-zA-Z0-9_/\\.-]+\.(?:py|js|ts|tsx|jsx))[`'\"]?\s*(?:line\s*)?:?\s*(\d+)?", re.I),
        re.compile(r"([a-zA-Z0-9_/\\.-]+\.(?:py|js|ts|tsx|jsx)):(\d+)", re.I),
        re.compile(r"line\s*(\d+)\s+(?:of|in)\s+[`'\"]?([a-zA-Z0-9_/\\.-]+\.(?:py|js|ts|tsx|jsx))[`'\"]?", re.I),
    ]

    # Negation patterns that invalidate keyword matches
    # Pattern: negation word followed by 0-3 words, then keyword
    NEGATION_WORDS = frozenset([
        "no", "not", "none", "never", "neither", "nor",
        "without", "lacking", "missing", "absent",
        "isn't", "aren't", "doesn't", "don't", "didn't",
        "hasn't", "haven't", "hadn't", "won't", "wouldn't",
        "can't", "cannot", "couldn't", "shouldn't",
        "free of", "clear of", "devoid of",
    ])

    # Positive affirmation patterns that strengthen keyword matches
    AFFIRMATION_PATTERNS = frozenset([
        "found", "detected", "discovered", "identified",
        "contains", "has", "have", "is", "are",
        "potential", "possible", "likely",
    ])

    def __init__(
        self,
        custom_keywords: dict[str, set[str]] | None = None,
    ) -> None:
        """
        Initialize classifier.

        Args:
            custom_keywords: Optional custom keywords to extend defaults.
        """
        self.custom_keywords = custom_keywords or {}

    def classify(self, raw_feedback: str, reviewer_name: str = "unknown") -> ClassificationResult:
        """
        Classify raw review feedback into structured items.

        Args:
            raw_feedback: Raw text feedback from a reviewer.
            reviewer_name: Name of the reviewer.

        Returns:
            ClassificationResult with classified feedback items.
        """
        # Split feedback into individual items (by newlines, bullets, or numbers)
        items = self._split_feedback(raw_feedback)

        classified_items = []
        for item in items:
            if not item.strip():
                continue

            classified = self._classify_single_item(item)
            classified_items.append(classified)

        return ClassificationResult(
            feedback_items=classified_items,
            reviewer_name=reviewer_name,
            raw_text=raw_feedback,
        )

    def _split_feedback(self, text: str) -> list[str]:
        """Split feedback text into individual items."""
        # Try to split by common patterns
        lines = text.strip().split("\n")

        items = []
        current_item = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_item:
                    items.append(" ".join(current_item))
                    current_item = []
                continue

            # Check if this is a new item (starts with bullet, number, or dash)
            is_new_item = bool(re.match(r"^[-*•]\s+|^\d+[.)]\s+|^#{1,3}\s+", line))

            if is_new_item and current_item:
                items.append(" ".join(current_item))
                current_item = []

            # Remove bullet/number prefix
            cleaned = re.sub(r"^[-*•]\s+|^\d+[.)]\s+|^#{1,3}\s+", "", line)
            current_item.append(cleaned)

        if current_item:
            items.append(" ".join(current_item))

        return items

    def _classify_single_item(self, text: str) -> ClassifiedFeedback:
        """Classify a single feedback item."""
        text_lower = text.lower()

        # Detect severity
        severity = self._detect_severity(text_lower)

        # Detect category
        category = self._detect_category(text_lower)

        # Detect actionability based on severity
        actionability = self._severity_to_actionability(severity)

        # Determine if blocker
        is_blocker = severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)

        # Extract location if present
        file_path, line_number = self._extract_location(text)

        # Extract fix suggestion if present
        fix_suggestion = self._extract_suggestion(text)

        return ClassifiedFeedback(
            original_text=text,
            severity=severity,
            category=category,
            actionability=actionability,
            is_blocker=is_blocker,
            file_path=file_path,
            line_number=line_number,
            fix_suggestion=fix_suggestion,
        )

    def _is_negated(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword is negated in the text.

        Looks for negation words within a window before the keyword.
        For example: "no security issues" or "not a vulnerability"

        Args:
            text: The full text to check.
            keyword: The keyword to check for negation.

        Returns:
            True if the keyword is negated, False otherwise.
        """
        # Find the keyword position
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return False

        # Look at the text window before the keyword (up to 30 chars)
        window_start = max(0, keyword_pos - 30)
        window = text[window_start:keyword_pos].lower()

        # Check for negation words in the window
        for negation in self.NEGATION_WORDS:
            if negation in window:
                # Verify it's reasonably close (within 3 words)
                # by checking there aren't too many spaces
                text_between = window[window.rfind(negation):]
                word_count = len(text_between.split())
                if word_count <= 4:  # negation + up to 3 words
                    return True

        return False

    def _has_affirmation(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword has affirmation context.

        Looks for affirmation words near the keyword.
        For example: "found vulnerability" or "detected security issue"

        Args:
            text: The full text to check.
            keyword: The keyword to check for affirmation.

        Returns:
            True if the keyword has affirmation, False otherwise.
        """
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return False

        # Look at the text window before the keyword
        window_start = max(0, keyword_pos - 30)
        window = text[window_start:keyword_pos].lower()

        for affirmation in self.AFFIRMATION_PATTERNS:
            if affirmation in window:
                return True

        return False

    def _keyword_matches(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword truly matches (not negated).

        Args:
            text: Text to search in.
            keyword: Keyword to look for.

        Returns:
            True if keyword matches and is not negated.
        """
        if keyword not in text:
            return False

        # Check for negation
        if self._is_negated(text, keyword):
            return False

        return True

    def _detect_severity(self, text_lower: str) -> IssueSeverity:
        """
        Detect severity from text with negation awareness.

        Uses keyword matching that respects negation patterns like
        "no issues" or "not a vulnerability" to avoid false positives.
        """
        # Check for critical keywords first
        for keyword in self.CRITICAL_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                return IssueSeverity.CRITICAL

        for keyword in self.HIGH_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                return IssueSeverity.HIGH

        for keyword in self.LOW_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                return IssueSeverity.LOW

        # Default to medium for anything else
        return IssueSeverity.MEDIUM

    def _detect_category(self, text_lower: str) -> IssueCategory:
        """
        Detect category from text with negation awareness.

        Uses keyword matching that respects negation patterns.
        """
        # Count keyword matches for each category (with negation awareness)
        scores: dict[IssueCategory, int] = {cat: 0 for cat in IssueCategory}

        for keyword in self.SECURITY_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                scores[IssueCategory.SECURITY] += 1

        for keyword in self.PERFORMANCE_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                scores[IssueCategory.PERFORMANCE] += 1

        for keyword in self.CORRECTNESS_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                scores[IssueCategory.CORRECTNESS] += 1

        for keyword in self.TESTING_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                scores[IssueCategory.TESTING] += 1

        for keyword in self.ARCHITECTURE_KEYWORDS:
            if self._keyword_matches(text_lower, keyword):
                scores[IssueCategory.ARCHITECTURE] += 1

        # Find category with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return IssueCategory.OTHER

        for category, score in scores.items():
            if score == max_score:
                return category

        return IssueCategory.OTHER

    def _severity_to_actionability(self, severity: IssueSeverity) -> Actionability:
        """Map severity to actionability."""
        mapping = {
            IssueSeverity.CRITICAL: Actionability.IMMEDIATE,
            IssueSeverity.HIGH: Actionability.IMMEDIATE,
            IssueSeverity.MEDIUM: Actionability.SHOULD_FIX,
            IssueSeverity.LOW: Actionability.NICE_TO_HAVE,
        }
        return mapping[severity]

    def _extract_location(self, text: str) -> tuple[str | None, int | None]:
        """Extract file path and line number from text."""
        for pattern in self.LOCATION_PATTERNS:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # Safely get groups with None checks
                    group0 = groups[0] if groups[0] else None
                    group1 = groups[1] if len(groups) > 1 and groups[1] else None

                    # Handle different pattern group orders
                    if group0 and group0.isdigit():
                        # Pattern: line N in file
                        try:
                            return group1, int(group0)
                        except (ValueError, TypeError):
                            return group1, None
                    else:
                        # Pattern: file:line or file line N
                        try:
                            line = int(group1) if group1 and group1.isdigit() else None
                        except (ValueError, TypeError):
                            line = None
                        return group0, line
                elif len(groups) == 1 and groups[0]:
                    return groups[0], None
        return None, None

    def _extract_suggestion(self, text: str) -> str | None:
        """Extract fix suggestion from text."""
        # Look for common suggestion patterns
        patterns = [
            re.compile(r"(?:suggest|recommend|should|could|consider)\s+(.+?)(?:\.|$)", re.I),
            re.compile(r"(?:fix|solution|instead)\s*:\s*(.+?)(?:\.|$)", re.I),
            re.compile(r"(?:try|use)\s+(.+?)(?:\s+instead|$)", re.I),
        ]

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()

        return None


def merge_classifications(
    results: list[ClassificationResult],
) -> ClassificationResult:
    """
    Merge multiple classification results into one.

    Useful for consolidating feedback from multiple reviewers.

    Args:
        results: List of classification results to merge.

    Returns:
        Merged ClassificationResult.
    """
    if not results:
        return ClassificationResult(
            feedback_items=[],
            reviewer_name="merged",
            raw_text="",
        )

    all_items = []
    all_raw = []
    reviewer_names = []

    for result in results:
        all_items.extend(result.feedback_items)
        all_raw.append(f"## {result.reviewer_name}\n{result.raw_text}")
        reviewer_names.append(result.reviewer_name)

    return ClassificationResult(
        feedback_items=all_items,
        reviewer_name="merged",
        raw_text="\n\n".join(all_raw),
        metadata={"source_reviewers": reviewer_names},
    )
