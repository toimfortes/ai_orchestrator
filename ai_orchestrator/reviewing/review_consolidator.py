"""Review consolidator for merging multi-reviewer feedback.

Combines feedback from multiple reviewers, detects duplicates,
calculates consensus severity, and aggregates fix suggestions.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ai_orchestrator.reviewing.feedback_classifier import (
    Actionability,
    ClassificationResult,
    ClassifiedFeedback,
    IssueCategory,
    IssueSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedIssue:
    """A single issue consolidated from multiple reviewers."""

    # Original feedback items that were merged
    source_items: list[ClassifiedFeedback]

    # Consolidated properties
    severity: IssueSeverity
    category: IssueCategory
    message: str
    location: str | None = None

    # Consensus metrics
    reviewer_count: int = 1
    consensus_score: float = 1.0  # 0-1, higher = more agreement
    severity_votes: dict[IssueSeverity, int] = field(default_factory=dict)

    # Aggregated suggestions
    fix_suggestions: list[str] = field(default_factory=list)

    # Reviewers who identified this issue
    reviewers: list[str] = field(default_factory=list)

    @property
    def is_blocker(self) -> bool:
        """Issue is blocker if majority voted it as blocker."""
        blocker_votes = sum(1 for item in self.source_items if item.is_blocker)
        return blocker_votes > len(self.source_items) / 2

    @property
    def has_consensus(self) -> bool:
        """True if majority of reviewers agree on severity."""
        if not self.severity_votes:
            return True
        max_votes = max(self.severity_votes.values())
        return max_votes > self.reviewer_count / 2

    def to_classified_feedback(self) -> ClassifiedFeedback:
        """Convert back to ClassifiedFeedback for workflow integration."""
        # Determine actionability from consolidated severity
        actionability_map = {
            IssueSeverity.CRITICAL: Actionability.IMMEDIATE,
            IssueSeverity.HIGH: Actionability.IMMEDIATE,
            IssueSeverity.MEDIUM: Actionability.SHOULD_FIX,
            IssueSeverity.LOW: Actionability.NICE_TO_HAVE,
        }

        # Combine fix suggestions
        combined_suggestion = None
        if self.fix_suggestions:
            unique_suggestions = list(dict.fromkeys(self.fix_suggestions))  # Dedupe
            combined_suggestion = " | ".join(unique_suggestions[:3])  # Top 3

        return ClassifiedFeedback(
            original_text=self.message,
            severity=self.severity,
            category=self.category,
            actionability=actionability_map[self.severity],
            is_blocker=self.is_blocker,
            file_path=self._extract_file_path(),
            line_number=self._extract_line_number(),
            fix_suggestion=combined_suggestion,
            confidence=self.consensus_score,
            metadata={
                "consolidated": True,
                "reviewer_count": self.reviewer_count,
                "reviewers": self.reviewers,
                "severity_votes": {k.value: v for k, v in self.severity_votes.items()},
            },
        )

    def _extract_file_path(self) -> str | None:
        """Extract file path from location or source items."""
        if self.location and ":" in self.location:
            return self.location.split(":")[0]
        for item in self.source_items:
            if item.file_path:
                return item.file_path
        return None

    def _extract_line_number(self) -> int | None:
        """Extract line number from location or source items."""
        if self.location and ":" in self.location:
            try:
                return int(self.location.split(":")[1])
            except (IndexError, ValueError):
                pass
        for item in self.source_items:
            if item.line_number:
                return item.line_number
        return None


@dataclass
class ConsolidationResult:
    """Result of consolidating multiple review results."""

    # Consolidated issues (deduplicated and merged)
    issues: list[ConsolidatedIssue]

    # Source results that were consolidated
    source_results: list[ClassificationResult]

    # Metrics
    total_source_items: int = 0
    duplicate_count: int = 0
    consensus_rate: float = 0.0  # Percentage of issues with consensus

    # Issues by severity (after consolidation)
    by_severity: dict[IssueSeverity, list[ConsolidatedIssue]] = field(
        default_factory=dict
    )

    @property
    def critical_count(self) -> int:
        """Count of critical issues after consolidation."""
        return len(self.by_severity.get(IssueSeverity.CRITICAL, []))

    @property
    def high_count(self) -> int:
        """Count of high severity issues after consolidation."""
        return len(self.by_severity.get(IssueSeverity.HIGH, []))

    @property
    def blocker_count(self) -> int:
        """Count of blocking issues."""
        return sum(1 for issue in self.issues if issue.is_blocker)

    @property
    def has_blockers(self) -> bool:
        """Whether there are any blocking issues."""
        return self.blocker_count > 0

    def get_blocking_issues(self) -> list[ConsolidatedIssue]:
        """Get all blocking issues."""
        return [issue for issue in self.issues if issue.is_blocker]

    def to_classification_result(self, reviewer_name: str = "consolidated") -> ClassificationResult:
        """Convert to ClassificationResult for workflow integration."""
        feedback_items = [issue.to_classified_feedback() for issue in self.issues]

        raw_text_parts = []
        for result in self.source_results:
            raw_text_parts.append(f"## {result.reviewer_name}\n{result.raw_text}")

        return ClassificationResult(
            feedback_items=feedback_items,
            reviewer_name=reviewer_name,
            raw_text="\n\n".join(raw_text_parts),
            metadata={
                "consolidated": True,
                "source_count": len(self.source_results),
                "duplicate_count": self.duplicate_count,
                "consensus_rate": self.consensus_rate,
            },
        )

    def summary(self) -> str:
        """Generate human-readable consolidation summary."""
        lines = [
            "Consolidation Summary:",
            f"  Source reviews: {len(self.source_results)}",
            f"  Total source items: {self.total_source_items}",
            f"  Consolidated issues: {len(self.issues)}",
            f"  Duplicates merged: {self.duplicate_count}",
            f"  Consensus rate: {self.consensus_rate:.1%}",
            "",
            "By severity:",
            f"  Critical: {self.critical_count}",
            f"  High: {self.high_count}",
            f"  Blockers: {self.blocker_count}",
        ]
        return "\n".join(lines)


class ReviewConsolidator:
    """
    Consolidates feedback from multiple reviewers.

    Features:
    - Duplicate detection using text similarity
    - Consensus severity calculation
    - Location-based grouping
    - Fix suggestion aggregation

    Example:
        consolidator = ReviewConsolidator()
        result = consolidator.consolidate([claude_result, gemini_result, codex_result])
        # result.issues contains deduplicated, consensus-weighted issues
    """

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        location_weight: float = 0.3,
        content_weight: float = 0.7,
    ) -> None:
        """
        Initialize the consolidator.

        Args:
            similarity_threshold: Minimum similarity (0-1) to consider duplicates.
            location_weight: Weight for location similarity in matching.
            content_weight: Weight for content similarity in matching.
        """
        self.similarity_threshold = similarity_threshold
        self.location_weight = location_weight
        self.content_weight = content_weight

    def consolidate(
        self,
        results: list[ClassificationResult],
    ) -> ConsolidationResult:
        """
        Consolidate multiple classification results into one.

        Args:
            results: List of ClassificationResult from different reviewers.

        Returns:
            ConsolidationResult with merged, deduplicated issues.
        """
        if not results:
            return ConsolidationResult(
                issues=[],
                source_results=[],
            )

        # Collect all feedback items with reviewer info
        all_items: list[tuple[ClassifiedFeedback, str]] = []
        for result in results:
            for item in result.feedback_items:
                all_items.append((item, result.reviewer_name))

        total_source = len(all_items)
        logger.info("Consolidating %d items from %d reviewers", total_source, len(results))

        # Group by location first (if available)
        location_groups = self._group_by_location(all_items)

        # Then deduplicate within and across groups
        consolidated_issues = self._deduplicate_and_merge(location_groups)

        # Calculate consensus metrics
        duplicate_count = total_source - len(consolidated_issues)
        consensus_issues = sum(1 for issue in consolidated_issues if issue.has_consensus)
        consensus_rate = consensus_issues / len(consolidated_issues) if consolidated_issues else 1.0

        # Group by severity
        by_severity: dict[IssueSeverity, list[ConsolidatedIssue]] = defaultdict(list)
        for issue in consolidated_issues:
            by_severity[issue.severity].append(issue)

        result = ConsolidationResult(
            issues=consolidated_issues,
            source_results=results,
            total_source_items=total_source,
            duplicate_count=duplicate_count,
            consensus_rate=consensus_rate,
            by_severity=dict(by_severity),
        )

        logger.info("Consolidation complete:\n%s", result.summary())
        return result

    def _group_by_location(
        self,
        items: list[tuple[ClassifiedFeedback, str]],
    ) -> dict[str, list[tuple[ClassifiedFeedback, str]]]:
        """Group items by file location."""
        groups: dict[str, list[tuple[ClassifiedFeedback, str]]] = defaultdict(list)

        for item, reviewer in items:
            location = item.location or "unknown"
            # Normalize location (just file, not line for grouping)
            if ":" in location:
                location = location.split(":")[0]
            groups[location].append((item, reviewer))

        return groups

    def _deduplicate_and_merge(
        self,
        location_groups: dict[str, list[tuple[ClassifiedFeedback, str]]],
    ) -> list[ConsolidatedIssue]:
        """Deduplicate items within location groups and merge similar ones."""
        consolidated: list[ConsolidatedIssue] = []
        processed: set[int] = set()  # Track processed item indices

        # Flatten to list for indexing
        all_items = []
        for location, items in location_groups.items():
            for item, reviewer in items:
                all_items.append((item, reviewer, location))

        for i, (item1, reviewer1, loc1) in enumerate(all_items):
            if i in processed:
                continue

            # Find all similar items
            similar_items = [(item1, reviewer1)]
            processed.add(i)

            for j, (item2, reviewer2, loc2) in enumerate(all_items):
                if j in processed:
                    continue

                similarity = self._calculate_similarity(item1, item2, loc1, loc2)
                if similarity >= self.similarity_threshold:
                    similar_items.append((item2, reviewer2))
                    processed.add(j)

            # Create consolidated issue from similar items
            consolidated_issue = self._merge_items(similar_items)
            consolidated.append(consolidated_issue)

        # Sort by severity (critical first)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }
        consolidated.sort(key=lambda x: severity_order.get(x.severity, 4))

        return consolidated

    def _calculate_similarity(
        self,
        item1: ClassifiedFeedback,
        item2: ClassifiedFeedback,
        loc1: str,
        loc2: str,
    ) -> float:
        """Calculate similarity between two feedback items."""
        # Location similarity
        location_sim = 1.0 if loc1 == loc2 else 0.0

        # Content similarity using SequenceMatcher
        text1 = self._normalize_text(item1.original_text)
        text2 = self._normalize_text(item2.original_text)
        content_sim = SequenceMatcher(None, text1, text2).ratio()

        # Category match bonus
        category_bonus = 0.1 if item1.category == item2.category else 0.0

        # Weighted combination
        similarity = (
            self.location_weight * location_sim
            + self.content_weight * content_sim
            + category_bonus
        )

        return min(similarity, 1.0)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove common prefixes
        text = re.sub(r"^(critical|high|medium|low|bug|issue|error):\s*", "", text)
        return text.strip()

    def _merge_items(
        self,
        items: list[tuple[ClassifiedFeedback, str]],
    ) -> ConsolidatedIssue:
        """Merge similar items into a single consolidated issue."""
        source_items = [item for item, _ in items]
        reviewers = [reviewer for _, reviewer in items]

        # Vote on severity
        severity_votes: dict[IssueSeverity, int] = defaultdict(int)
        for item in source_items:
            severity_votes[item.severity] += 1

        # Consensus severity = most voted
        consensus_severity = max(severity_votes.keys(), key=lambda s: severity_votes[s])

        # Vote on category
        category_votes: dict[IssueCategory, int] = defaultdict(int)
        for item in source_items:
            category_votes[item.category] += 1
        consensus_category = max(category_votes.keys(), key=lambda c: category_votes[c])

        # Use the most detailed message (longest)
        best_message = max(source_items, key=lambda x: len(x.original_text)).original_text

        # Collect unique fix suggestions
        fix_suggestions = []
        for item in source_items:
            if item.fix_suggestion and item.fix_suggestion not in fix_suggestions:
                fix_suggestions.append(item.fix_suggestion)

        # Get location from any item that has one
        location = None
        for item in source_items:
            if item.location:
                location = item.location
                break

        # Calculate consensus score
        max_votes = max(severity_votes.values())
        consensus_score = max_votes / len(items)

        return ConsolidatedIssue(
            source_items=source_items,
            severity=consensus_severity,
            category=consensus_category,
            message=best_message,
            location=location,
            reviewer_count=len(items),
            consensus_score=consensus_score,
            severity_votes=dict(severity_votes),
            fix_suggestions=fix_suggestions,
            reviewers=list(dict.fromkeys(reviewers)),  # Unique, preserve order
        )


def consolidate_reviews(results: list[ClassificationResult]) -> ConsolidationResult:
    """Convenience function for simple consolidation."""
    consolidator = ReviewConsolidator()
    return consolidator.consolidate(results)
