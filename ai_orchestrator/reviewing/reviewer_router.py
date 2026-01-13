"""Specialist reviewer routing for targeted code review.

Routes specific issue categories to expert reviewers based on their strengths.
For example, security issues go to Claude/GPT-5, performance to Gemini.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ai_orchestrator.reviewing.feedback_classifier import (
    ClassificationResult,
    ClassifiedFeedback,
    IssueCategory,
    IssueSeverity,
)
from ai_orchestrator.utils import truncate_with_marker

if TYPE_CHECKING:
    from ai_orchestrator.cli_adapters.base import CLIAdapter

logger = logging.getLogger(__name__)


class ReviewerStrength(str, Enum):
    """Areas where each reviewer excels."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    CORRECTNESS = "correctness"
    TESTING = "testing"
    GENERAL = "general"


@dataclass
class ReviewerProfile:
    """Profile defining a reviewer's strengths and capabilities."""

    name: str
    strengths: list[ReviewerStrength]
    tier: int = 1  # 1=ultra, 2=pro, 3=free
    cost_weight: float = 1.0  # Relative cost (for budget optimization)
    max_concurrent_reviews: int = 3


# Default reviewer profiles based on model capabilities
DEFAULT_REVIEWER_PROFILES: dict[str, ReviewerProfile] = {
    "claude": ReviewerProfile(
        name="claude",
        strengths=[
            ReviewerStrength.SECURITY,
            ReviewerStrength.ARCHITECTURE,
            ReviewerStrength.CORRECTNESS,
            ReviewerStrength.GENERAL,
        ],
        tier=1,
        cost_weight=1.0,
    ),
    "codex": ReviewerProfile(
        name="codex",
        strengths=[
            ReviewerStrength.CORRECTNESS,
            ReviewerStrength.TESTING,
            ReviewerStrength.GENERAL,
        ],
        tier=1,
        cost_weight=1.0,
    ),
    "gemini": ReviewerProfile(
        name="gemini",
        strengths=[
            ReviewerStrength.PERFORMANCE,
            ReviewerStrength.ARCHITECTURE,
            ReviewerStrength.GENERAL,
        ],
        tier=2,
        cost_weight=0.5,  # Lower cost
    ),
    "kilocode": ReviewerProfile(
        name="kilocode",
        strengths=[ReviewerStrength.GENERAL],
        tier=3,
        cost_weight=0.1,  # Free tier models
    ),
}

# Mapping from issue category to reviewer strength
CATEGORY_TO_STRENGTH: dict[IssueCategory, ReviewerStrength] = {
    IssueCategory.SECURITY: ReviewerStrength.SECURITY,
    IssueCategory.PERFORMANCE: ReviewerStrength.PERFORMANCE,
    IssueCategory.ARCHITECTURE: ReviewerStrength.ARCHITECTURE,
    IssueCategory.CORRECTNESS: ReviewerStrength.CORRECTNESS,
    IssueCategory.TESTING: ReviewerStrength.TESTING,
    IssueCategory.MAINTAINABILITY: ReviewerStrength.ARCHITECTURE,
    IssueCategory.DOCUMENTATION: ReviewerStrength.GENERAL,
    IssueCategory.STYLE: ReviewerStrength.GENERAL,
    IssueCategory.OTHER: ReviewerStrength.GENERAL,
}


@dataclass
class RoutingDecision:
    """Decision about which reviewers should handle which issues."""

    category: IssueCategory
    assigned_reviewers: list[str]
    reason: str
    priority: int = 1  # Lower = higher priority


@dataclass
class ReviewRoutingPlan:
    """Complete routing plan for a review round."""

    # General reviewers (review everything)
    general_reviewers: list[str]

    # Specialist assignments (category -> reviewers)
    specialist_assignments: dict[IssueCategory, list[str]]

    # Issues that need specialist attention
    specialist_issues: list[ClassifiedFeedback]

    # Routing decisions made
    decisions: list[RoutingDecision] = field(default_factory=list)

    def get_reviewers_for_category(self, category: IssueCategory) -> list[str]:
        """Get all reviewers that should review a category."""
        reviewers = set(self.general_reviewers)
        if category in self.specialist_assignments:
            reviewers.update(self.specialist_assignments[category])
        return list(reviewers)

    def summary(self) -> str:
        """Generate human-readable routing summary."""
        lines = ["Review Routing Plan:"]
        lines.append(f"  General reviewers: {', '.join(self.general_reviewers)}")

        if self.specialist_assignments:
            lines.append("  Specialist assignments:")
            for category, reviewers in self.specialist_assignments.items():
                lines.append(f"    {category.value}: {', '.join(reviewers)}")

        if self.specialist_issues:
            lines.append(f"  Issues requiring specialist review: {len(self.specialist_issues)}")

        return "\n".join(lines)


class ReviewerRouter:
    """
    Routes review tasks to appropriate specialist reviewers.

    Strategy:
    1. Round 1: All available reviewers do general review
    2. Classify issues by category and severity
    3. Round 2+: Route critical/high issues to specialists

    Example:
        router = ReviewerRouter(available_reviewers=["claude", "gemini", "codex"])
        plan = router.create_routing_plan(classified_feedback)
        # plan.specialist_assignments = {SECURITY: ["claude"], PERFORMANCE: ["gemini"]}
    """

    def __init__(
        self,
        available_reviewers: list[str],
        profiles: dict[str, ReviewerProfile] | None = None,
        require_specialist_for_critical: bool = True,
        max_specialists_per_category: int = 2,
    ) -> None:
        """
        Initialize the reviewer router.

        Args:
            available_reviewers: List of available CLI reviewer names.
            profiles: Custom reviewer profiles (defaults to DEFAULT_REVIEWER_PROFILES).
            require_specialist_for_critical: If True, critical issues must have specialists.
            max_specialists_per_category: Max specialists to assign per category.
        """
        self.available_reviewers = available_reviewers
        self.profiles = profiles or DEFAULT_REVIEWER_PROFILES
        self.require_specialist_for_critical = require_specialist_for_critical
        self.max_specialists_per_category = max_specialists_per_category

    def create_routing_plan(
        self,
        feedback: ClassificationResult | None = None,
        force_general_only: bool = False,
    ) -> ReviewRoutingPlan:
        """
        Create a routing plan based on classified feedback.

        Args:
            feedback: Classified feedback from previous review round.
                      If None, creates general-only plan (for round 1).
            force_general_only: If True, skip specialist routing.

        Returns:
            ReviewRoutingPlan with reviewer assignments.
        """
        # Round 1 or forced general: all reviewers do general review
        if feedback is None or force_general_only:
            return ReviewRoutingPlan(
                general_reviewers=self.available_reviewers,
                specialist_assignments={},
                specialist_issues=[],
                decisions=[
                    RoutingDecision(
                        category=IssueCategory.OTHER,
                        assigned_reviewers=self.available_reviewers,
                        reason="Initial review round - all reviewers",
                    )
                ],
            )

        # Analyze feedback to determine specialist needs
        specialist_needs = self._analyze_specialist_needs(feedback)

        # Assign specialists to categories
        specialist_assignments = self._assign_specialists(specialist_needs)

        # Identify issues requiring specialist attention
        specialist_issues = [
            item
            for item in feedback.feedback_items
            if item.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
            and item.category in specialist_assignments
        ]

        # Build routing decisions
        decisions = []
        for category, reviewers in specialist_assignments.items():
            decisions.append(
                RoutingDecision(
                    category=category,
                    assigned_reviewers=reviewers,
                    reason=f"Specialist review for {category.value} issues",
                    priority=1 if category == IssueCategory.SECURITY else 2,
                )
            )

        # Determine general reviewers (at least one required)
        general_reviewers = self._select_general_reviewers(specialist_assignments)

        plan = ReviewRoutingPlan(
            general_reviewers=general_reviewers,
            specialist_assignments=specialist_assignments,
            specialist_issues=specialist_issues,
            decisions=decisions,
        )

        logger.info("Created routing plan:\n%s", plan.summary())
        return plan

    def _analyze_specialist_needs(
        self,
        feedback: ClassificationResult,
    ) -> dict[IssueCategory, int]:
        """
        Analyze feedback to determine which categories need specialists.

        Returns dict of category -> priority score (higher = more urgent).
        """
        needs: dict[IssueCategory, int] = {}

        for item in feedback.feedback_items:
            category = item.category

            # Calculate priority score
            score = 0
            if item.severity == IssueSeverity.CRITICAL:
                score = 10
            elif item.severity == IssueSeverity.HIGH:
                score = 5
            elif item.severity == IssueSeverity.MEDIUM:
                score = 2

            if item.is_blocker:
                score += 5

            # Accumulate scores per category
            if category not in needs:
                needs[category] = 0
            needs[category] += score

        # Filter to categories with significant scores
        threshold = 5  # At least one HIGH or multiple MEDIUMs
        return {cat: score for cat, score in needs.items() if score >= threshold}

    def _assign_specialists(
        self,
        needs: dict[IssueCategory, int],
    ) -> dict[IssueCategory, list[str]]:
        """
        Assign specialist reviewers to categories based on needs.

        Args:
            needs: Category -> priority score mapping.

        Returns:
            Category -> list of assigned reviewer names.
        """
        assignments: dict[IssueCategory, list[str]] = {}

        # Sort categories by priority (highest first)
        sorted_categories = sorted(needs.keys(), key=lambda c: needs[c], reverse=True)

        for category in sorted_categories:
            strength = CATEGORY_TO_STRENGTH.get(category, ReviewerStrength.GENERAL)

            # Find reviewers with this strength
            specialists = self._find_specialists_for_strength(strength)

            if specialists:
                # Limit to max_specialists_per_category
                assignments[category] = specialists[: self.max_specialists_per_category]
                logger.debug(
                    "Assigned %s to %s category",
                    assignments[category],
                    category.value,
                )
            elif self.require_specialist_for_critical and needs[category] >= 10:
                # Critical issues without specialists - use best available
                logger.warning(
                    "No specialists for %s category with critical issues, using general reviewers",
                    category.value,
                )
                assignments[category] = self.available_reviewers[:1]

        return assignments

    def _find_specialists_for_strength(
        self,
        strength: ReviewerStrength,
    ) -> list[str]:
        """Find available reviewers with a specific strength."""
        specialists = []

        for reviewer_name in self.available_reviewers:
            profile = self.profiles.get(reviewer_name)
            if profile and strength in profile.strengths:
                specialists.append(reviewer_name)

        # Sort by tier (lower = better)
        specialists.sort(
            key=lambda r: self.profiles.get(r, DEFAULT_REVIEWER_PROFILES.get(r, ReviewerProfile(r, []))).tier
        )

        return specialists

    def _select_general_reviewers(
        self,
        specialist_assignments: dict[IssueCategory, list[str]],
    ) -> list[str]:
        """
        Select general reviewers (those not assigned as specialists).

        Ensures at least one reviewer is available for general review.
        """
        # Get all specialists
        all_specialists = set()
        for reviewers in specialist_assignments.values():
            all_specialists.update(reviewers)

        # General reviewers = available - specialists (but keep at least one)
        general = [r for r in self.available_reviewers if r not in all_specialists]

        # If all reviewers are specialists, include lowest-tier one as general
        if not general and self.available_reviewers:
            # Sort by tier and pick the lowest-tier specialist
            sorted_reviewers = sorted(
                self.available_reviewers,
                key=lambda r: self.profiles.get(r, ReviewerProfile(r, [])).tier,
                reverse=True,  # Highest tier (lowest priority) first
            )
            general = [sorted_reviewers[0]]

        return general

    def get_prompt_for_specialist(
        self,
        reviewer_name: str,
        category: IssueCategory,
        issues: list[ClassifiedFeedback],
    ) -> str:
        """
        Generate a focused review prompt for a specialist reviewer.

        Args:
            reviewer_name: Name of the specialist reviewer.
            category: Category they're reviewing.
            issues: Specific issues to focus on.

        Returns:
            Focused review prompt string.
        """
        category_guidance = {
            IssueCategory.SECURITY: """
Focus your review on security aspects:
- Authentication and authorization flaws
- Input validation and sanitization
- Injection vulnerabilities (SQL, XSS, command)
- Data exposure and privacy concerns
- Cryptographic weaknesses
- Session management issues""",
            IssueCategory.PERFORMANCE: """
Focus your review on performance aspects:
- Algorithm complexity (O(n^2), etc.)
- Database query efficiency (N+1, missing indexes)
- Memory management and leaks
- Caching opportunities
- Async/concurrent execution patterns
- Resource pooling and connection management""",
            IssueCategory.ARCHITECTURE: """
Focus your review on architectural aspects:
- Design pattern usage and appropriateness
- Dependency management and coupling
- Interface segregation
- Single responsibility adherence
- Extensibility and maintainability
- Integration boundaries""",
            IssueCategory.TESTING: """
Focus your review on testing aspects:
- Test coverage gaps
- Edge case handling
- Mock/stub appropriateness
- Test isolation and independence
- Assertion quality
- Integration test boundaries""",
        }

        guidance = category_guidance.get(
            category,
            f"Focus your review on {category.value} aspects.",
        )

        issues_text = "\n".join(
            f"- [{i.severity.value.upper()}] {truncate_with_marker(i.original_text, 200)}"
            for i in issues[:10]  # Limit to top 10
        )

        return f"""You are a specialist reviewer focusing on {category.value} issues.

{guidance}

Previous review identified these {category.value}-related issues:
{issues_text}

Please provide a deep-dive review focusing specifically on these concerns.
For each issue:
1. Confirm if it's a real problem or false positive
2. Assess the actual severity
3. Provide specific fix recommendations with code examples where possible
4. Identify any related issues that might have been missed

Be thorough but focused. Only comment on {category.value}-related aspects.
"""


def create_default_router(available_reviewers: list[str]) -> ReviewerRouter:
    """Create a router with default configuration."""
    return ReviewerRouter(
        available_reviewers=available_reviewers,
        profiles=DEFAULT_REVIEWER_PROFILES,
    )
