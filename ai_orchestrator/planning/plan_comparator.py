"""Plan comparator for scoring and ranking implementation plans."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ai_orchestrator.core.workflow_phases import Plan

logger = logging.getLogger(__name__)


class ScoringCriterion(str, Enum):
    """Criteria for scoring plans."""

    COMPLETENESS = "completeness"  # All required sections present
    CORRECTNESS = "correctness"  # Technically sound approach
    SECURITY = "security"  # Security considerations addressed
    TESTABILITY = "testability"  # Testing strategy included
    MAINTAINABILITY = "maintainability"  # Code quality focus
    BLAST_RADIUS = "blast_radius"  # Scope of changes
    INTEGRATION_FRICTION = "integration_friction"  # Ease of integration
    PERFORMANCE = "performance"  # Performance considerations


# Default weights for scoring criteria
DEFAULT_WEIGHTS: dict[ScoringCriterion, float] = {
    ScoringCriterion.COMPLETENESS: 0.20,
    ScoringCriterion.CORRECTNESS: 0.25,
    ScoringCriterion.SECURITY: 0.20,
    ScoringCriterion.TESTABILITY: 0.15,
    ScoringCriterion.MAINTAINABILITY: 0.15,
    ScoringCriterion.BLAST_RADIUS: 0.10,
    ScoringCriterion.INTEGRATION_FRICTION: 0.10,
    ScoringCriterion.PERFORMANCE: 0.10,
}


@dataclass
class CriterionScore:
    """Score for a single criterion."""

    criterion: ScoringCriterion
    score: float  # 0-1
    weight: float
    reasoning: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight


@dataclass
class PlanScore:
    """Complete scoring for a plan."""

    plan_source: str
    criterion_scores: list[CriterionScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall weighted score."""
        if not self.criterion_scores:
            return 0.0

        total_weighted = sum(cs.weighted_score for cs in self.criterion_scores)
        total_weight = sum(cs.weight for cs in self.criterion_scores)

        return total_weighted / total_weight if total_weight > 0 else 0.0

    @property
    def strengths(self) -> list[str]:
        """Get criteria where plan scores well (>0.7)."""
        return [
            cs.criterion.value
            for cs in self.criterion_scores
            if cs.score >= 0.7
        ]

    @property
    def weaknesses(self) -> list[str]:
        """Get criteria where plan scores poorly (<0.4)."""
        return [
            cs.criterion.value
            for cs in self.criterion_scores
            if cs.score < 0.4
        ]

    def get_score(self, criterion: ScoringCriterion) -> float:
        """Get score for a specific criterion."""
        for cs in self.criterion_scores:
            if cs.criterion == criterion:
                return cs.score
        return 0.0

    def to_summary(self) -> str:
        """Generate a summary of the plan score."""
        lines = [
            f"Plan: {self.plan_source}",
            f"Overall Score: {self.overall_score:.2f}",
            "",
            "Criterion Scores:",
        ]

        for cs in sorted(self.criterion_scores, key=lambda x: x.weighted_score, reverse=True):
            bar = "=" * int(cs.score * 10)
            lines.append(f"  {cs.criterion.value:<20} {cs.score:.2f} [{bar:<10}]")

        if self.strengths:
            lines.append(f"\nStrengths: {', '.join(self.strengths)}")
        if self.weaknesses:
            lines.append(f"Weaknesses: {', '.join(self.weaknesses)}")

        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing multiple plans."""

    scores: list[PlanScore]
    best_plan_source: str
    ranking: list[str]  # Ordered list of plan sources
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> PlanScore | None:
        """Get the highest-scoring plan."""
        for score in self.scores:
            if score.plan_source == self.best_plan_source:
                return score
        return None


class PlanComparator:
    """
    Score and rank implementation plans based on multiple criteria.

    Uses heuristic analysis to score plans without requiring LLM calls.
    For more accurate scoring, consider LLM-based evaluation.
    """

    # Keywords for criterion detection
    CRITERION_KEYWORDS: dict[ScoringCriterion, list[str]] = {
        ScoringCriterion.COMPLETENESS: [
            "overview", "architecture", "implementation", "steps",
            "files", "testing", "security", "risks",
        ],
        ScoringCriterion.CORRECTNESS: [
            "correct", "proper", "best practice", "standard",
            "pattern", "solid", "design",
        ],
        ScoringCriterion.SECURITY: [
            "security", "auth", "permission", "access", "sanitize",
            "validate", "encrypt", "token", "csrf", "xss",
        ],
        ScoringCriterion.TESTABILITY: [
            "test", "unit test", "integration", "e2e", "coverage",
            "mock", "fixture", "assert", "verify",
        ],
        ScoringCriterion.MAINTAINABILITY: [
            "maintainable", "readable", "clean", "modular",
            "separation", "concern", "refactor", "document",
        ],
        ScoringCriterion.BLAST_RADIUS: [
            "minimal", "focused", "isolated", "contained",
            "scope", "impact", "change",
        ],
        ScoringCriterion.INTEGRATION_FRICTION: [
            "compatible", "backward", "migration", "smooth",
            "gradual", "incremental", "phased",
        ],
        ScoringCriterion.PERFORMANCE: [
            "performance", "optimize", "cache", "efficient",
            "fast", "latency", "memory", "cpu",
        ],
    }

    # Required sections for completeness check
    REQUIRED_SECTIONS = [
        "overview",
        "architecture",
        "implementation",
        "testing",
        "security",
    ]

    def __init__(
        self,
        weights: dict[ScoringCriterion, float] | None = None,
    ) -> None:
        """
        Initialize comparator.

        Args:
            weights: Custom weights for scoring criteria.
        """
        self.weights = weights or DEFAULT_WEIGHTS

    def score_plan(self, plan: Plan) -> PlanScore:
        """
        Score a single plan on all criteria.

        Args:
            plan: Plan to score.

        Returns:
            PlanScore with scores for each criterion.
        """
        content_lower = plan.content.lower()

        criterion_scores = []

        for criterion in ScoringCriterion:
            score, reasoning, evidence = self._score_criterion(
                content_lower,
                plan.content,
                criterion,
            )

            weight = self.weights.get(criterion, 0.1)

            criterion_scores.append(
                CriterionScore(
                    criterion=criterion,
                    score=score,
                    weight=weight,
                    reasoning=reasoning,
                    evidence=evidence,
                )
            )

        return PlanScore(
            plan_source=plan.source_cli,
            criterion_scores=criterion_scores,
        )

    def compare_plans(self, plans: list[Plan]) -> ComparisonResult:
        """
        Compare and rank multiple plans.

        Args:
            plans: List of plans to compare.

        Returns:
            ComparisonResult with rankings.
        """
        if not plans:
            return ComparisonResult(
                scores=[],
                best_plan_source="",
                ranking=[],
            )

        # Score all plans
        scores = [self.score_plan(plan) for plan in plans]

        # Sort by overall score
        sorted_scores = sorted(
            scores,
            key=lambda s: s.overall_score,
            reverse=True,
        )

        ranking = [s.plan_source for s in sorted_scores]
        best_source = ranking[0] if ranking else ""

        return ComparisonResult(
            scores=scores,
            best_plan_source=best_source,
            ranking=ranking,
        )

    def _score_criterion(
        self,
        content_lower: str,
        content_original: str,
        criterion: ScoringCriterion,
    ) -> tuple[float, str, list[str]]:
        """Score a single criterion."""
        keywords = self.CRITERION_KEYWORDS.get(criterion, [])
        evidence = []

        if criterion == ScoringCriterion.COMPLETENESS:
            return self._score_completeness(content_lower)

        if criterion == ScoringCriterion.BLAST_RADIUS:
            return self._score_blast_radius(content_lower, content_original)

        # Default keyword-based scoring
        keyword_count = 0
        for keyword in keywords:
            if keyword in content_lower:
                keyword_count += 1
                # Extract context around keyword
                idx = content_lower.find(keyword)
                if idx != -1:
                    start = max(0, idx - 20)
                    end = min(len(content_original), idx + len(keyword) + 50)
                    evidence.append(content_original[start:end].strip())

        # Normalize score based on keyword presence
        max_keywords = len(keywords)
        score = min(1.0, keyword_count / (max_keywords * 0.5)) if max_keywords > 0 else 0.5

        reasoning = f"Found {keyword_count}/{max_keywords} relevant keywords"

        return score, reasoning, evidence[:3]  # Limit evidence

    def _score_completeness(
        self,
        content_lower: str,
    ) -> tuple[float, str, list[str]]:
        """Score plan completeness based on required sections."""
        found_sections = []
        missing_sections = []

        for section in self.REQUIRED_SECTIONS:
            if section in content_lower:
                found_sections.append(section)
            else:
                missing_sections.append(section)

        score = len(found_sections) / len(self.REQUIRED_SECTIONS)
        reasoning = f"Found {len(found_sections)}/{len(self.REQUIRED_SECTIONS)} required sections"

        if missing_sections:
            reasoning += f". Missing: {', '.join(missing_sections)}"

        return score, reasoning, found_sections

    def _score_blast_radius(
        self,
        content_lower: str,
        content_original: str,
    ) -> tuple[float, str, list[str]]:
        """Score blast radius (smaller changes = higher score)."""
        evidence = []

        # Look for file count indicators
        file_patterns = [
            r"(\d+)\s*files?",
            r"modify\s*(\d+)",
            r"change\s*(\d+)",
        ]

        file_count = 0
        for pattern in file_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                file_count = max(file_count, max(int(m) for m in matches))

        # Look for modular/focused language
        positive_keywords = ["focused", "minimal", "isolated", "targeted", "contained"]
        positive_count = sum(1 for k in positive_keywords if k in content_lower)

        # Score inversely proportional to file count
        if file_count == 0:
            file_score = 0.7  # Unknown, assume moderate
        elif file_count <= 3:
            file_score = 1.0
        elif file_count <= 5:
            file_score = 0.8
        elif file_count <= 10:
            file_score = 0.6
        else:
            file_score = 0.4

        # Boost for positive language
        positive_boost = min(0.2, positive_count * 0.05)

        score = min(1.0, file_score + positive_boost)
        reasoning = f"Estimated {file_count} files affected"

        if positive_count > 0:
            evidence.append(f"Found {positive_count} indicators of focused scope")

        return score, reasoning, evidence

    def explain_comparison(self, result: ComparisonResult) -> str:
        """Generate a human-readable explanation of comparison results."""
        lines = [
            "=" * 60,
            "PLAN COMPARISON RESULTS",
            "=" * 60,
            "",
        ]

        for i, source in enumerate(result.ranking, 1):
            score = next(s for s in result.scores if s.plan_source == source)
            medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(i, f"[{i}th]")
            lines.append(f"{medal} {source}: {score.overall_score:.2f}")

            if score.strengths:
                lines.append(f"     Strengths: {', '.join(score.strengths[:3])}")
            if score.weaknesses:
                lines.append(f"     Weaknesses: {', '.join(score.weaknesses[:3])}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
