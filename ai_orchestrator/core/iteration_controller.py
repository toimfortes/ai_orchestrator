"""Adaptive iteration controller with DDI-based convergence detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ai_orchestrator.core.workflow_phases import (
    ConvergenceStatus,
    Severity,
    WorkflowState,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationConfig:
    """Configuration for adaptive iteration control."""

    max_iterations: int = 4
    consecutive_clean_rounds_required: int = 2
    agreement_threshold: float = 0.8  # 80% agreement = converged
    stop_on_plateau: bool = True
    plateau_detection_window: int = 2  # Compare last N rounds


@dataclass
class IterationDecision:
    """Decision about whether to continue iterating."""

    should_continue: bool
    status: ConvergenceStatus
    reason: str
    current_iteration: int
    critical_count: int
    consecutive_clean: int


class AdaptiveIterationController:
    """
    DDI-based convergence detection with measurable criteria.

    Determines when to stop reviewing based on:
    1. Zero critical issues for N consecutive rounds
    2. Maximum iterations reached
    3. Feedback staleness (same issues repeated)

    Measurable Definitions:
    - Critical Issue: Severity=CRITICAL in ClassifiedFeedback
    - Converged: Zero critical issues for 2+ consecutive rounds OR 80%+ agreement
    - Plateau: Same issues repeated for 2 rounds (no progress)
    """

    def __init__(self, config: IterationConfig | None = None) -> None:
        self.config = config or IterationConfig()
        self._previous_issues: set[str] = set()

    async def should_continue_reviewing(self, state: WorkflowState) -> IterationDecision:
        """
        Determine whether to continue the review loop.

        Args:
            state: Current workflow state.

        Returns:
            IterationDecision with continuation decision and reason.
        """
        current_iteration = state.current_iteration
        consecutive_clean = self._consecutive_clean_rounds(state)
        critical_count = self._current_critical_count(state)

        # Exit condition 1: Zero critical issues for N consecutive rounds
        if consecutive_clean >= self.config.consecutive_clean_rounds_required:
            logger.info(
                "CONVERGED: %d consecutive clean rounds",
                consecutive_clean,
            )
            state.convergence_status = ConvergenceStatus.CONVERGED
            return IterationDecision(
                should_continue=False,
                status=ConvergenceStatus.CONVERGED,
                reason=f"Zero critical issues for {consecutive_clean} consecutive rounds",
                current_iteration=current_iteration,
                critical_count=critical_count,
                consecutive_clean=consecutive_clean,
            )

        # Exit condition 2: Max iterations reached
        if current_iteration >= self.config.max_iterations:
            logger.warning(
                "PLATEAU: Max iterations (%d) reached with %d critical issues",
                self.config.max_iterations,
                critical_count,
            )
            state.convergence_status = ConvergenceStatus.PLATEAU
            return IterationDecision(
                should_continue=False,
                status=ConvergenceStatus.PLATEAU,
                reason=f"Max iterations ({self.config.max_iterations}) reached",
                current_iteration=current_iteration,
                critical_count=critical_count,
                consecutive_clean=consecutive_clean,
            )

        # Exit condition 3: Same issues repeated (no progress)
        if self.config.stop_on_plateau and self._is_feedback_stale(state):
            logger.warning("PLATEAU: Feedback is stale (same issues repeated)")
            state.convergence_status = ConvergenceStatus.PLATEAU
            return IterationDecision(
                should_continue=False,
                status=ConvergenceStatus.PLATEAU,
                reason="Same issues repeated - no progress",
                current_iteration=current_iteration,
                critical_count=critical_count,
                consecutive_clean=consecutive_clean,
            )

        # Continue improving
        logger.info(
            "IMPROVING: Iteration %d, %d critical issues, %d consecutive clean",
            current_iteration,
            critical_count,
            consecutive_clean,
        )
        state.convergence_status = ConvergenceStatus.IMPROVING
        return IterationDecision(
            should_continue=True,
            status=ConvergenceStatus.IMPROVING,
            reason=f"Still {critical_count} critical issues to resolve",
            current_iteration=current_iteration,
            critical_count=critical_count,
            consecutive_clean=consecutive_clean,
        )

    def _consecutive_clean_rounds(self, state: WorkflowState) -> int:
        """Count consecutive rounds with zero CRITICAL issues."""
        count = 0
        for review_round in reversed(state.review_rounds):
            critical_count = sum(
                1
                for f in review_round.feedback
                if f.severity == Severity.CRITICAL
            )
            if critical_count == 0:
                count += 1
            else:
                break
        return count

    def _current_critical_count(self, state: WorkflowState) -> int:
        """Get critical issue count from the latest round."""
        if not state.review_rounds:
            return 0
        return state.latest_review_round.critical_count if state.latest_review_round else 0

    def _is_feedback_stale(self, state: WorkflowState) -> bool:
        """
        Check if feedback is stale (same issues repeated).

        Compares issue fingerprints across recent rounds.
        """
        if len(state.review_rounds) < self.config.plateau_detection_window:
            return False

        # Get recent rounds
        recent_rounds = state.review_rounds[-self.config.plateau_detection_window:]

        # Extract issue fingerprints (message + category)
        fingerprints_per_round = []
        for round_ in recent_rounds:
            fingerprints = {
                f"{f.category.value}:{f.message[:50]}"
                for f in round_.feedback
                if f.severity in (Severity.CRITICAL, Severity.HIGH)
            }
            fingerprints_per_round.append(fingerprints)

        # Check if all recent rounds have the same issues
        if len(fingerprints_per_round) < 2:
            return False

        first_set = fingerprints_per_round[0]
        for other_set in fingerprints_per_round[1:]:
            # Calculate Jaccard similarity
            if not first_set and not other_set:
                continue  # Both empty = converged, not stale
            if not first_set or not other_set:
                return False  # One empty, one not = still changing

            intersection = len(first_set & other_set)
            union = len(first_set | other_set)
            similarity = intersection / union if union > 0 else 0

            if similarity < 0.8:  # Less than 80% similar = still changing
                return False

        # All rounds have 80%+ similar issues = stale
        return True

    def reset(self) -> None:
        """Reset controller state for a new workflow."""
        self._previous_issues.clear()


def create_controller(
    max_iterations: int = 4,
    consecutive_clean_required: int = 2,
) -> AdaptiveIterationController:
    """Create an iteration controller with common settings."""
    config = IterationConfig(
        max_iterations=max_iterations,
        consecutive_clean_rounds_required=consecutive_clean_required,
    )
    return AdaptiveIterationController(config)
