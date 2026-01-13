"""Graceful degradation handler for terminal failures."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any

from ai_orchestrator.core.workflow_phases import WorkflowPhase, WorkflowState

logger = logging.getLogger(__name__)


class DegradedStatus(str, Enum):
    """Status of degraded operation."""

    DEGRADED = "degraded"  # Continue with reduced functionality
    BLOCKED = "blocked"  # Cannot continue, human input needed
    FAILED = "failed"  # Unrecoverable, rollback required


class DegradedAction(str, Enum):
    """Action to take on degraded status."""

    CONTINUE_DEGRADED = "continue_degraded"
    HUMAN_PLAN_REQUIRED = "human_plan_required"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    ROLLBACK_AND_REPORT = "rollback_and_report"
    RETRY_WITH_FALLBACK = "retry_with_fallback"


@dataclass
class CLIFailure:
    """Record of a CLI failure."""

    cli_name: str
    error_type: str  # timeout, rate_limited, auth_error, error
    error_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class FailureReport:
    """Human-readable failure report."""

    phase: WorkflowPhase
    failures: list[CLIFailure]
    summary: str
    recommendations: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_string(self) -> str:
        """Generate formatted report string."""
        lines = [
            "=" * 60,
            "FAILURE REPORT",
            "=" * 60,
            f"Phase: {self.phase.value}",
            f"Time: {self.timestamp.isoformat()}",
            "",
            "Summary:",
            f"  {self.summary}",
            "",
            "Failed CLIs:",
        ]

        for failure in self.failures:
            lines.append(f"  - {failure.cli_name}: {failure.error_type}")
            lines.append(f"    {failure.error_message}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DegradedResult:
    """Result of degradation handling."""

    status: DegradedStatus
    action: DegradedAction
    message: str
    checkpoint_id: str | None = None
    report: FailureReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GracefulDegradation:
    """
    Terminal failure handler when all CLIs exhaust retries.

    Determines appropriate degradation strategy based on workflow phase:
    - PLANNING: Block and require human plan
    - REVIEWING: Continue with partial reviews or block
    - IMPLEMENTING: Rollback and report
    """

    def __init__(self, state_manager: Any = None) -> None:
        """
        Initialize degradation handler.

        Args:
            state_manager: StateManager for checkpoint saving.
        """
        self.state_manager = state_manager

    async def handle_total_failure(
        self,
        phase: WorkflowPhase,
        failed_clis: list[CLIFailure],
        state: WorkflowState,
    ) -> DegradedResult:
        """
        Handle the case where all CLIs have failed.

        Args:
            phase: Current workflow phase.
            failed_clis: List of CLI failures.
            state: Current workflow state.

        Returns:
            DegradedResult with appropriate action.
        """
        # Generate failure report
        report = self._generate_failure_report(phase, failed_clis)

        # Save checkpoint for recovery
        checkpoint_id = None
        if self.state_manager:
            checkpoint_id = await self.state_manager.save_checkpoint(
                state, reason="total_cli_failure"
            )
            state.checkpoint_id = checkpoint_id

        # Determine strategy based on phase
        if phase == WorkflowPhase.PLANNING:
            return self._handle_planning_failure(checkpoint_id, report)

        elif phase == WorkflowPhase.REVIEWING:
            return self._handle_reviewing_failure(state, checkpoint_id, report)

        elif phase == WorkflowPhase.IMPLEMENTING:
            return self._handle_implementing_failure(checkpoint_id, report)

        else:
            # Generic failure handling for other phases
            return DegradedResult(
                status=DegradedStatus.BLOCKED,
                action=DegradedAction.ROLLBACK_AND_REPORT,
                message=f"All CLIs failed during {phase.value}. Manual intervention required.",
                checkpoint_id=checkpoint_id,
                report=report,
            )

    def _handle_planning_failure(
        self,
        checkpoint_id: str | None,
        report: FailureReport,
    ) -> DegradedResult:
        """Handle failure during planning phase."""
        logger.error("All planners failed - human intervention required")

        return DegradedResult(
            status=DegradedStatus.BLOCKED,
            action=DegradedAction.HUMAN_PLAN_REQUIRED,
            message="All planners failed. Manual plan required.",
            checkpoint_id=checkpoint_id,
            report=report,
        )

    def _handle_reviewing_failure(
        self,
        state: WorkflowState,
        checkpoint_id: str | None,
        report: FailureReport,
    ) -> DegradedResult:
        """Handle failure during reviewing phase."""
        # Can continue with partial reviews if we have at least one
        if len(state.review_rounds) >= 1:
            logger.warning(
                "Some reviewers failed. Continuing with %d partial reviews.",
                len(state.review_rounds),
            )

            return DegradedResult(
                status=DegradedStatus.DEGRADED,
                action=DegradedAction.CONTINUE_DEGRADED,
                message=f"Continuing with {len(state.review_rounds)} partial reviews",
                checkpoint_id=checkpoint_id,
                report=report,
                metadata={"partial_reviews": len(state.review_rounds)},
            )
        else:
            logger.error("No reviews obtained - human review required")

            return DegradedResult(
                status=DegradedStatus.BLOCKED,
                action=DegradedAction.HUMAN_REVIEW_REQUIRED,
                message="No reviews obtained. Human review required.",
                checkpoint_id=checkpoint_id,
                report=report,
            )

    def _handle_implementing_failure(
        self,
        checkpoint_id: str | None,
        report: FailureReport,
    ) -> DegradedResult:
        """Handle failure during implementing phase."""
        logger.error("Implementation failed - rollback required")

        return DegradedResult(
            status=DegradedStatus.FAILED,
            action=DegradedAction.ROLLBACK_AND_REPORT,
            message="Implementation failed. State saved for retry.",
            checkpoint_id=checkpoint_id,
            report=report,
        )

    def _generate_failure_report(
        self,
        phase: WorkflowPhase,
        failed_clis: list[CLIFailure],
    ) -> FailureReport:
        """Generate a human-readable failure report."""
        # Categorize failures
        timeouts = [f for f in failed_clis if f.error_type == "timeout"]
        rate_limits = [f for f in failed_clis if f.error_type == "rate_limited"]
        auth_errors = [f for f in failed_clis if f.error_type == "auth_error"]
        other_errors = [f for f in failed_clis if f.error_type not in ("timeout", "rate_limited", "auth_error")]

        # Build summary
        parts = []
        if timeouts:
            parts.append(f"{len(timeouts)} CLI(s) timed out")
        if rate_limits:
            parts.append(f"{len(rate_limits)} CLI(s) rate limited")
        if auth_errors:
            parts.append(f"{len(auth_errors)} CLI(s) have auth errors")
        if other_errors:
            parts.append(f"{len(other_errors)} CLI(s) encountered errors")

        summary = "; ".join(parts) if parts else "Unknown failure"

        # Build recommendations
        recommendations = []
        if timeouts:
            recommendations.append("Consider increasing timeout values in config")
        if rate_limits:
            recommendations.append("Wait for rate limits to reset, or reduce request frequency")
        if auth_errors:
            recommendations.append("Re-authenticate CLIs with their respective auth commands")
        if other_errors:
            recommendations.append("Check CLI installation and network connectivity")

        recommendations.append(f"Resume with: --resume {phase.value}")

        return FailureReport(
            phase=phase,
            failures=failed_clis,
            summary=summary,
            recommendations=recommendations,
        )

    def print_failure_ui(self, result: DegradedResult) -> None:
        """Print a formatted failure UI to console."""
        print()
        print("+" + "=" * 61 + "+")
        print("|  [!] ALL CLIs FAILED - HUMAN INTERVENTION REQUIRED" + " " * 9 + "|")
        print("+" + "=" * 61 + "+")

        if result.checkpoint_id:
            print(f"|  Checkpoint saved: {result.checkpoint_id:<40} |")

        print("|" + " " * 62 + "|")
        print(f"|  Status: {result.status.value:<51} |")
        print(f"|  Action: {result.action.value:<51} |")
        print("|" + " " * 62 + "|")

        if result.report:
            print("|  Failed CLIs:" + " " * 47 + "|")
            for failure in result.report.failures:
                line = f"    - {failure.cli_name}: {failure.error_type}"
                print(f"|{line:<62}|")

        print("|" + " " * 62 + "|")
        print("|  Options:" + " " * 51 + "|")
        if result.checkpoint_id:
            print(f"|    1. Fix issues and run: --resume {result.checkpoint_id[:20]}... |")
        print("|    2. Provide manual input: --manual-plan plan.md" + " " * 11 + "|")
        print("|    3. Abort workflow: --abort" + " " * 31 + "|")
        print("+" + "=" * 61 + "+")
        print()
