"""Human-in-the-loop decision gates for strategic workflow points."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Callable

from ai_orchestrator.core.workflow_phases import Plan, WorkflowPhase, WorkflowState

logger = logging.getLogger(__name__)


class DecisionPriority(str, Enum):
    """Priority level for human decisions."""

    HIGH = "high"  # Must review before proceeding
    MEDIUM = "medium"  # Should review, but can auto-proceed
    LOW = "low"  # FYI only, auto-proceeds


class DecisionType(str, Enum):
    """Type of decision being requested."""

    APPROVAL = "approval"  # Simple yes/no
    CHOICE = "choice"  # Select from options
    INPUT = "input"  # Free-form input
    REVIEW = "review"  # Review and optionally modify


class GateTrigger(str, Enum):
    """Trigger conditions for decision gates."""

    AFTER_PLAN_SYNTHESIS = "after_plan_synthesis"
    AFTER_FINAL_ITERATION = "after_final_iteration"
    BEFORE_IMPLEMENTING = "before_implementing"
    AFTER_POST_CHECKS_FAILURE = "after_post_checks_failure"
    SECURITY_CHANGE_DETECTED = "security_change_detected"
    HIGH_BLAST_RADIUS = "high_blast_radius"


@dataclass
class DecisionOption:
    """An option for a decision."""

    key: str
    label: str
    description: str = ""
    is_recommended: bool = False


@dataclass
class DecisionRequest:
    """A request for human decision."""

    gate: GateTrigger
    decision_type: DecisionType
    priority: DecisionPriority
    title: str
    description: str
    options: list[DecisionOption] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None  # None = no timeout
    auto_approve_on_timeout: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DecisionResponse:
    """Human's response to a decision request."""

    approved: bool
    selected_option: str | None = None
    input_value: str | None = None
    modifications: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    responded_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    was_timeout: bool = False


@dataclass
class GateConfig:
    """Configuration for decision gates."""

    # Which gates are enabled
    enabled_gates: set[GateTrigger] = field(
        default_factory=lambda: {
            GateTrigger.AFTER_PLAN_SYNTHESIS,
            GateTrigger.AFTER_POST_CHECKS_FAILURE,
        }
    )

    # Conditional gates (only trigger if condition met)
    conditional_gates: dict[GateTrigger, str] = field(
        default_factory=lambda: {
            GateTrigger.AFTER_FINAL_ITERATION: "critical_issues_remain",
            GateTrigger.BEFORE_IMPLEMENTING: "security_or_auth_changes",
        }
    )

    # Auto-approve settings
    auto_approve_timeout: float = 300.0  # 5 minutes
    auto_approve_on_low_priority: bool = True

    # UI settings
    interactive: bool = True  # False for CI/automated runs


class DecisionGates:
    """
    Human-in-the-loop decision gates at strategic workflow points.

    Strategic Decision Points:
    1. After PLAN_SYNTHESIS: Review plan before review cycles
    2. After FINAL_ITERATION: When critical issues remain
    3. Before IMPLEMENTING: For auth/security changes
    4. After POST_CHECKS_FAILURE: When automated checks fail
    """

    def __init__(
        self,
        config: GateConfig | None = None,
        input_handler: Callable[[DecisionRequest], DecisionResponse] | None = None,
    ) -> None:
        """
        Initialize decision gates.

        Args:
            config: Gate configuration.
            input_handler: Custom input handler for testing/automation.
        """
        self.config = config or GateConfig()
        self._input_handler = input_handler or self._default_input_handler
        self._decision_log: list[tuple[DecisionRequest, DecisionResponse]] = []

    async def check_gate(
        self,
        gate: GateTrigger,
        state: WorkflowState,
        context: dict[str, Any] | None = None,
    ) -> DecisionResponse:
        """
        Check if a gate should trigger and handle the decision.

        Args:
            gate: The gate to check.
            state: Current workflow state.
            context: Additional context for the decision.

        Returns:
            DecisionResponse with the human's decision.
        """
        # Check if gate is enabled
        if not self._is_gate_active(gate, state):
            return DecisionResponse(approved=True, reason="Gate not active")

        # Build decision request
        request = self._build_request(gate, state, context or {})

        # Handle low priority auto-approval
        if (
            request.priority == DecisionPriority.LOW
            and self.config.auto_approve_on_low_priority
        ):
            response = DecisionResponse(
                approved=True,
                reason="Auto-approved (low priority)",
            )
            self._decision_log.append((request, response))
            return response

        # Get human decision
        if self.config.interactive:
            response = await self._get_interactive_decision(request)
        else:
            # Non-interactive mode - auto-approve or fail based on config
            response = self._get_automated_decision(request)

        self._decision_log.append((request, response))
        logger.info(
            "Decision gate %s: %s",
            gate.value,
            "approved" if response.approved else "rejected",
        )

        return response

    def _is_gate_active(self, gate: GateTrigger, state: WorkflowState) -> bool:
        """Check if a gate should be triggered."""
        # Check enabled gates
        if gate in self.config.enabled_gates:
            return True

        # Check conditional gates
        condition = self.config.conditional_gates.get(gate)
        if condition:
            return self._evaluate_condition(condition, state)

        return False

    def _evaluate_condition(self, condition: str, state: WorkflowState) -> bool:
        """Evaluate a conditional gate trigger."""
        if condition == "critical_issues_remain":
            if state.latest_review_round:
                return state.latest_review_round.critical_count > 0
            return False

        if condition == "security_or_auth_changes":
            # Check if plan mentions security/auth changes
            if state.current_plan:
                content_lower = state.current_plan.content.lower()
                security_keywords = ["auth", "security", "permission", "token", "credential"]
                return any(kw in content_lower for kw in security_keywords)
            return False

        return False

    def _build_request(
        self,
        gate: GateTrigger,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> DecisionRequest:
        """Build a decision request for a gate."""
        if gate == GateTrigger.AFTER_PLAN_SYNTHESIS:
            return self._build_plan_approval_request(state, context)

        if gate == GateTrigger.AFTER_FINAL_ITERATION:
            return self._build_iteration_decision_request(state, context)

        if gate == GateTrigger.BEFORE_IMPLEMENTING:
            return self._build_implementation_approval_request(state, context)

        if gate == GateTrigger.AFTER_POST_CHECKS_FAILURE:
            return self._build_post_checks_decision_request(state, context)

        # Default request
        return DecisionRequest(
            gate=gate,
            decision_type=DecisionType.APPROVAL,
            priority=DecisionPriority.MEDIUM,
            title=f"Decision Required: {gate.value}",
            description="Please review and approve to continue.",
            context=context,
        )

    def _build_plan_approval_request(
        self,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> DecisionRequest:
        """Build request for plan approval."""
        plan_preview = ""
        if state.current_plan:
            plan_preview = state.current_plan.content[:500] + "..."

        return DecisionRequest(
            gate=GateTrigger.AFTER_PLAN_SYNTHESIS,
            decision_type=DecisionType.APPROVAL,
            priority=DecisionPriority.HIGH,
            title="Review Implementation Plan",
            description=(
                "The following plan has been synthesized from multiple planners. "
                "Please review before proceeding to the review phase."
            ),
            options=[
                DecisionOption(
                    key="approve",
                    label="Approve Plan",
                    description="Proceed with this plan",
                    is_recommended=True,
                ),
                DecisionOption(
                    key="modify",
                    label="Request Modifications",
                    description="Provide feedback for plan adjustment",
                ),
                DecisionOption(
                    key="reject",
                    label="Reject Plan",
                    description="Start over with new planning",
                ),
            ],
            context={
                "plan_preview": plan_preview,
                "consensus_score": context.get("consensus_score", "N/A"),
                **context,
            },
        )

    def _build_iteration_decision_request(
        self,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> DecisionRequest:
        """Build request for iteration decision when issues remain."""
        critical_count = 0
        if state.latest_review_round:
            critical_count = state.latest_review_round.critical_count

        return DecisionRequest(
            gate=GateTrigger.AFTER_FINAL_ITERATION,
            decision_type=DecisionType.CHOICE,
            priority=DecisionPriority.HIGH,
            title="Critical Issues Remain",
            description=(
                f"After {state.current_iteration} iterations, "
                f"{critical_count} critical issue(s) remain unresolved."
            ),
            options=[
                DecisionOption(
                    key="continue",
                    label="Continue Iterations",
                    description="Try more review/fix cycles",
                ),
                DecisionOption(
                    key="proceed",
                    label="Proceed Anyway",
                    description="Accept current state and implement",
                ),
                DecisionOption(
                    key="manual",
                    label="Fix Manually",
                    description="Take over and fix issues manually",
                ),
                DecisionOption(
                    key="abort",
                    label="Abort Workflow",
                    description="Stop the workflow entirely",
                ),
            ],
            context={
                "iteration": state.current_iteration,
                "critical_count": critical_count,
                **context,
            },
        )

    def _build_implementation_approval_request(
        self,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> DecisionRequest:
        """Build request for implementation approval."""
        return DecisionRequest(
            gate=GateTrigger.BEFORE_IMPLEMENTING,
            decision_type=DecisionType.APPROVAL,
            priority=DecisionPriority.HIGH,
            title="Approve Implementation",
            description=(
                "The plan involves security or authentication changes. "
                "Please confirm before implementation begins."
            ),
            options=[
                DecisionOption(
                    key="approve",
                    label="Approve Implementation",
                    description="Proceed with implementation",
                    is_recommended=True,
                ),
                DecisionOption(
                    key="review",
                    label="Review Plan Again",
                    description="Go back to review phase",
                ),
                DecisionOption(
                    key="abort",
                    label="Abort",
                    description="Stop the workflow",
                ),
            ],
            context=context,
        )

    def _build_post_checks_decision_request(
        self,
        state: WorkflowState,
        context: dict[str, Any],
    ) -> DecisionRequest:
        """Build request when POST_CHECKS fail."""
        failed_gates = context.get("failed_gates", [])

        return DecisionRequest(
            gate=GateTrigger.AFTER_POST_CHECKS_FAILURE,
            decision_type=DecisionType.CHOICE,
            priority=DecisionPriority.HIGH,
            title="Post-Implementation Checks Failed",
            description=f"The following checks failed: {', '.join(failed_gates)}",
            options=[
                DecisionOption(
                    key="fix",
                    label="Attempt Auto-Fix",
                    description="Try to automatically fix the issues",
                    is_recommended=True,
                ),
                DecisionOption(
                    key="manual",
                    label="Fix Manually",
                    description="Take over and fix issues manually",
                ),
                DecisionOption(
                    key="proceed",
                    label="Proceed Anyway",
                    description="Accept failures and continue",
                ),
                DecisionOption(
                    key="rollback",
                    label="Rollback Changes",
                    description="Revert all implementation changes",
                ),
            ],
            context={
                "failed_gates": failed_gates,
                **context,
            },
        )

    async def _get_interactive_decision(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        """Get decision interactively from the user."""
        self._print_decision_ui(request)

        # Wait for input with optional timeout
        try:
            if request.timeout_seconds:
                response = await asyncio.wait_for(
                    self._wait_for_input(request),
                    timeout=request.timeout_seconds,
                )
            else:
                response = await self._wait_for_input(request)

            return response

        except asyncio.TimeoutError:
            if request.auto_approve_on_timeout:
                return DecisionResponse(
                    approved=True,
                    reason="Auto-approved (timeout)",
                    was_timeout=True,
                )
            return DecisionResponse(
                approved=False,
                reason="Timeout waiting for decision",
                was_timeout=True,
            )

    async def _wait_for_input(self, request: DecisionRequest) -> DecisionResponse:
        """Wait for user input."""
        # Use custom handler if provided
        if self._input_handler:
            return self._input_handler(request)

        # Default: read from stdin
        return await self._read_stdin_decision(request)

    async def _read_stdin_decision(
        self,
        request: DecisionRequest,
    ) -> DecisionResponse:
        """Read decision from stdin."""
        loop = asyncio.get_event_loop()

        # Run blocking input in thread pool
        def get_input() -> str:
            return input("\nEnter your choice: ").strip().lower()

        choice = await loop.run_in_executor(None, get_input)

        # Map choice to response
        if request.decision_type == DecisionType.APPROVAL:
            approved = choice in ("y", "yes", "approve", "1")
            return DecisionResponse(
                approved=approved,
                selected_option="approve" if approved else "reject",
            )

        if request.options:
            # Find matching option
            for opt in request.options:
                if choice in (opt.key.lower(), opt.label.lower()):
                    approved = opt.key not in ("reject", "abort", "rollback")
                    return DecisionResponse(
                        approved=approved,
                        selected_option=opt.key,
                    )

        # Default to first option if no match
        if request.options:
            return DecisionResponse(
                approved=True,
                selected_option=request.options[0].key,
            )

        return DecisionResponse(approved=True)

    def _get_automated_decision(self, request: DecisionRequest) -> DecisionResponse:
        """Get automated decision for non-interactive mode."""
        # In automated mode, auto-approve unless high priority
        if request.priority == DecisionPriority.HIGH:
            # For high priority, reject to force manual intervention
            return DecisionResponse(
                approved=False,
                reason="High priority gate requires manual intervention",
            )

        return DecisionResponse(
            approved=True,
            reason="Auto-approved (non-interactive mode)",
        )

    def _print_decision_ui(self, request: DecisionRequest) -> None:
        """Print decision UI to console."""
        print()
        print("+" + "=" * 61 + "+")
        print(f"|  [{request.priority.value.upper()}] {request.title:<47} |")
        print("+" + "=" * 61 + "+")
        print()

        # Print description
        for line in request.description.split("\n"):
            print(f"  {line}")
        print()

        # Print context if relevant
        if request.context:
            print("  Context:")
            for key, value in request.context.items():
                if key not in ("plan_preview",):  # Skip large items
                    print(f"    {key}: {value}")
            print()

        # Print options
        if request.options:
            print("  Options:")
            for i, opt in enumerate(request.options, 1):
                rec = " (Recommended)" if opt.is_recommended else ""
                print(f"    [{i}] {opt.label}{rec}")
                if opt.description:
                    print(f"        {opt.description}")
            print()

        if request.decision_type == DecisionType.APPROVAL:
            print("  Enter 'y' to approve, 'n' to reject")

        print("+" + "-" * 61 + "+")

    def _default_input_handler(self, request: DecisionRequest) -> DecisionResponse:
        """Default synchronous input handler."""
        # This is a fallback - prefer async methods
        return DecisionResponse(
            approved=True,
            reason="Default handler - auto-approved",
        )

    def get_decision_log(self) -> list[tuple[DecisionRequest, DecisionResponse]]:
        """Get the log of all decisions made."""
        return self._decision_log.copy()

    def clear_decision_log(self) -> None:
        """Clear the decision log."""
        self._decision_log.clear()
