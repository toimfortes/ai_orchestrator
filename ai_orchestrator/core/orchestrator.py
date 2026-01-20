"""Main orchestration engine for multi-AI code generation workflow.

This orchestrator supports concurrent multi-CLI invocation for maximum efficiency:
- Multiple planners run in parallel during MULTI_PLANNING phase
- Multiple reviewers run in parallel during MULTI_REVIEWING phase
- Semaphore limits concurrent CLI processes to prevent overload
- Circuit breakers prevent repeated failures to failing CLIs
- 5-layer error recovery: Retry → Fallback → Circuit Breaker → Graceful Degradation → Human Escalation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

from ai_orchestrator.cli_adapters import (
    CLIAdapter,
    CLIResult,
    CLIStatus,
    get_adapter,
    get_available_adapters,
)
from ai_orchestrator.config.settings import Settings, get_settings
from ai_orchestrator.core.iteration_controller import (
    AdaptiveIterationController,
    IterationConfig,
)
from ai_orchestrator.core.state_manager import StateManager
from ai_orchestrator.core.workflow_phases import (
    ClassifiedFeedback as WorkflowFeedback,
    IssueCategory as WorkflowCategory,
    Plan,
    Severity as WorkflowSeverity,
    WorkflowPhase,
    WorkflowState,
)
from ai_orchestrator.project.context import ProjectContext
from ai_orchestrator.project.loader import load_project_context
from ai_orchestrator.project.scaffolder import FoundationScaffolder
from ai_orchestrator.reviewing.feedback_classifier import (
    FeedbackClassifier,
    IssueSeverity,
    IssueCategory,
    ClassificationResult,
)
from ai_orchestrator.utils import truncate_with_marker

logger = logging.getLogger(__name__)


# Circuit breaker states
class CircuitState(str, Enum):
    """Circuit breaker state tracking."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class ErrorType(str, Enum):
    """Error taxonomy for circuit breaker decisions."""
    TIMEOUT = "timeout"           # Request timed out
    RATE_LIMITED = "rate_limited"  # 429 errors
    SERVER_ERROR = "server_error"  # 5xx errors
    AUTH_ERROR = "auth_error"      # Authentication failures
    CLIENT_ERROR = "client_error"  # 4xx errors (non-retriable)
    UNKNOWN = "unknown"           # Unknown errors


# Error weights for circuit breaker (higher = more severe)
ERROR_WEIGHTS: dict[ErrorType, float] = {
    ErrorType.TIMEOUT: 1.0,
    ErrorType.RATE_LIMITED: 0.5,  # Less severe, often transient
    ErrorType.SERVER_ERROR: 1.0,
    ErrorType.AUTH_ERROR: 2.0,   # More severe, likely persistent
    ErrorType.CLIENT_ERROR: 0.0,  # Don't count client errors
    ErrorType.UNKNOWN: 1.0,
}


@dataclass
class CircuitBreaker:
    """
    Enhanced circuit breaker with jitter, error taxonomy, and persistence.

    Improvements over basic implementation:
    - Decorrelated jitter prevents thundering herd on reset
    - Error taxonomy weights different error types
    - Exponential backoff for consecutive open periods
    - State persistence survives process restarts
    - Rolling window for failure counting (optional)
    """

    name: str
    fail_threshold: float = 3.0  # Weighted threshold
    base_reset_timeout: float = 60.0
    max_reset_timeout: float = 300.0  # 5 minute max
    jitter_factor: float = 0.5  # 0-50% random jitter
    state_file: Path | None = None

    # State tracking
    state: str = field(default=CircuitState.CLOSED)
    weighted_failure_count: float = 0.0
    failure_count: int = 0  # Raw count for logging
    last_failure_time: datetime | None = None
    consecutive_opens: int = 0  # For exponential backoff
    last_error_type: ErrorType | None = None

    def __post_init__(self) -> None:
        """Load persisted state if available."""
        if self.state_file and self.state_file.exists():
            self._load_state()

    def record_success(self) -> None:
        """Record successful invocation."""
        was_half_open = self.state == CircuitState.HALF_OPEN

        self.weighted_failure_count = 0.0
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_error_type = None

        if was_half_open:
            # Successful probe - reset consecutive opens
            self.consecutive_opens = 0
            logger.info("Circuit breaker CLOSED for %s after successful probe", self.name)

        self._save_state()

    def record_failure(self, error_type: ErrorType = ErrorType.UNKNOWN) -> None:
        """
        Record failed invocation with error taxonomy.

        Args:
            error_type: Type of error for weighted counting.
        """
        weight = ERROR_WEIGHTS.get(error_type, 1.0)
        self.weighted_failure_count += weight
        self.failure_count += 1
        self.last_failure_time = datetime.now(UTC)
        self.last_error_type = error_type

        if self.weighted_failure_count >= self.fail_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.consecutive_opens += 1
                logger.warning(
                    "Circuit breaker OPEN for %s: %d failures (weighted: %.1f), "
                    "last error: %s, consecutive opens: %d",
                    self.name,
                    self.failure_count,
                    self.weighted_failure_count,
                    error_type.value,
                    self.consecutive_opens,
                )

        self._save_state()

    def is_available(self) -> bool:
        """Check if CLI is available (circuit not open)."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if reset timeout (with jitter) has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
                reset_timeout = self._get_reset_timeout_with_jitter()

                if elapsed >= reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit breaker HALF_OPEN for %s after %.1fs (timeout was %.1fs)",
                        self.name, elapsed, reset_timeout
                    )
                    self._save_state()
                    return True
            return False

        # Half-open: allow one request (probe)
        return True

    def _get_reset_timeout_with_jitter(self) -> float:
        """
        Calculate reset timeout with exponential backoff and decorrelated jitter.

        Uses AWS-recommended decorrelated jitter pattern to prevent
        thundering herd when multiple breakers reset simultaneously.
        """
        # Exponential backoff based on consecutive opens
        backoff_multiplier = min(2 ** self.consecutive_opens, 8)  # Cap at 8x
        base_timeout = min(
            self.base_reset_timeout * backoff_multiplier,
            self.max_reset_timeout
        )

        # Decorrelated jitter: random value between 0 and jitter_factor * base
        jitter = random.uniform(0, self.jitter_factor * base_timeout)

        return base_timeout + jitter

    def _save_state(self) -> None:
        """Persist circuit breaker state to file."""
        if not self.state_file:
            return

        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                "state": self.state if isinstance(self.state, str) else self.state.value,
                "weighted_failure_count": self.weighted_failure_count,
                "failure_count": self.failure_count,
                "consecutive_opens": self.consecutive_opens,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_error_type": self.last_error_type.value if self.last_error_type else None,
                "saved_at": datetime.now(UTC).isoformat(),
            }

            # Atomic write: temp file + fsync + rename
            tmp_file = self.state_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(state_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_file.replace(self.state_file)

        except OSError as e:
            logger.warning("Failed to save circuit breaker state for %s: %s", self.name, e)

    def _load_state(self) -> None:
        """Load circuit breaker state from file."""
        if not self.state_file or not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state_data = json.load(f)

            # Check if state is stale (older than max reset timeout)
            saved_at = state_data.get("saved_at")
            if saved_at:
                saved_time = datetime.fromisoformat(saved_at)
                age_seconds = (datetime.now(UTC) - saved_time).total_seconds()

                if age_seconds > self.max_reset_timeout:
                    # State is stale, start fresh
                    logger.info(
                        "Circuit breaker state for %s is stale (%.0fs old), starting fresh",
                        self.name, age_seconds
                    )
                    return

            # Restore state
            self.state = state_data.get("state", CircuitState.CLOSED)
            self.weighted_failure_count = state_data.get("weighted_failure_count", 0.0)
            self.failure_count = state_data.get("failure_count", 0)
            self.consecutive_opens = state_data.get("consecutive_opens", 0)

            if state_data.get("last_failure_time"):
                self.last_failure_time = datetime.fromisoformat(state_data["last_failure_time"])

            if state_data.get("last_error_type"):
                self.last_error_type = ErrorType(state_data["last_error_type"])

            logger.info(
                "Restored circuit breaker state for %s: state=%s, failures=%d",
                self.name, self.state, self.failure_count
            )

        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to load circuit breaker state for %s: %s", self.name, e)


@dataclass
class CLIInvocationResult:
    """Result of invoking a CLI with metadata."""
    cli_name: str
    result: CLIResult | None
    success: bool
    error: str | None = None
    duration_seconds: float = 0.0
    retries_used: int = 0


class Orchestrator:
    """
    Main orchestration engine for multi-AI code generation workflow.

    Manages the workflow phases:
    INIT → PLANNING → REVIEWING → FIXING → IMPLEMENTING → POST_CHECKS → COMPLETED

    Supports concurrent multi-CLI execution:
    - Multiple planners generate plans in parallel (MULTI_PLANNING)
    - Multiple reviewers review in parallel (MULTI_REVIEWING)
    - Semaphore limits concurrent processes
    - Circuit breakers prevent cascading failures
    - 5-layer error recovery for resilience

    Optional incremental review:
    - Micro-reviews after each file/commit change during implementation
    - Any available agent can act as reviewer (default: gemini for cost efficiency)
    - Immediate feedback to coding agent for critical issues
    """

    # Default fallback chains per phase
    FALLBACK_CHAINS: dict[str, list[str]] = {
        "planning": ["claude", "codex", "gemini"],
        "reviewing": ["claude", "gemini", "codex"],
        "implementing": ["claude"],  # Claude-only for implementation
    }

    def __init__(
        self,
        project_path: Path,
        settings: Settings | None = None,
        max_concurrent_clis: int = 5,
        incremental_review: bool = False,
        review_agent: str = "gemini",
        review_granularity: str = "file",
        review_threshold: int = 10,
    ) -> None:
        self.project_path = project_path.resolve()
        self.settings = settings or get_settings()
        self.max_concurrent = max_concurrent_clis

        # Incremental review configuration
        self.incremental_review_enabled = incremental_review
        self.review_agent = review_agent
        self.review_granularity = review_granularity
        self.review_threshold = review_threshold

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_clis)

        # Initialize components
        self.state_manager = StateManager(self.project_path)
        self.iteration_controller = AdaptiveIterationController(
            IterationConfig(
                max_iterations=self.settings.iteration.max_iterations,
                consecutive_clean_rounds_required=self.settings.iteration.consecutive_clean_required,
            )
        )

        # CLI adapters - all available
        self.adapters: dict[str, CLIAdapter] = {}
        self._init_adapters()

        # Circuit breakers per CLI with persistence
        state_dir = self.project_path / ".ai_orchestrator" / "circuit_state"
        self.circuit_breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(
                name=name,
                state_file=state_dir / f"{name}.json",
            )
            for name in self.adapters
        }

        # Incremental reviewer (initialized lazily)
        self._incremental_reviewer: Any = None

        # Project context (discovered or loaded)
        self.project_context: ProjectContext | None = None

        # Current state
        self.state: WorkflowState | None = None

    def _get_incremental_reviewer(self) -> Any:
        """Get or create the incremental reviewer."""
        if self._incremental_reviewer is None and self.incremental_review_enabled:
            from ai_orchestrator.reviewing.incremental_reviewer import (
                IncrementalReviewer,
                ReviewGranularity,
            )
            self._incremental_reviewer = IncrementalReviewer(
                project_path=self.project_path,
                review_agent=self.review_agent,
                granularity=ReviewGranularity(self.review_granularity),
                min_lines_threshold=self.review_threshold,
                adapters=self.adapters,
            )
        return self._incremental_reviewer

    def _init_adapters(self) -> None:
        """Initialize all available CLI adapters."""
        # Get all available adapters
        available = get_available_adapters()

        # Filter by enabled CLIs in settings
        for cli_name in self.settings.enabled_clis:
            if cli_name in available:
                adapter = available[cli_name]
                self.adapters[cli_name] = adapter
                logger.info("Initialized %s adapter", cli_name)
            else:
                logger.debug("CLI %s not available or not enabled", cli_name)

        # Log summary
        logger.info(
            "CLI adapters initialized: %s",
            ", ".join(self.adapters.keys()) or "none"
        )

    async def initialize(self) -> None:
        """Initialize the orchestrator (load project context, state)."""
        # Load project context
        self.project_context = await load_project_context(self.project_path)
        logger.info("Project context loaded:\n%s", self.project_context.summary())

        # Try to load existing state
        self.state = await self.state_manager.load_state()
        if self.state:
            logger.info(
                "Resumed from state: phase=%s, iteration=%d",
                self.state.current_phase.value,
                self.state.current_iteration,
            )

    async def run(
        self,
        prompt: str,
        *,
        resume: bool = False,
        dry_run: bool = False,
        plan_only: bool = False,
        max_iterations: int | None = None,
    ) -> WorkflowState:
        """
        Run the orchestration workflow.

        Args:
            prompt: The task prompt.
            resume: Whether to resume from saved state.
            dry_run: If True, don't actually invoke CLIs.
            plan_only: If True, stop after planning phase.
            max_iterations: Override max iterations.

        Returns:
            Final workflow state.
        """
        # Initialize if needed
        if self.project_context is None:
            await self.initialize()

        # Create or resume state
        if resume and self.state:
            logger.info("Resuming workflow from %s", self.state.current_phase.value)
        else:
            self.state = WorkflowState(
                prompt=prompt,
                project_path=str(self.project_path),
            )
            logger.info("Starting new workflow")

        # Override max iterations if specified
        if max_iterations:
            self.iteration_controller.config.max_iterations = max_iterations

        # Dry run mode
        if dry_run:
            return await self._dry_run()

        # Run workflow phases
        try:
            await self._run_workflow(plan_only=plan_only)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            # Let these propagate without wrapping
            raise
        except Exception as e:
            logger.error("Workflow failed: %s", e, exc_info=True)
            self.state.add_error(str(e))
            self.state.transition_to(WorkflowPhase.FAILED)
            await self.state_manager.save_state_atomic(self.state)
            raise

        return self.state

    async def _run_workflow(self, plan_only: bool = False) -> None:
        """Execute the workflow state machine."""
        assert self.state is not None

        while True:
            phase = self.state.current_phase

            if phase == WorkflowPhase.INIT:
                await self._phase_init()

            elif phase == WorkflowPhase.SCAFFOLDING:
                await self._phase_scaffolding()

            elif phase == WorkflowPhase.PLANNING:
                await self._phase_planning()
                if plan_only:
                    logger.info("Plan-only mode: stopping after planning")
                    break

            elif phase == WorkflowPhase.REVIEWING:
                await self._phase_reviewing()

            elif phase == WorkflowPhase.FIXING:
                await self._phase_fixing()

            elif phase == WorkflowPhase.IMPLEMENTING:
                await self._phase_implementing()

            elif phase == WorkflowPhase.POST_CHECKS:
                await self._phase_post_checks()

            elif phase == WorkflowPhase.COMPLETED:
                logger.info("Workflow completed successfully")
                break

            elif phase == WorkflowPhase.FAILED:
                logger.error("Workflow in FAILED state")
                break

            else:
                # Skip unimplemented phases for MVP
                logger.debug("Skipping phase: %s", phase.value)
                self._advance_phase()

            # Save state after each phase
            await self.state_manager.save_state_atomic(self.state)

    async def _phase_init(self) -> None:
        """Initialize phase: prepare context and validate."""
        logger.info("=== INIT PHASE ===")

        # Validate we have at least one working CLI
        if not self.adapters:
            raise RuntimeError("No CLI adapters available")

        # Check authentication
        for name, adapter in self.adapters.items():
            if await adapter.check_auth():
                logger.info("CLI %s: authenticated", name)
            else:
                logger.warning(
                    "CLI %s: not authenticated. Run: %s",
                    name,
                    adapter.get_auth_command(),
                )

        # Check if this is a greenfield project that needs scaffolding
        if self.project_context and self.project_context.needs_foundation_scaffold:
            logger.info("Greenfield project detected - scaffolding foundations")
            self.state.transition_to(WorkflowPhase.SCAFFOLDING)
        else:
            self.state.transition_to(WorkflowPhase.PLANNING)

    async def _phase_scaffolding(self) -> None:
        """
        Scaffolding phase: set up foundations for greenfield projects.

        Creates:
        - CLAUDE.md with project-specific coding standards
        - scripts/ directory with workflow scripts (measure twice, post-check, debug)
        - best_practices/ with patterns.json
        - data/ directory for code catalog
        """
        logger.info("=== SCAFFOLDING PHASE ===")

        scaffolder = FoundationScaffolder(self.project_path)

        try:
            # Scaffold all foundations
            results = await scaffolder.scaffold_all(
                project_name=self.project_path.name,
            )

            # Log what was created
            if results.get("created_files"):
                logger.info(
                    "Created %d files: %s",
                    len(results["created_files"]),
                    ", ".join(results["created_files"]),
                )
            if results.get("created_dirs"):
                logger.info(
                    "Created %d directories: %s",
                    len(results["created_dirs"]),
                    ", ".join(results["created_dirs"]),
                )
            if results.get("skipped"):
                logger.info(
                    "Skipped %d existing items: %s",
                    len(results["skipped"]),
                    ", ".join(results["skipped"]),
                )

            logger.info("Foundation scaffolding complete")

            # Re-discover project context with new foundations
            self.project_context = await load_project_context(self.project_path)
            logger.info("Updated project context:\n%s", self.project_context.summary())

        except Exception as e:
            logger.error("Scaffolding failed: %s", e, exc_info=True)
            self.state.add_error(f"Scaffolding failed: {e}")
            # Continue to planning anyway - scaffolding is helpful but not required

        self.state.transition_to(WorkflowPhase.PLANNING)

    async def _phase_planning(self) -> None:
        """
        Planning phase: generate implementation plans from multiple CLIs.

        Multi-planner concurrent execution:
        - All available planners run in parallel (limited by semaphore)
        - Plans from each CLI are collected
        - Best plan is selected or synthesized
        - Graceful degradation if some planners fail
        """
        logger.info("=== PLANNING PHASE (Multi-Planner) ===")

        # Build planning prompt
        planning_prompt = self._build_planning_prompt()

        # Get available planners
        available_planners = self._get_available_clis_for_phase("planning")

        if not available_planners:
            self.state.add_error("No planning CLIs available")
            self.state.transition_to(WorkflowPhase.FAILED)
            return

        # If multiple planners available, use concurrent invocation
        if len(available_planners) > 1:
            logger.info(
                "Multi-planner mode: invoking %d planners concurrently",
                len(available_planners)
            )

            results = await self._invoke_clis_concurrent(
                available_planners,
                planning_prompt,
                planning_mode=True,
                min_successes=1,
            )

            # Collect successful plans
            for inv_result in results:
                if inv_result.success and inv_result.result:
                    plan = Plan(
                        content=inv_result.result.output,
                        source_cli=inv_result.cli_name,
                    )
                    self.state.plans.append(plan)
                    logger.info("Plan generated from %s", inv_result.cli_name)

        else:
            # Single planner fallback
            cli_name = available_planners[0]
            logger.info("Single-planner mode: using %s", cli_name)

            inv_result = await self._invoke_cli_with_retry(
                cli_name,
                planning_prompt,
                planning_mode=True,
            )

            if inv_result.success and inv_result.result:
                plan = Plan(
                    content=inv_result.result.output,
                    source_cli=inv_result.cli_name,
                )
                self.state.plans.append(plan)
                logger.info("Plan generated from %s", inv_result.cli_name)
            else:
                self.state.add_error(
                    f"Planning failed: {inv_result.error}"
                )
                self.state.transition_to(WorkflowPhase.FAILED)
                return

        # Check if we have at least one plan
        if not self.state.plans:
            self.state.add_error("No plans generated - all planners failed")
            self.state.transition_to(WorkflowPhase.FAILED)
            return

        # Select or synthesize best plan
        # MVP: Use first successful plan
        # Future: CONSENSAGENT plan synthesis
        self.state.synthesized_plan = self.state.plans[0]
        logger.info(
            "Planning complete: %d plans generated, using %s",
            len(self.state.plans),
            self.state.synthesized_plan.source_cli,
        )

        self.state.transition_to(WorkflowPhase.REVIEWING)

    async def _phase_reviewing(self) -> None:
        """
        Reviewing phase: review the plan with multiple reviewers concurrently.

        Multi-reviewer concurrent execution:
        - All available reviewers run in parallel (limited by semaphore)
        - Feedback from each reviewer is collected and classified
        - Convergence is checked using adaptive iteration controller
        - Graceful degradation if some reviewers fail
        """
        logger.info(
            "=== REVIEWING PHASE (Multi-Reviewer, iteration %d) ===",
            self.state.current_iteration + 1
        )

        # Check if we should continue
        decision = await self.iteration_controller.should_continue_reviewing(self.state)

        if not decision.should_continue:
            logger.info(
                "Review complete: %s - %s",
                decision.status.value,
                decision.reason,
            )
            self.state.transition_to(WorkflowPhase.IMPLEMENTING)
            return

        # Build review prompt
        review_prompt = self._build_review_prompt()

        # Get available reviewers
        available_reviewers = self._get_available_clis_for_phase("reviewing")

        if not available_reviewers:
            self.state.add_error("No reviewing CLIs available")
            self.state.transition_to(WorkflowPhase.FAILED)
            return

        all_feedback: list[WorkflowFeedback] = []
        participating_reviewers: list[str] = []

        # If multiple reviewers available, use concurrent invocation
        if len(available_reviewers) > 1:
            logger.info(
                "Multi-reviewer mode: invoking %d reviewers concurrently",
                len(available_reviewers)
            )

            results = await self._invoke_clis_concurrent(
                available_reviewers,
                review_prompt,
                min_successes=1,
            )

            # Collect and classify feedback from all successful reviewers
            for inv_result in results:
                if inv_result.success and inv_result.result:
                    participating_reviewers.append(inv_result.cli_name)

                    # Classify raw output into structured feedback
                    raw_output = inv_result.result.output or ""
                    classified = self._classify_review_output(raw_output, inv_result.cli_name)
                    all_feedback.extend(classified)

                    logger.info(
                        "Classified %d feedback items from %s",
                        len(classified),
                        inv_result.cli_name,
                    )

        else:
            # Single reviewer fallback
            cli_name = available_reviewers[0]
            logger.info("Single-reviewer mode: using %s", cli_name)

            inv_result = await self._invoke_cli_with_retry(
                cli_name,
                review_prompt,
            )

            if inv_result.success and inv_result.result:
                participating_reviewers.append(inv_result.cli_name)

                # Classify raw output into structured feedback
                raw_output = inv_result.result.output or ""
                classified = self._classify_review_output(raw_output, inv_result.cli_name)
                all_feedback.extend(classified)
            else:
                self.state.add_error(f"Review failed: {inv_result.error}")

        # Check if we got any feedback
        if all_feedback:
            self.state.add_review_round(all_feedback, participating_reviewers)

            # Log severity breakdown
            critical_count = sum(1 for f in all_feedback if f.severity == WorkflowSeverity.CRITICAL)
            high_count = sum(1 for f in all_feedback if f.severity == WorkflowSeverity.HIGH)
            blocker_count = sum(1 for f in all_feedback if f.is_blocker)

            logger.info(
                "Review round %d completed: %d reviewers, %d feedback items "
                "(critical=%d, high=%d, blockers=%d)",
                self.state.current_iteration,
                len(participating_reviewers),
                len(all_feedback),
                critical_count,
                high_count,
                blocker_count,
            )
        else:
            logger.warning("No feedback received in review round")

        # Check convergence again
        decision = await self.iteration_controller.should_continue_reviewing(self.state)
        if decision.should_continue:
            self.state.transition_to(WorkflowPhase.FIXING)
        else:
            self.state.transition_to(WorkflowPhase.IMPLEMENTING)

    async def _phase_fixing(self) -> None:
        """
        Fixing phase: address review feedback.

        Generates fixes based on classified feedback from the latest review round.
        Focuses on CRITICAL and HIGH severity issues first.
        """
        logger.info("=== FIXING PHASE ===")

        # Get latest review round
        latest_round = self.state.latest_review_round
        if not latest_round or not latest_round.feedback:
            logger.info("No feedback to fix, skipping to review")
            self.state.transition_to(WorkflowPhase.REVIEWING)
            return

        # Filter for actionable issues (CRITICAL and HIGH)
        issues_to_fix = [
            f for f in latest_round.feedback
            if f.severity in (WorkflowSeverity.CRITICAL, WorkflowSeverity.HIGH)
        ]

        if not issues_to_fix:
            logger.info("No CRITICAL/HIGH issues to fix, continuing review")
            self.state.transition_to(WorkflowPhase.REVIEWING)
            return

        logger.info(
            "Fixing %d issues (CRITICAL: %d, HIGH: %d)",
            len(issues_to_fix),
            sum(1 for f in issues_to_fix if f.severity == WorkflowSeverity.CRITICAL),
            sum(1 for f in issues_to_fix if f.severity == WorkflowSeverity.HIGH),
        )

        # Build fix prompt
        fix_prompt = self._build_fix_prompt(issues_to_fix)

        # Invoke primary CLI with fix prompt
        inv_result = await self._invoke_with_fallback(
            "fixing",
            fix_prompt,
            timeout_seconds=self.settings.get_timeout_for_cli(
                self.settings.default_cli, "implementing"
            ),
        )

        if inv_result.success and inv_result.result:
            logger.info(
                "Fix generation completed by %s",
                inv_result.cli_name
            )
            # Track that we attempted fixes
            self.state.metadata["last_fix_attempt"] = {
                "round": latest_round.round_number,
                "issues_addressed": len(issues_to_fix),
                "cli": inv_result.cli_name,
            }
        else:
            logger.warning(
                "Fix generation failed: %s",
                inv_result.error
            )
            self.state.add_error(f"Fix generation failed: {inv_result.error}")

        # Transition back to reviewing to verify fixes
        self.state.transition_to(WorkflowPhase.REVIEWING)

    async def _phase_implementing(self) -> None:
        """
        Implementing phase: execute the plan with fallback support.

        Implementation uses fallback strategy (not concurrent):
        - Primary CLI attempts implementation
        - On failure, falls back to next CLI in chain
        - Claude is typically the only implementation CLI (most capable)

        If incremental review is enabled:
        - After implementation, detect changes
        - Quick review of changes for critical issues
        - Feed issues back to coding agent for immediate fix
        """
        logger.info("=== IMPLEMENTING PHASE (with Fallback) ===")

        # Initialize incremental reviewer if enabled
        reviewer = self._get_incremental_reviewer()
        if reviewer:
            await reviewer.initialize()
            logger.info(
                "Incremental review enabled: %s reviewing at %s granularity",
                self.review_agent,
                self.review_granularity,
            )

        # Build implementation prompt
        impl_prompt = self._build_implementation_prompt()

        # Use fallback strategy for implementation
        inv_result = await self._invoke_with_fallback(
            "implementing",
            impl_prompt,
            timeout_seconds=self.settings.get_timeout_for_cli(
                self.settings.default_cli, "implementing"
            ),
        )

        if inv_result.success and inv_result.result:
            self.state.implementation_result = inv_result.result.output
            logger.info(
                "Implementation completed by %s",
                inv_result.cli_name
            )

            # Incremental review of changes if enabled
            if reviewer:
                await self._incremental_review_cycle(inv_result.cli_name, reviewer)
        else:
            self.state.add_error(
                f"Implementation failed: {inv_result.error}"
            )

        self.state.transition_to(WorkflowPhase.POST_CHECKS)

    async def _incremental_review_cycle(
        self,
        coding_agent: str,
        reviewer: Any,
        max_fix_attempts: int = 3,
    ) -> None:
        """
        Run incremental review cycle after implementation.

        Detects changes, reviews them, and feeds back critical issues
        to the coding agent for immediate fixing.

        Args:
            coding_agent: Name of CLI that did implementation.
            reviewer: IncrementalReviewer instance.
            max_fix_attempts: Maximum attempts to fix critical issues.
        """
        for attempt in range(max_fix_attempts):
            # Detect what changed
            changes = await reviewer.detect_changes()

            if not changes.has_changes:
                logger.info("No changes detected for incremental review")
                break

            logger.info(
                "Incremental review (attempt %d/%d): %s",
                attempt + 1,
                max_fix_attempts,
                changes.summary(),
            )

            # Quick review of changes
            review_result, feedback = await reviewer.review_and_feedback(changes)

            if not feedback:
                # No blocking issues found
                logger.info(
                    "Incremental review passed: %d issues (none blocking)",
                    len(review_result.issues),
                )
                break

            # Critical issues found - feed back to coding agent
            logger.warning(
                "Incremental review found %d blocking issues, requesting fix",
                len([i for i in review_result.issues if i.severity.value in ("critical", "high")]),
            )

            # Build fix prompt
            fix_prompt = f"""The following issues were found in your recent implementation changes:

{feedback}

Please fix these issues immediately. Focus only on the issues listed above.
Do not make any other changes."""

            # Invoke coding agent to fix
            fix_result = await self._invoke_cli_with_retry(
                coding_agent,
                fix_prompt,
                max_retries=1,
            )

            if not fix_result.success:
                logger.error("Failed to fix issues: %s", fix_result.error)
                self.state.add_error(f"Incremental fix failed: {fix_result.error}")
                break

            logger.info("Fix attempt %d completed", attempt + 1)

        # Log final stats
        if reviewer:
            stats = reviewer.get_stats()
            logger.info(
                "Incremental review stats: %d reviews, %d total issues found",
                stats["total_reviews"],
                stats["total_issues_found"],
            )

    async def _phase_post_checks(self) -> None:
        """
        Post-checks phase: 5-gate verification after implementation.

        Gates:
        1. Static Analysis - ruff + mypy
        2. Unit Tests - pytest
        3. Build - project-specific (if configured)
        4. Security Scan - bandit (recommended, not blocking)
        5. Manual Smoke - human verification (optional)
        """
        logger.info("=== POST_CHECKS PHASE (5-Gate Verification) ===")

        from ai_orchestrator.core.post_checks import PostChecks, PostCheckConfig

        # Configure post-checks based on project context
        config = PostCheckConfig()

        # Try to get project-specific commands
        if self.project_context:
            if hasattr(self.project_context, "verification"):
                v = self.project_context.verification
                if hasattr(v, "static_analysis") and v.static_analysis:
                    config.static_analysis_commands = v.static_analysis
                if hasattr(v, "unit_tests") and v.unit_tests:
                    config.unit_test_command = v.unit_tests
                if hasattr(v, "build") and v.build:
                    config.build_command = v.build

        # Run 5-gate verification
        checker = PostChecks(config=config, working_dir=self.project_path)
        result = await checker.run_all()

        # Store results in state
        self.state.post_check_results = {
            gate.gate.value: {
                "status": gate.status.value,
                "passed": gate.passed,
                "message": gate.message,
                "duration": gate.duration_seconds,
            }
            for gate in result.gates
        }

        # Log summary
        logger.info("\n%s", result.to_summary())

        if result.passed:
            logger.info("POST_CHECKS passed - all required gates succeeded")
            self.state.transition_to(WorkflowPhase.COMPLETED)
        else:
            # Failed required gates
            failed = [g.gate.value for g in result.failed_gates if g.required]
            logger.error(
                "POST_CHECKS failed - %d required gate(s) failed: %s",
                len(failed),
                ", ".join(failed),
            )
            self.state.add_error(f"Post-checks failed: {', '.join(failed)}")

            # Don't fail the workflow, but mark with warnings
            # Human can decide whether to proceed
            self.state.transition_to(WorkflowPhase.COMPLETED)

    async def _dry_run(self) -> WorkflowState:
        """Execute a dry run (no CLI invocations)."""
        logger.info("=== DRY RUN ===")

        print(f"\nProject: {self.project_path}")
        print(f"Discovery: {self.project_context.discovery_method if self.project_context else 'N/A'}")

        if self.project_context:
            print(f"\n{self.project_context.summary()}")

        print("\n" + "=" * 60)
        print("CONCURRENT CLI CONFIGURATION")
        print("=" * 60)
        print(f"Max concurrent CLIs: {self.max_concurrent}")
        print(f"Semaphore limit: {self._semaphore._value}")

        print("\nCLIs available:")
        for name, adapter in self.adapters.items():
            auth_status = "Ready" if await adapter.check_auth() else "Not authenticated"
            breaker = self.circuit_breakers.get(name)
            breaker_status = breaker.state if breaker else "N/A"
            print(f"  - {name}: {auth_status} (circuit: {breaker_status})")

        print("\nFallback chains:")
        for phase, chain in self.FALLBACK_CHAINS.items():
            available = [c for c in chain if c in self.adapters]
            print(f"  {phase}: {' → '.join(available) or 'none'}")

        print("\nPhase concurrency:")
        print(f"  Planning: {len(self._get_available_clis_for_phase('planning'))} CLIs in parallel")
        print(f"  Reviewing: {len(self._get_available_clis_for_phase('reviewing'))} CLIs in parallel")
        print(f"  Implementing: 1 CLI (with fallback)")

        print("\n" + "=" * 60)
        print("INCREMENTAL REVIEW CONFIGURATION")
        print("=" * 60)
        if self.incremental_review_enabled:
            print(f"Status: ENABLED")
            print(f"Review agent: {self.review_agent}")
            print(f"Granularity: {self.review_granularity}")
            print(f"Threshold: {self.review_threshold} lines")
            reviewer_available = self.review_agent in self.adapters
            print(f"Reviewer available: {'Yes' if reviewer_available else 'No'}")
        else:
            print("Status: DISABLED")
            print("Enable with: --incremental-review")

        print(f"\nSettings:")
        print(f"  Max iterations: {self.settings.iteration.max_iterations}")
        print(f"  Default CLI: {self.settings.default_cli}")

        return self.state or WorkflowState(
            prompt="[dry run]",
            project_path=str(self.project_path),
        )

    def _get_primary_cli(self) -> CLIAdapter:
        """Get the primary CLI adapter."""
        default_cli = self.settings.default_cli
        if default_cli in self.adapters:
            return self.adapters[default_cli]

        # Fallback to first available
        if self.adapters:
            return next(iter(self.adapters.values()))

        raise RuntimeError("No CLI adapters available")

    def _build_planning_prompt(self) -> str:
        """Build the planning prompt with context."""
        parts = [f"Task: {self.state.prompt}"]

        # Add project instructions if available
        if self.project_context and self.project_context.instructions_content:
            parts.append(f"\n## Project Instructions\n{self.project_context.instructions_content[:5000]}")

        parts.append("\n## Instructions\nCreate a detailed implementation plan for the above task.")

        return "\n".join(parts)

    def _build_review_prompt(self) -> str:
        """Build the review prompt."""
        parts = [f"Review the following implementation plan:\n\n{self.state.synthesized_plan.content if self.state.synthesized_plan else '[No plan]'}"]

        parts.append("\n## Instructions\nIdentify any issues, security concerns, or improvements needed.")

        return "\n".join(parts)

    def _build_implementation_prompt(self) -> str:
        """Build the implementation prompt."""
        parts = [f"Implement the following plan:\n\n{self.state.synthesized_plan.content if self.state.synthesized_plan else '[No plan]'}"]

        parts.append("\n## Instructions\nImplement the plan. Make the necessary code changes.")

        return "\n".join(parts)

    def _build_fix_prompt(self, issues: list[WorkflowFeedback]) -> str:
        """
        Build a fix prompt based on review feedback.

        Args:
            issues: List of issues to address (typically CRITICAL/HIGH severity).

        Returns:
            Prompt instructing the CLI to fix the identified issues.
        """
        parts = [
            "# Code Review Fixes Required",
            "",
            "The following issues were identified during code review and need to be addressed:",
            "",
        ]

        # Group issues by severity
        critical_issues = [i for i in issues if i.severity == WorkflowSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == WorkflowSeverity.HIGH]

        if critical_issues:
            parts.append("## CRITICAL Issues (Must Fix)")
            for i, issue in enumerate(critical_issues, 1):
                location = f" at {issue.file_path}:{issue.line_number}" if issue.file_path else ""
                parts.append(f"{i}. [{issue.category.value.upper()}]{location}")
                parts.append(f"   {issue.original_text}")
                if issue.fix_suggestion:
                    parts.append(f"   Suggested fix: {issue.fix_suggestion}")
                parts.append("")

        if high_issues:
            parts.append("## HIGH Priority Issues (Should Fix)")
            for i, issue in enumerate(high_issues, 1):
                location = f" at {issue.file_path}:{issue.line_number}" if issue.file_path else ""
                parts.append(f"{i}. [{issue.category.value.upper()}]{location}")
                parts.append(f"   {issue.original_text}")
                if issue.fix_suggestion:
                    parts.append(f"   Suggested fix: {issue.fix_suggestion}")
                parts.append("")

        parts.extend([
            "## Context",
            f"Original task: {self.state.prompt}",
            "",
            f"Current plan: {self.state.synthesized_plan.content[:500] if self.state.synthesized_plan else '[No plan]'}...",
            "",
            "## Instructions",
            "1. Address each issue listed above",
            "2. For CRITICAL issues, ensure they are completely resolved",
            "3. For HIGH issues, fix them or explain why they cannot be fixed now",
            "4. Maintain consistency with the original implementation plan",
            "5. Run any relevant tests to verify fixes",
            "",
            "Please make the necessary code changes to address these issues.",
        ])

        return "\n".join(parts)

    def _classify_review_output(
        self,
        raw_output: str,
        reviewer_name: str,
    ) -> list[WorkflowFeedback]:
        """
        Classify raw review output into structured feedback.

        Uses FeedbackClassifier to parse raw text and convert to workflow format.

        Args:
            raw_output: Raw text output from the reviewer CLI.
            reviewer_name: Name of the CLI that produced the output.

        Returns:
            List of WorkflowFeedback items for state tracking.
        """
        classifier = FeedbackClassifier()
        result = classifier.classify(raw_output, reviewer_name)

        # Convert classifier feedback to workflow feedback format
        workflow_feedback: list[WorkflowFeedback] = []

        # Mapping from classifier severity to workflow severity
        severity_map = {
            IssueSeverity.CRITICAL: WorkflowSeverity.CRITICAL,
            IssueSeverity.HIGH: WorkflowSeverity.HIGH,
            IssueSeverity.MEDIUM: WorkflowSeverity.MEDIUM,
            IssueSeverity.LOW: WorkflowSeverity.LOW,
        }

        # Mapping from classifier category to workflow category
        # Note: WorkflowCategory may not have all categories, so map to OTHER for missing ones
        category_map = {
            IssueCategory.SECURITY: WorkflowCategory.SECURITY,
            IssueCategory.PERFORMANCE: WorkflowCategory.PERFORMANCE,
            IssueCategory.CORRECTNESS: WorkflowCategory.OTHER,  # No CORRECTNESS in workflow
            IssueCategory.MAINTAINABILITY: WorkflowCategory.MAINTAINABILITY,
            IssueCategory.ARCHITECTURE: WorkflowCategory.ARCHITECTURE,
            IssueCategory.TESTING: WorkflowCategory.TESTING,
            IssueCategory.DOCUMENTATION: WorkflowCategory.DOCUMENTATION,
            IssueCategory.STYLE: WorkflowCategory.OTHER,
            IssueCategory.OTHER: WorkflowCategory.OTHER,
        }

        for item in result.feedback_items:
            wf = WorkflowFeedback(
                severity=severity_map.get(item.severity, WorkflowSeverity.MEDIUM),
                category=category_map.get(item.category, WorkflowCategory.OTHER),
                message=item.original_text,
                location=item.location,
                fix_suggestion=item.fix_suggestion,
                is_blocker=item.is_blocker,
                reviewer=reviewer_name,
            )
            workflow_feedback.append(wf)

        # If no items were extracted, create a single summary item
        if not workflow_feedback and raw_output.strip():
            # Check if it's a LGTM/clean response
            lgtm_phrases = ["lgtm", "looks good", "no issues", "all good", "approved"]
            is_clean = any(phrase in raw_output.lower() for phrase in lgtm_phrases)

            if is_clean:
                workflow_feedback.append(WorkflowFeedback(
                    severity=WorkflowSeverity.LOW,
                    category=WorkflowCategory.OTHER,
                    message="Review approved: No critical issues found.",
                    reviewer=reviewer_name,
                ))
            else:
                # Create general feedback item
                workflow_feedback.append(WorkflowFeedback(
                    severity=WorkflowSeverity.MEDIUM,
                    category=WorkflowCategory.OTHER,
                    message=truncate_with_marker(raw_output, 500),
                    reviewer=reviewer_name,
                ))

        logger.debug(
            "Classified %d feedback items from %s (raw length: %d)",
            len(workflow_feedback),
            reviewer_name,
            len(raw_output),
        )

        return workflow_feedback

    def _advance_phase(self) -> None:
        """Advance to the next logical phase."""
        phase_order = [
            WorkflowPhase.INIT,
            WorkflowPhase.PLANNING,
            WorkflowPhase.REVIEWING,
            WorkflowPhase.IMPLEMENTING,
            WorkflowPhase.POST_CHECKS,
            WorkflowPhase.COMPLETED,
        ]

        current_idx = phase_order.index(self.state.current_phase)
        if current_idx < len(phase_order) - 1:
            self.state.transition_to(phase_order[current_idx + 1])

    # =========================================================================
    # CONCURRENT CLI INVOCATION METHODS
    # =========================================================================

    def _get_available_clis_for_phase(self, phase: str) -> list[str]:
        """
        Get available CLIs for a phase, respecting circuit breakers.

        Args:
            phase: Phase name (planning, reviewing, implementing)

        Returns:
            List of available CLI names in priority order.
        """
        fallback_chain = self.FALLBACK_CHAINS.get(phase, list(self.adapters.keys()))

        available = []
        for cli_name in fallback_chain:
            # Check if adapter exists
            if cli_name not in self.adapters:
                continue

            # Check circuit breaker
            breaker = self.circuit_breakers.get(cli_name)
            if breaker and not breaker.is_available():
                logger.debug("CLI %s skipped (circuit open)", cli_name)
                continue

            available.append(cli_name)

        return available

    async def _invoke_cli_with_retry(
        self,
        cli_name: str,
        prompt: str,
        *,
        timeout_seconds: float | None = None,
        planning_mode: bool = False,
        max_retries: int = 2,
        retry_backoff: tuple[float, ...] = (1.0, 2.0, 4.0),
    ) -> CLIInvocationResult:
        """
        Invoke a single CLI with retry and circuit breaker integration.

        Args:
            cli_name: Name of CLI to invoke.
            prompt: Prompt to send.
            timeout_seconds: Optional timeout override.
            planning_mode: Whether to use planning mode.
            max_retries: Maximum retry attempts.
            retry_backoff: Backoff delays between retries.

        Returns:
            CLIInvocationResult with success/failure info.
        """
        adapter = self.adapters.get(cli_name)
        if not adapter:
            return CLIInvocationResult(
                cli_name=cli_name,
                result=None,
                success=False,
                error=f"CLI adapter not found: {cli_name}",
            )

        breaker = self.circuit_breakers.get(cli_name)
        if breaker and not breaker.is_available():
            return CLIInvocationResult(
                cli_name=cli_name,
                result=None,
                success=False,
                error=f"Circuit breaker open for {cli_name}",
            )

        start_time = datetime.now(UTC)
        retries = 0

        while retries <= max_retries:
            try:
                # Acquire semaphore for concurrency control
                async with self._semaphore:
                    result = await adapter.invoke(
                        prompt,
                        timeout_seconds=timeout_seconds,
                        planning_mode=planning_mode,
                        working_dir=str(self.project_path),
                    )

                duration = (datetime.now(UTC) - start_time).total_seconds()

                if result.success:
                    if breaker:
                        breaker.record_success()
                    return CLIInvocationResult(
                        cli_name=cli_name,
                        result=result,
                        success=True,
                        duration_seconds=duration,
                        retries_used=retries,
                    )
                else:
                    # Non-success result but no exception
                    # Map CLIStatus to ErrorType for weighted circuit breaker
                    error_type = self._cli_status_to_error_type(result.status)

                    if breaker:
                        breaker.record_failure(error_type)

                    # Check if retryable
                    if result.status == CLIStatus.TIMEOUT:
                        # Timeout - retry with shorter timeout
                        logger.warning(
                            "CLI %s timed out, retry %d/%d",
                            cli_name, retries + 1, max_retries
                        )
                    elif result.status == CLIStatus.RATE_LIMITED:
                        # Rate limited - retry with backoff
                        logger.warning(
                            "CLI %s rate limited, retry %d/%d",
                            cli_name, retries + 1, max_retries
                        )
                    elif result.status == CLIStatus.AUTH_ERROR:
                        # Auth error - don't retry (persistent failure)
                        return CLIInvocationResult(
                            cli_name=cli_name,
                            result=result,
                            success=False,
                            error=result.stderr or "Authentication failed",
                            duration_seconds=duration,
                            retries_used=retries,
                        )
                    else:
                        # Other error - don't retry
                        return CLIInvocationResult(
                            cli_name=cli_name,
                            result=result,
                            success=False,
                            error=result.stderr or "CLI invocation failed",
                            duration_seconds=duration,
                            retries_used=retries,
                        )

                    retries += 1
                    if retries <= max_retries:
                        # Add jitter to retry backoff to prevent synchronized retries
                        base_delay = retry_backoff[min(retries - 1, len(retry_backoff) - 1)]
                        jittered_delay = base_delay + random.uniform(0, base_delay * 0.5)
                        await asyncio.sleep(jittered_delay)

            except asyncio.TimeoutError:
                duration = (datetime.now(UTC) - start_time).total_seconds()
                if breaker:
                    breaker.record_failure(ErrorType.TIMEOUT)

                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        "CLI %s timeout, retry %d/%d",
                        cli_name, retries, max_retries
                    )
                    base_delay = retry_backoff[min(retries - 1, len(retry_backoff) - 1)]
                    jittered_delay = base_delay + random.uniform(0, base_delay * 0.5)
                    await asyncio.sleep(jittered_delay)
                else:
                    return CLIInvocationResult(
                        cli_name=cli_name,
                        result=None,
                        success=False,
                        error=f"Timeout after {max_retries} retries",
                        duration_seconds=duration,
                        retries_used=retries,
                    )

            except Exception as e:
                duration = (datetime.now(UTC) - start_time).total_seconds()
                if breaker:
                    breaker.record_failure(ErrorType.UNKNOWN)

                logger.error("CLI %s error: %s", cli_name, e, exc_info=True)
                return CLIInvocationResult(
                    cli_name=cli_name,
                    result=None,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                    retries_used=retries,
                )

        # Should not reach here, but safety return
        return CLIInvocationResult(
            cli_name=cli_name,
            result=None,
            success=False,
            error="Max retries exceeded",
            retries_used=retries,
        )

    async def _invoke_clis_concurrent(
        self,
        cli_names: list[str],
        prompt: str,
        *,
        timeout_seconds: float | None = None,
        planning_mode: bool = False,
        min_successes: int = 1,
        per_task_timeout: float | None = None,
    ) -> list[CLIInvocationResult]:
        """
        Invoke multiple CLIs concurrently with semaphore limiting and per-task timeouts.

        This is the core parallelization method. All CLIs in cli_names are
        invoked simultaneously (limited by semaphore), and results are
        collected and returned.

        Each task has an individual timeout to prevent a single hanging CLI
        from blocking the entire operation.

        Args:
            cli_names: List of CLI names to invoke.
            prompt: Prompt to send to all CLIs.
            timeout_seconds: Optional timeout override for CLI invocation.
            planning_mode: Whether to use planning mode.
            min_successes: Minimum successful results required.
            per_task_timeout: Individual timeout per task (default: 900s).

        Returns:
            List of CLIInvocationResult objects (successful and failed).

        Example:
            # Invoke claude, codex, and gemini in parallel for planning
            results = await self._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                planning_prompt,
                planning_mode=True,
                per_task_timeout=600.0,  # 10 min per CLI
            )
            # Results come back in parallel, limited by semaphore
        """
        if not cli_names:
            logger.warning("No CLIs specified for concurrent invocation")
            return []

        # Default per-task timeout (15 minutes if not specified)
        task_timeout = per_task_timeout or 900.0

        logger.info(
            "Invoking %d CLIs concurrently: %s (per-task timeout: %.0fs)",
            len(cli_names), ", ".join(cli_names), task_timeout
        )

        async def invoke_with_timeout(cli_name: str) -> CLIInvocationResult:
            """Wrap CLI invocation with individual timeout."""
            try:
                return await asyncio.wait_for(
                    self._invoke_cli_with_retry(
                        cli_name,
                        prompt,
                        timeout_seconds=timeout_seconds,
                        planning_mode=planning_mode,
                    ),
                    timeout=task_timeout,
                )
            except asyncio.TimeoutError:
                # Record timeout in circuit breaker
                breaker = self.circuit_breakers.get(cli_name)
                if breaker:
                    breaker.record_failure(ErrorType.TIMEOUT)

                logger.warning(
                    "CLI %s timed out after %.0fs (per-task limit)",
                    cli_name, task_timeout
                )
                return CLIInvocationResult(
                    cli_name=cli_name,
                    result=None,
                    success=False,
                    error=f"Task timeout after {task_timeout}s",
                    duration_seconds=task_timeout,
                )

        # Create tasks for all CLIs with individual timeouts
        tasks = [invoke_with_timeout(cli_name) for cli_name in cli_names]

        # Execute all tasks concurrently
        # Note: asyncio.gather runs all coroutines concurrently, but our
        # semaphore inside _invoke_cli_with_retry limits actual concurrent
        # subprocess invocations to max_concurrent_clis
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed: list[CLIInvocationResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Task raised an exception (not timeout - those are handled above)
                # Log with full traceback for debugging
                logger.warning(
                    "CLI %s raised exception during concurrent invocation: %s",
                    cli_names[i],
                    result,
                    exc_info=result,
                )
                processed.append(CLIInvocationResult(
                    cli_name=cli_names[i],
                    result=None,
                    success=False,
                    error=str(result),
                ))
            else:
                processed.append(result)

        # Log summary
        successes = sum(1 for r in processed if r.success)
        failures = len(processed) - successes
        logger.info(
            "Concurrent invocation complete: %d successes, %d failures",
            successes, failures
        )

        # Check minimum successes
        if successes < min_successes:
            logger.warning(
                "Fewer successes (%d) than required minimum (%d)",
                successes, min_successes
            )

        return processed

    def _cli_status_to_error_type(self, status: CLIStatus) -> ErrorType:
        """
        Map CLI status to error type for circuit breaker weighting.

        Different error types have different weights in the circuit breaker:
        - TIMEOUT: Full weight (1.0) - indicates possible overload
        - RATE_LIMITED: Half weight (0.5) - transient, often recovers quickly
        - AUTH_ERROR: Double weight (2.0) - persistent, needs manual intervention
        - CLIENT_ERROR: Zero weight (0.0) - our fault, not CLI's fault
        - SERVER_ERROR: Full weight (1.0) - CLI-side issues

        Args:
            status: The CLIStatus from the invocation result.

        Returns:
            ErrorType for circuit breaker weighting.
        """
        status_to_error: dict[CLIStatus, ErrorType] = {
            CLIStatus.TIMEOUT: ErrorType.TIMEOUT,
            CLIStatus.RATE_LIMITED: ErrorType.RATE_LIMITED,
            CLIStatus.AUTH_ERROR: ErrorType.AUTH_ERROR,
            CLIStatus.ERROR: ErrorType.SERVER_ERROR,
            CLIStatus.SUCCESS: ErrorType.UNKNOWN,  # Should not happen
        }
        return status_to_error.get(status, ErrorType.UNKNOWN)

    async def _invoke_with_fallback(
        self,
        phase: str,
        prompt: str,
        *,
        timeout_seconds: float | None = None,
        planning_mode: bool = False,
    ) -> CLIInvocationResult:
        """
        Invoke a CLI with automatic fallback on failure.

        Tries CLIs in fallback chain order until one succeeds.

        Args:
            phase: Phase name for fallback chain selection.
            prompt: Prompt to send.
            timeout_seconds: Optional timeout override.
            planning_mode: Whether to use planning mode.

        Returns:
            CLIInvocationResult from first successful CLI.
        """
        available = self._get_available_clis_for_phase(phase)

        if not available:
            return CLIInvocationResult(
                cli_name="none",
                result=None,
                success=False,
                error=f"No CLIs available for {phase}",
            )

        for cli_name in available:
            result = await self._invoke_cli_with_retry(
                cli_name,
                prompt,
                timeout_seconds=timeout_seconds,
                planning_mode=planning_mode,
            )

            if result.success:
                return result

            logger.warning(
                "CLI %s failed for %s, trying next fallback",
                cli_name, phase
            )

        # All fallbacks exhausted
        return CLIInvocationResult(
            cli_name="all",
            result=None,
            success=False,
            error=f"All {len(available)} CLIs failed for {phase}",
        )
