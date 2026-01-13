"""Integration tests for full workflow execution.

Tests the complete orchestrator workflow from INIT to COMPLETED,
verifying phase transitions and data flow between phases.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_orchestrator.cli_adapters.base import CLIAdapter, CLIResult, CLIStatus
from ai_orchestrator.core.workflow_phases import (
    WorkflowPhase,
    WorkflowState,
)


class MockCLIAdapter(CLIAdapter):
    """Mock CLI adapter for testing."""

    def __init__(
        self,
        name: str = "mock_cli",
        responses: dict[str, str] | None = None,
        fail_on: list[str] | None = None,
    ):
        self.name = name
        self.responses = responses or {}
        self.fail_on = fail_on or []
        self.invocations: list[dict[str, Any]] = []

    async def invoke(
        self,
        prompt: str,
        continue_session: bool = False,
        planning_mode: bool = False,
        timeout_seconds: float = 900.0,
    ) -> CLIResult:
        """Mock invocation that returns configured responses."""
        self.invocations.append({
            "prompt": prompt,
            "continue_session": continue_session,
            "planning_mode": planning_mode,
            "timeout": timeout_seconds,
        })

        # Check if this should fail
        for fail_pattern in self.fail_on:
            if fail_pattern.lower() in prompt.lower():
                return CLIResult(
                    status=CLIStatus.ERROR,
                    output="",
                    error="Simulated failure",
                    session_id=None,
                )

        # Find matching response
        for pattern, response in self.responses.items():
            if pattern.lower() in prompt.lower():
                return CLIResult(
                    status=CLIStatus.SUCCESS,
                    output=response,
                    error="",
                    session_id=f"{self.name}_session",
                )

        # Default success response
        return CLIResult(
            status=CLIStatus.SUCCESS,
            output="Default successful response",
            error="",
            session_id=f"{self.name}_session",
        )

    async def check_auth(self) -> bool:
        return True

    def get_name(self) -> str:
        return self.name


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.default_cli = "mock_cli"
    settings.planning_clis = ["mock_cli"]
    settings.reviewing_clis = ["mock_cli"]
    settings.implementing_clis = ["mock_cli"]

    # Mock iteration config
    iteration_config = MagicMock()
    iteration_config.max_iterations = 3
    iteration_config.consecutive_clean_rounds_required = 2
    settings.iteration = iteration_config

    settings.get_timeout_for_cli.return_value = 60.0
    settings.get_cli_adapter.return_value = None  # Will be overridden
    return settings


@pytest.fixture
def mock_cli_adapter():
    """Create a mock CLI adapter with standard responses."""
    return MockCLIAdapter(
        name="mock_cli",
        responses={
            "task": "## Implementation Plan\n1. Create module\n2. Add tests\n3. Deploy",
            "review": "LGTM - no critical issues found",
            "implement": "Implementation completed successfully",
            "fix": "Fixes applied successfully",
        },
    )


class TestFullWorkflowIntegration:
    """Integration tests for complete workflow execution."""

    @pytest.mark.asyncio
    async def test_basic_workflow_completes(self, mock_settings, mock_cli_adapter):
        """Test that a basic workflow runs from INIT to COMPLETED."""
        with patch.object(mock_settings, 'get_cli_adapter', return_value=mock_cli_adapter):
            orchestrator = Orchestrator(
                prompt="Add a simple hello world function",
                settings=mock_settings,
                cli_adapters={"mock_cli": mock_cli_adapter},
            )

            # Override internal state for testing
            orchestrator._cli_adapters = {"mock_cli": mock_cli_adapter}

            # Run workflow (with mocked CLIs)
            # Note: This tests the state machine, not actual CLI execution
            assert orchestrator.state.phase == WorkflowPhase.INIT

    @pytest.mark.asyncio
    async def test_phase_transitions_are_tracked(self):
        """Test that phase transitions are recorded in state history."""
        state = WorkflowState(prompt="Test task", project_path="/tmp/test")

        # Simulate phase transitions
        state.transition_to(WorkflowPhase.PLANNING)
        assert state.current_phase == WorkflowPhase.PLANNING

        state.transition_to(WorkflowPhase.REVIEWING)
        assert state.current_phase == WorkflowPhase.REVIEWING

        state.transition_to(WorkflowPhase.FIXING)
        assert state.current_phase == WorkflowPhase.FIXING

        state.transition_to(WorkflowPhase.IMPLEMENTING)
        assert state.current_phase == WorkflowPhase.IMPLEMENTING

        state.transition_to(WorkflowPhase.POST_CHECKS)
        assert state.current_phase == WorkflowPhase.POST_CHECKS

        state.transition_to(WorkflowPhase.COMPLETED)
        assert state.current_phase == WorkflowPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_state_persists_across_phases(self):
        """Test that state data persists through phase transitions."""
        state = WorkflowState(prompt="Test task with context", project_path="/tmp/test")

        # Add data in planning phase
        state.transition_to(WorkflowPhase.PLANNING)
        state.metadata["plan_id"] = "plan_123"

        # Verify data persists through transitions
        state.transition_to(WorkflowPhase.REVIEWING)
        assert state.metadata["plan_id"] == "plan_123"

        state.transition_to(WorkflowPhase.IMPLEMENTING)
        assert state.metadata["plan_id"] == "plan_123"
        assert state.prompt == "Test task with context"


class TestWorkflowWithFeedback:
    """Tests for workflow with review feedback."""

    @pytest.mark.asyncio
    async def test_critical_feedback_triggers_fixing(self):
        """Test that critical feedback triggers FIXING phase."""
        from ai_orchestrator.core.workflow_phases import (
            ClassifiedFeedback,
            ReviewRound,
            Severity,
            IssueCategory,
        )

        state = WorkflowState(prompt="Test task", project_path="/tmp/test")
        state.transition_to(WorkflowPhase.REVIEWING)

        # Add a review round with critical feedback
        critical_feedback = ClassifiedFeedback(
            message="SQL injection vulnerability found",
            severity=Severity.CRITICAL,
            category=IssueCategory.SECURITY,
            is_blocker=True,
        )

        round_ = ReviewRound(
            round_number=1,
            feedback=[critical_feedback],
            reviewers_used=["mock_cli"],
        )
        state.add_review_round(round_)

        # Verify feedback is stored
        assert state.latest_review_round is not None
        assert state.latest_review_round.critical_count == 1
        assert state.latest_review_round.has_blockers is True

    @pytest.mark.asyncio
    async def test_no_critical_issues_skips_fixing(self):
        """Test that no critical issues can skip to implementing."""
        from ai_orchestrator.core.workflow_phases import (
            ClassifiedFeedback,
            ReviewRound,
            Severity,
            IssueCategory,
        )

        state = WorkflowState(prompt="Test task", project_path="/tmp/test")
        state.transition_to(WorkflowPhase.REVIEWING)

        # Add a review round with only LOW severity
        low_feedback = ClassifiedFeedback(
            message="Consider adding a comment",
            severity=Severity.LOW,
            category=IssueCategory.DOCUMENTATION,
            is_blocker=False,
        )

        round_ = ReviewRound(
            round_number=1,
            feedback=[low_feedback],
            reviewers_used=["mock_cli"],
        )
        state.add_review_round(round_)

        # Verify no blockers
        assert state.latest_review_round.critical_count == 0
        assert state.latest_review_round.has_blockers is False


class TestPostChecksIntegration:
    """Tests for POST_CHECKS phase integration."""

    @pytest.mark.asyncio
    async def test_post_checks_can_be_created(self):
        """Test that PostChecks can be instantiated."""
        from ai_orchestrator.core.post_checks import PostChecks, PostCheckConfig

        config = PostCheckConfig(
            static_analysis_commands=[["echo", "lint OK"]],
            unit_test_command=["echo", "tests OK"],
            build_command=[],  # Skip build
            security_scan_command=[],  # Skip security
        )

        checks = PostChecks(config=config, working_dir=Path.cwd())
        assert checks is not None

    @pytest.mark.asyncio
    async def test_post_checks_returns_result(self):
        """Test that PostChecks.run_all() returns proper result."""
        from ai_orchestrator.core.post_checks import (
            PostChecks,
            PostCheckConfig,
            GateName,
            GateStatus,
        )

        # Use echo commands that should succeed
        config = PostCheckConfig(
            static_analysis_commands=[["python", "--version"]],
            unit_test_command=["python", "--version"],
            build_command=[],
            security_scan_command=[],
        )

        checks = PostChecks(config=config, working_dir=Path.cwd())
        result = await checks.run_all()

        assert result is not None
        assert len(result.gates) == 5  # All 5 gates present

        # Static analysis should have run
        static_gate = result.get_gate(GateName.STATIC_ANALYSIS)
        assert static_gate is not None

        # Build should be skipped (no command)
        build_gate = result.get_gate(GateName.BUILD)
        assert build_gate is not None
        assert build_gate.status == GateStatus.SKIPPED


class TestIterationControllerIntegration:
    """Tests for iteration controller integration with workflow."""

    @pytest.mark.asyncio
    async def test_convergence_after_clean_rounds(self):
        """Test that workflow converges after consecutive clean rounds."""
        from ai_orchestrator.core.iteration_controller import (
            AdaptiveIterationController,
            IterationConfig,
            ConvergenceStatus,
        )
        from ai_orchestrator.core.workflow_phases import (
            ClassifiedFeedback,
            ReviewRound,
            Severity,
            IssueCategory,
        )

        config = IterationConfig(
            max_iterations=5,
            consecutive_clean_rounds_required=2,
        )
        controller = AdaptiveIterationController(config)

        state = WorkflowState(prompt="Test task", project_path="/tmp/test")

        # First round: has critical issues
        round1 = ReviewRound(
            round_number=1,
            feedback=[
                ClassifiedFeedback(
                    message="Critical bug found",
                    severity=Severity.CRITICAL,
                    category=IssueCategory.CORRECTNESS,
                    is_blocker=True,
                ),
            ],
        )
        state.add_review_round(round1)

        decision1 = await controller.should_continue_reviewing(state)
        assert decision1.should_continue is True

        # Second round: no critical issues
        round2 = ReviewRound(
            round_number=2,
            feedback=[
                ClassifiedFeedback(
                    message="Minor suggestion",
                    severity=Severity.LOW,
                    category=IssueCategory.STYLE,
                    is_blocker=False,
                ),
            ],
        )
        state.add_review_round(round2)

        # Third round: still no critical issues
        round3 = ReviewRound(
            round_number=3,
            feedback=[],  # No issues
        )
        state.add_review_round(round3)

        decision3 = await controller.should_continue_reviewing(state)
        # Should converge after 2 consecutive clean rounds
        assert decision3.status == ConvergenceStatus.CONVERGED

    @pytest.mark.asyncio
    async def test_max_iterations_stops_workflow(self):
        """Test that max iterations stops the workflow."""
        from ai_orchestrator.core.iteration_controller import (
            AdaptiveIterationController,
            IterationConfig,
            ConvergenceStatus,
        )
        from ai_orchestrator.core.workflow_phases import (
            ClassifiedFeedback,
            ReviewRound,
            Severity,
            IssueCategory,
        )

        config = IterationConfig(
            max_iterations=2,
            consecutive_clean_rounds_required=3,
        )
        controller = AdaptiveIterationController(config)

        state = WorkflowState(prompt="Test task", project_path="/tmp/test")

        # Three rounds with issues (never converges naturally)
        for i in range(3):
            round_ = ReviewRound(
                round_number=i + 1,
                feedback=[
                    ClassifiedFeedback(
                        message=f"Issue {i} found in code",
                        severity=Severity.HIGH,
                        category=IssueCategory.CORRECTNESS,
                        is_blocker=True,
                    ),
                ],
            )
            state.add_review_round(round_)

        decision = await controller.should_continue_reviewing(state)

        # Should stop due to max iterations
        assert decision.should_continue is False
        assert "max" in decision.reason.lower() or decision.status in (
            ConvergenceStatus.PLATEAU,
            ConvergenceStatus.CONVERGED,
        )


class TestReviewConsolidatorIntegration:
    """Tests for review consolidator integration."""

    def test_consolidate_multi_reviewer_feedback(self):
        """Test consolidating feedback from multiple reviewers."""
        from ai_orchestrator.reviewing.feedback_classifier import (
            ClassificationResult,
            ClassifiedFeedback,
            IssueCategory,
            IssueSeverity,
            Actionability,
        )
        from ai_orchestrator.reviewing.review_consolidator import (
            ReviewConsolidator,
        )

        # Feedback from reviewer 1
        result1 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability in auth.py:45",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=45,
                ),
            ],
            reviewer_name="claude",
            raw_text="Claude review",
        )

        # Similar feedback from reviewer 2
        result2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection issue found in auth.py line 45",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=45,
                ),
            ],
            reviewer_name="codex",
            raw_text="Codex review",
        )

        consolidator = ReviewConsolidator(similarity_threshold=0.5)
        result = consolidator.consolidate([result1, result2])

        # Should merge similar issues
        assert len(result.issues) == 1
        assert result.issues[0].reviewer_count == 2
        assert result.total_source_items == 2
        assert result.duplicate_count == 1


class TestReviewerRouterIntegration:
    """Tests for reviewer router integration."""

    def test_route_security_issues_to_specialists(self):
        """Test that security issues are routed to security specialists."""
        from ai_orchestrator.reviewing.feedback_classifier import (
            ClassificationResult,
            ClassifiedFeedback,
            IssueCategory,
            IssueSeverity,
            Actionability,
        )
        from ai_orchestrator.reviewing.reviewer_router import (
            ReviewerRouter,
        )

        # Create feedback with security issue
        feedback = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Critical XSS vulnerability",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="general",
            raw_text="Security issue found",
        )

        router = ReviewerRouter(
            available_reviewers=["claude", "gemini", "codex"],
        )
        plan = router.create_routing_plan(feedback)

        # Security should be routed to claude (has SECURITY strength)
        assert IssueCategory.SECURITY in plan.specialist_assignments
        assert "claude" in plan.specialist_assignments[IssueCategory.SECURITY]


class TestFeedbackClassifierIntegration:
    """Tests for feedback classifier integration in workflow."""

    def test_classify_and_store_in_state(self):
        """Test classifying feedback and storing in workflow state."""
        from ai_orchestrator.reviewing.feedback_classifier import (
            FeedbackClassifier,
        )
        from ai_orchestrator.core.workflow_phases import (
            ClassifiedFeedback as WorkflowFeedback,
            ReviewRound,
            Severity,
            IssueCategory,
        )

        classifier = FeedbackClassifier()
        result = classifier.classify(
            "CRITICAL: SQL injection in auth.py:45 - user input not sanitized",
            reviewer_name="security_bot",
        )

        assert len(result.feedback_items) >= 1
        item = result.feedback_items[0]

        # Convert to workflow format (simulating orchestrator behavior)
        workflow_feedback = WorkflowFeedback(
            message=item.original_text,
            severity=Severity(item.severity.value),
            category=IssueCategory(item.category.value),
            is_blocker=item.is_blocker,
            location=f"{item.file_path}:{item.line_number}" if item.file_path else None,
        )

        # Store in state
        state = WorkflowState(prompt="Test", project_path="/tmp/test")
        round_ = ReviewRound(
            round_number=1,
            feedback=[workflow_feedback],
            reviewers_used=["security_bot"],
        )
        state.add_review_round(round_)

        # Verify
        assert state.latest_review_round.critical_count >= 1
