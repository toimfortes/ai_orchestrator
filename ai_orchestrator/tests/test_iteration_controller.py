"""Tests for adaptive iteration controller."""

from __future__ import annotations

import pytest

from ai_orchestrator.core.iteration_controller import (
    AdaptiveIterationController,
    IterationConfig,
    create_controller,
)
from ai_orchestrator.core.workflow_phases import (
    ClassifiedFeedback,
    ConvergenceStatus,
    IssueCategory,
    Severity,
    WorkflowState,
)


class TestAdaptiveIterationController:
    """Tests for AdaptiveIterationController."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        controller = AdaptiveIterationController()

        assert controller.config.max_iterations == 4
        assert controller.config.consecutive_clean_rounds_required == 2

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = IterationConfig(
            max_iterations=6,
            consecutive_clean_rounds_required=3,
        )
        controller = AdaptiveIterationController(config)

        assert controller.config.max_iterations == 6

    @pytest.mark.asyncio
    async def test_continue_on_first_iteration(self) -> None:
        """Test that controller continues on first iteration."""
        controller = create_controller()
        state = WorkflowState(prompt="Test", project_path="/test")

        decision = await controller.should_continue_reviewing(state)

        assert decision.should_continue is True
        assert decision.status == ConvergenceStatus.IMPROVING
        assert decision.current_iteration == 0

    @pytest.mark.asyncio
    async def test_converge_on_clean_rounds(self) -> None:
        """Test convergence after consecutive clean rounds."""
        controller = create_controller(consecutive_clean_required=2)
        state = WorkflowState(prompt="Test", project_path="/test")

        # Add two clean rounds (no critical issues)
        clean_feedback = [
            ClassifiedFeedback(
                severity=Severity.LOW,
                category=IssueCategory.OTHER,
                message="Minor suggestion",
            ),
        ]

        state.add_review_round(clean_feedback, ["claude"])
        state.add_review_round(clean_feedback, ["claude"])

        decision = await controller.should_continue_reviewing(state)

        assert decision.should_continue is False
        assert decision.status == ConvergenceStatus.CONVERGED
        assert "consecutive" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_stop_at_max_iterations(self) -> None:
        """Test stopping at max iterations."""
        controller = create_controller(max_iterations=2)
        state = WorkflowState(prompt="Test", project_path="/test")

        # Add rounds with critical issues
        critical_feedback = [
            ClassifiedFeedback(
                severity=Severity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Security issue",
            ),
        ]

        state.add_review_round(critical_feedback, ["claude"])
        state.add_review_round(critical_feedback, ["claude"])

        decision = await controller.should_continue_reviewing(state)

        assert decision.should_continue is False
        assert decision.status == ConvergenceStatus.PLATEAU
        assert "max iterations" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_continue_with_critical_issues(self) -> None:
        """Test continuing when there are critical issues."""
        controller = create_controller()
        state = WorkflowState(prompt="Test", project_path="/test")

        # Add round with critical issue
        critical_feedback = [
            ClassifiedFeedback(
                severity=Severity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Security issue",
            ),
        ]

        state.add_review_round(critical_feedback, ["claude"])

        decision = await controller.should_continue_reviewing(state)

        assert decision.should_continue is True
        assert decision.status == ConvergenceStatus.IMPROVING
        assert decision.critical_count == 1

    @pytest.mark.asyncio
    async def test_detect_stale_feedback(self) -> None:
        """Test detection of stale (repeated) feedback."""
        config = IterationConfig(
            max_iterations=10,
            stop_on_plateau=True,
            plateau_detection_window=2,
        )
        controller = AdaptiveIterationController(config)
        state = WorkflowState(prompt="Test", project_path="/test")

        # Add identical rounds with CRITICAL issues (stale feedback)
        # Must use CRITICAL severity so rounds aren't considered "clean"
        # which would trigger CONVERGED before stale check
        same_feedback = [
            ClassifiedFeedback(
                severity=Severity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Same issue over and over",
            ),
        ]

        state.add_review_round(same_feedback, ["claude"])
        state.add_review_round(same_feedback, ["claude"])

        decision = await controller.should_continue_reviewing(state)

        assert decision.should_continue is False
        assert decision.status == ConvergenceStatus.PLATEAU
        assert "no progress" in decision.reason.lower()


class TestCreateController:
    """Tests for controller factory function."""

    def test_create_with_defaults(self) -> None:
        """Test creating controller with defaults."""
        controller = create_controller()

        assert controller.config.max_iterations == 4

    def test_create_with_custom_values(self) -> None:
        """Test creating controller with custom values."""
        controller = create_controller(
            max_iterations=8,
            consecutive_clean_required=3,
        )

        assert controller.config.max_iterations == 8
        assert controller.config.consecutive_clean_rounds_required == 3
