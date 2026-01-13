"""Tests for workflow phases and state models."""

from __future__ import annotations

import pytest

from ai_orchestrator.core.workflow_phases import (
    ClassifiedFeedback,
    ConvergenceStatus,
    IssueCategory,
    Plan,
    ReviewRound,
    Severity,
    WorkflowPhase,
    WorkflowState,
)


class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_phase_values(self) -> None:
        """Test that all expected phases exist."""
        assert WorkflowPhase.INIT.value == "init"
        assert WorkflowPhase.PLANNING.value == "planning"
        assert WorkflowPhase.REVIEWING.value == "reviewing"
        assert WorkflowPhase.IMPLEMENTING.value == "implementing"
        assert WorkflowPhase.COMPLETED.value == "completed"


class TestClassifiedFeedback:
    """Tests for ClassifiedFeedback model."""

    def test_create_feedback(self) -> None:
        """Test creating feedback."""
        feedback = ClassifiedFeedback(
            severity=Severity.CRITICAL,
            category=IssueCategory.SECURITY,
            message="SQL injection vulnerability",
            location="auth.py:42",
            is_blocker=True,
        )

        assert feedback.severity == Severity.CRITICAL
        assert feedback.category == IssueCategory.SECURITY
        assert feedback.is_blocker is True
        assert feedback.id  # Auto-generated


class TestReviewRound:
    """Tests for ReviewRound model."""

    def test_critical_count(self) -> None:
        """Test counting critical issues."""
        feedback = [
            ClassifiedFeedback(
                severity=Severity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Issue 1",
            ),
            ClassifiedFeedback(
                severity=Severity.HIGH,
                category=IssueCategory.PERFORMANCE,
                message="Issue 2",
            ),
            ClassifiedFeedback(
                severity=Severity.CRITICAL,
                category=IssueCategory.SECURITY,
                message="Issue 3",
            ),
        ]

        round_ = ReviewRound(round_number=1, feedback=feedback)

        assert round_.critical_count == 2
        assert round_.has_blockers is False

    def test_has_blockers(self) -> None:
        """Test blocker detection."""
        feedback = [
            ClassifiedFeedback(
                severity=Severity.HIGH,
                category=IssueCategory.SECURITY,
                message="Blocker issue",
                is_blocker=True,
            ),
        ]

        round_ = ReviewRound(round_number=1, feedback=feedback)

        assert round_.has_blockers is True


class TestWorkflowState:
    """Tests for WorkflowState model."""

    def test_create_state(self) -> None:
        """Test creating workflow state."""
        state = WorkflowState(
            prompt="Add user authentication",
            project_path="/path/to/project",
        )

        assert state.prompt == "Add user authentication"
        assert state.current_phase == WorkflowPhase.INIT
        assert state.current_iteration == 0
        assert state.workflow_id  # Auto-generated

    def test_transition_to(self) -> None:
        """Test phase transitions."""
        state = WorkflowState(
            prompt="Test",
            project_path="/test",
        )

        state.transition_to(WorkflowPhase.PLANNING)

        assert state.current_phase == WorkflowPhase.PLANNING
        assert len(state.phase_history) == 1
        assert state.phase_history[0][0] == WorkflowPhase.INIT

    def test_add_review_round(self) -> None:
        """Test adding review rounds."""
        state = WorkflowState(
            prompt="Test",
            project_path="/test",
        )

        feedback = [
            ClassifiedFeedback(
                severity=Severity.MEDIUM,
                category=IssueCategory.OTHER,
                message="Test feedback",
            ),
        ]

        state.add_review_round(feedback, ["claude"])

        assert state.current_iteration == 1
        assert len(state.review_rounds) == 1
        assert state.latest_review_round is not None
        assert state.latest_review_round.round_number == 1

    def test_checkpoint_roundtrip(self) -> None:
        """Test serialization and deserialization."""
        state = WorkflowState(
            prompt="Test prompt",
            project_path="/test/path",
        )
        state.transition_to(WorkflowPhase.PLANNING)

        # Serialize
        checkpoint = state.to_checkpoint_dict()

        # Deserialize
        restored = WorkflowState.from_checkpoint_dict(checkpoint)

        assert restored.prompt == state.prompt
        assert restored.current_phase == state.current_phase
        assert restored.workflow_id == state.workflow_id

    def test_add_error(self) -> None:
        """Test error tracking."""
        state = WorkflowState(
            prompt="Test",
            project_path="/test",
        )

        state.add_error("Something went wrong")

        assert len(state.errors) == 1
        assert "Something went wrong" in state.errors[0]


class TestPlan:
    """Tests for Plan model."""

    def test_create_plan(self) -> None:
        """Test creating a plan."""
        plan = Plan(
            content="## Implementation Plan\n\n1. Step one",
            source_cli="claude",
            scores={"completeness": 0.9, "security": 0.85},
        )

        assert plan.source_cli == "claude"
        assert plan.scores["completeness"] == 0.9
        assert plan.id  # Auto-generated
