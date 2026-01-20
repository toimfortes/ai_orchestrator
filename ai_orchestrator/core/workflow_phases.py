"""Workflow phase definitions and state models."""

from __future__ import annotations

from datetime import datetime, UTC
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class WorkflowPhase(str, Enum):
    """Phases of the orchestration workflow."""

    INIT = "init"
    SCAFFOLDING = "scaffolding"  # Foundation scaffolding for greenfield projects
    DEEP_RESEARCH = "deep_research"
    MEASURE_TWICE = "measure_twice"
    PLANNING = "planning"
    PLAN_COMPARISON = "plan_comparison"
    PLAN_SYNTHESIS = "plan_synthesis"
    REVIEWING = "reviewing"
    CONSOLIDATING = "consolidating"
    FIXING = "fixing"
    IMPLEMENTING = "implementing"
    POST_CHECKS = "post_checks"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCategory(str, Enum):
    """Issue category for routing to specialists."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ConvergenceStatus(str, Enum):
    """Status of review convergence."""

    IMPROVING = "improving"  # Still finding new issues
    CONVERGED = "converged"  # Zero critical issues for N rounds
    PLATEAU = "plateau"  # No progress, but not converged


class ClassifiedFeedback(BaseModel):
    """Structured feedback from a reviewer."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    severity: Severity
    category: IssueCategory
    message: str
    location: str | None = None  # e.g., "file.py:123"
    fix_suggestion: str | None = None
    is_blocker: bool = False
    reviewer: str = ""  # Which CLI/model provided this feedback


class ReviewRound(BaseModel):
    """A single round of reviews."""

    round_number: int
    feedback: list[ClassifiedFeedback] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewers_used: list[str] = Field(default_factory=list)

    @property
    def critical_count(self) -> int:
        """Count of CRITICAL severity issues."""
        return sum(1 for f in self.feedback if f.severity == Severity.CRITICAL)

    @property
    def has_blockers(self) -> bool:
        """Whether any issues are marked as blockers."""
        return any(f.is_blocker for f in self.feedback)


class Plan(BaseModel):
    """A generated implementation plan."""

    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    content: str
    source_cli: str  # Which CLI generated this plan
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scores: dict[str, float] = Field(default_factory=dict)  # criterion -> score
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Complete state of an orchestration workflow."""

    # Identity
    workflow_id: str = Field(default_factory=lambda: str(uuid4()))
    checkpoint_id: str | None = None

    # Task
    prompt: str
    project_path: str

    # Phase tracking
    current_phase: WorkflowPhase = WorkflowPhase.INIT
    phase_history: list[tuple[WorkflowPhase, datetime]] = Field(default_factory=list)

    # Research
    research_context: str | None = None

    # Planning
    plans: list[Plan] = Field(default_factory=list)
    synthesized_plan: Plan | None = None

    # Reviewing
    current_iteration: int = 0
    review_rounds: list[ReviewRound] = Field(default_factory=list)
    convergence_status: ConvergenceStatus = ConvergenceStatus.IMPROVING

    # Implementation
    implementation_result: str | None = None
    files_modified: list[str] = Field(default_factory=list)

    # Post-checks
    post_check_results: dict[str, bool] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Error tracking
    errors: list[str] = Field(default_factory=list)

    def transition_to(self, phase: WorkflowPhase) -> None:
        """Transition to a new phase, recording history."""
        self.phase_history.append((self.current_phase, datetime.now(UTC)))
        self.current_phase = phase
        self.updated_at = datetime.now(UTC)

    def add_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"[{datetime.now(UTC).isoformat()}] {error}")
        self.updated_at = datetime.now(UTC)

    def add_review_round(self, feedback: list[ClassifiedFeedback], reviewers: list[str]) -> None:
        """Add a new review round."""
        self.current_iteration += 1
        round_ = ReviewRound(
            round_number=self.current_iteration,
            feedback=feedback,
            reviewers_used=reviewers,
        )
        self.review_rounds.append(round_)
        self.updated_at = datetime.now(UTC)

    @property
    def latest_review_round(self) -> ReviewRound | None:
        """Get the most recent review round."""
        return self.review_rounds[-1] if self.review_rounds else None

    @property
    def total_critical_issues(self) -> int:
        """Total critical issues across all rounds."""
        return sum(r.critical_count for r in self.review_rounds)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Convert to dictionary for checkpointing."""
        return self.model_dump(mode="json")

    @classmethod
    def from_checkpoint_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Restore from checkpoint dictionary."""
        return cls.model_validate(data)
