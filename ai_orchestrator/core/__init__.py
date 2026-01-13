"""Core orchestration module."""

from ai_orchestrator.core.auth_checker import AuthChecker, AuthResult, AuthStatus
from ai_orchestrator.core.graceful_degradation import (
    CLIFailure,
    DegradedAction,
    DegradedResult,
    DegradedStatus,
    FailureReport,
    GracefulDegradation,
)
from ai_orchestrator.core.iteration_controller import AdaptiveIterationController
from ai_orchestrator.core.orchestrator import Orchestrator
from ai_orchestrator.core.post_checks import (
    GateName,
    GateResult,
    GateStatus,
    PostCheckConfig,
    PostCheckResult,
    PostChecks,
)
from ai_orchestrator.core.retry_utils import (
    RetryableError,
    TimeoutRetryError,
    RateLimitRetryError,
    AuthError,
    create_retry_decorator,
    retry_with_backoff,
    retry_rate_limited,
    retry_timeout,
)
from ai_orchestrator.core.state_manager import StateManager
from ai_orchestrator.core.workflow_phases import WorkflowPhase, WorkflowState

__all__ = [
    # Auth
    "AuthChecker",
    "AuthResult",
    "AuthStatus",
    # Graceful degradation
    "CLIFailure",
    "DegradedAction",
    "DegradedResult",
    "DegradedStatus",
    "FailureReport",
    "GracefulDegradation",
    # Iteration control
    "AdaptiveIterationController",
    # Orchestration
    "Orchestrator",
    # Post-checks
    "GateName",
    "GateResult",
    "GateStatus",
    "PostCheckConfig",
    "PostCheckResult",
    "PostChecks",
    # State
    "StateManager",
    # Retry utilities
    "RetryableError",
    "TimeoutRetryError",
    "RateLimitRetryError",
    "AuthError",
    "create_retry_decorator",
    "retry_with_backoff",
    "retry_rate_limited",
    "retry_timeout",
    # Workflow
    "WorkflowPhase",
    "WorkflowState",
]
