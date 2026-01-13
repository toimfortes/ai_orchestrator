"""Tests for orchestrator concurrent CLI invocation."""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_orchestrator.cli_adapters.base import CLIResult, CLIStatus
from ai_orchestrator.core.orchestrator import (
    CircuitBreaker,
    CircuitState,
    CLIInvocationResult,
    ErrorType,
    ERROR_WEIGHTS,
    Orchestrator,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in closed state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available() is True

    def test_record_success_resets_failure_count(self):
        """Recording success resets failure count."""
        breaker = CircuitBreaker(name="test")
        breaker.failure_count = 2
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    def test_record_failure_increments_count(self):
        """Recording failure increments count."""
        breaker = CircuitBreaker(name="test", fail_threshold=3)
        breaker.record_failure()
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(name="test", fail_threshold=3)

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_available() is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to half-open after reset timeout."""
        breaker = CircuitBreaker(
            name="test",
            fail_threshold=1,
            base_reset_timeout=0.01,  # Very short for testing
            jitter_factor=0.0,  # No jitter for deterministic test
        )

        # Trip the breaker
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_available() is False

        # Wait for reset timeout
        import time
        time.sleep(0.02)

        # Should transition to half-open
        assert breaker.is_available() is True
        assert breaker.state == CircuitState.HALF_OPEN


class TestCircuitBreakerEnhanced:
    """Tests for enhanced circuit breaker features (jitter, error taxonomy)."""

    def test_error_taxonomy_weights_different_errors(self):
        """Test that different error types have different weights."""
        # AUTH_ERROR should have higher weight (2.0)
        assert ERROR_WEIGHTS[ErrorType.AUTH_ERROR] == 2.0

        # RATE_LIMITED should have lower weight (0.5)
        assert ERROR_WEIGHTS[ErrorType.RATE_LIMITED] == 0.5

        # CLIENT_ERROR should not count (0.0)
        assert ERROR_WEIGHTS[ErrorType.CLIENT_ERROR] == 0.0

    def test_weighted_failure_counting(self):
        """Test that failures are weighted by error type."""
        breaker = CircuitBreaker(name="test", fail_threshold=3.0)

        # Rate limited errors count as 0.5 each
        breaker.record_failure(ErrorType.RATE_LIMITED)
        assert breaker.weighted_failure_count == 0.5

        breaker.record_failure(ErrorType.RATE_LIMITED)
        assert breaker.weighted_failure_count == 1.0

        # Circuit should still be closed (1.0 < 3.0)
        assert breaker.state == CircuitState.CLOSED

    def test_auth_error_opens_faster(self):
        """Test that AUTH_ERROR (weight=2.0) opens circuit faster."""
        breaker = CircuitBreaker(name="test", fail_threshold=3.0)

        # Two auth errors = 4.0 weighted, exceeds threshold of 3.0
        breaker.record_failure(ErrorType.AUTH_ERROR)
        assert breaker.weighted_failure_count == 2.0
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure(ErrorType.AUTH_ERROR)
        assert breaker.weighted_failure_count == 4.0
        assert breaker.state == CircuitState.OPEN

    def test_client_error_does_not_count(self):
        """Test that CLIENT_ERROR (weight=0.0) doesn't count toward threshold."""
        breaker = CircuitBreaker(name="test", fail_threshold=3.0)

        # Client errors have 0 weight, should never open
        for _ in range(10):
            breaker.record_failure(ErrorType.CLIENT_ERROR)

        assert breaker.weighted_failure_count == 0.0
        assert breaker.failure_count == 10  # Raw count still increments
        assert breaker.state == CircuitState.CLOSED

    def test_jitter_is_applied_to_reset_timeout(self):
        """Test that jitter is applied to reset timeout."""
        breaker = CircuitBreaker(
            name="test",
            base_reset_timeout=60.0,
            jitter_factor=0.5,  # 0-50% jitter
        )

        timeouts = []
        for _ in range(10):
            timeouts.append(breaker._get_reset_timeout_with_jitter())

        # All timeouts should be >= base (60.0)
        assert all(t >= 60.0 for t in timeouts)

        # With 50% jitter, max should be <= 90.0 (60 + 60*0.5)
        assert all(t <= 90.0 for t in timeouts)

        # There should be variation (not all the same)
        assert len(set(timeouts)) > 1

    def test_exponential_backoff_on_consecutive_opens(self):
        """Test that reset timeout increases with consecutive opens."""
        breaker = CircuitBreaker(
            name="test",
            fail_threshold=1.0,
            base_reset_timeout=10.0,
            max_reset_timeout=100.0,
            jitter_factor=0.0,  # No jitter for deterministic test
        )

        # First open
        breaker.record_failure()
        assert breaker.consecutive_opens == 1
        timeout1 = breaker._get_reset_timeout_with_jitter()

        # Reset and second open
        breaker.record_success()
        breaker.state = CircuitState.HALF_OPEN  # Simulate half-open
        breaker.consecutive_opens = 1  # Still counting
        breaker.record_failure()
        breaker.consecutive_opens = 2  # Manually set for test
        timeout2 = breaker._get_reset_timeout_with_jitter()

        # Second timeout should be 2x the first (exponential backoff)
        assert timeout2 > timeout1

    def test_max_reset_timeout_cap(self):
        """Test that reset timeout is capped at max value."""
        breaker = CircuitBreaker(
            name="test",
            base_reset_timeout=60.0,
            max_reset_timeout=120.0,  # 2 minute max
            jitter_factor=0.5,
        )

        # Simulate many consecutive opens
        breaker.consecutive_opens = 10  # Would be 60 * 2^10 without cap

        timeout = breaker._get_reset_timeout_with_jitter()

        # Should be capped near max (120 + up to 50% jitter = max 180)
        assert timeout <= 180.0

    def test_success_after_half_open_resets_consecutive(self):
        """Test that success in half-open state resets consecutive opens."""
        breaker = CircuitBreaker(name="test", fail_threshold=1.0)

        # Trip the breaker multiple times
        breaker.consecutive_opens = 5
        breaker.state = CircuitState.HALF_OPEN

        # Success in half-open resets consecutive opens
        breaker.record_success()

        assert breaker.consecutive_opens == 0
        assert breaker.state == CircuitState.CLOSED

    def test_last_error_type_tracked(self):
        """Test that last error type is tracked."""
        breaker = CircuitBreaker(name="test", fail_threshold=10.0)

        breaker.record_failure(ErrorType.TIMEOUT)
        assert breaker.last_error_type == ErrorType.TIMEOUT

        breaker.record_failure(ErrorType.RATE_LIMITED)
        assert breaker.last_error_type == ErrorType.RATE_LIMITED


class TestCircuitBreakerPersistence:
    """Tests for circuit breaker state persistence."""

    def test_save_and_load_state(self, tmp_path):
        """Test that circuit breaker state can be saved and restored."""
        state_file = tmp_path / "breaker.json"

        # Create breaker with some state
        breaker1 = CircuitBreaker(
            name="test",
            fail_threshold=2.0,  # Low threshold to trigger OPEN
            state_file=state_file,
        )
        # Record failures to trigger OPEN state (which saves consecutive_opens)
        breaker1.record_failure(ErrorType.TIMEOUT)
        breaker1.record_failure(ErrorType.TIMEOUT)  # This opens the breaker

        assert breaker1.state == CircuitState.OPEN
        assert breaker1.consecutive_opens == 1

        # Create new breaker that loads from same file
        breaker2 = CircuitBreaker(
            name="test",
            fail_threshold=2.0,
            state_file=state_file,
        )

        # State should be restored
        assert breaker2.weighted_failure_count == breaker1.weighted_failure_count
        assert breaker2.failure_count == breaker1.failure_count
        assert breaker2.consecutive_opens == breaker1.consecutive_opens

    def test_stale_state_ignored(self, tmp_path):
        """Test that stale persisted state is ignored."""
        import json

        state_file = tmp_path / "breaker.json"

        # Write stale state (very old timestamp)
        stale_state = {
            "state": "open",
            "weighted_failure_count": 10.0,
            "failure_count": 5,
            "consecutive_opens": 3,
            "saved_at": "2020-01-01T00:00:00+00:00",  # Very old
        }
        with open(state_file, "w") as f:
            json.dump(stale_state, f)

        # Create new breaker - should ignore stale state
        breaker = CircuitBreaker(
            name="test",
            fail_threshold=3.0,
            max_reset_timeout=60.0,  # State is older than this
            state_file=state_file,
        )

        # Should start fresh
        assert breaker.state == CircuitState.CLOSED
        assert breaker.weighted_failure_count == 0.0


class TestCLIInvocationResult:
    """Tests for CLIInvocationResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        cli_result = CLIResult(
            cli_name="claude",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="Success",
            stderr="",
        )
        result = CLIInvocationResult(
            cli_name="claude",
            result=cli_result,
            success=True,
            duration_seconds=1.5,
        )
        assert result.success is True
        assert result.cli_name == "claude"
        assert result.error is None

    def test_failed_result(self):
        """Test creating a failed result."""
        result = CLIInvocationResult(
            cli_name="claude",
            result=None,
            success=False,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestOrchestratorConcurrentMethods:
    """Tests for Orchestrator concurrent CLI invocation methods."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.enabled_clis = ["claude", "codex", "gemini"]
        settings.default_cli = "claude"
        settings.iteration.max_iterations = 4
        settings.iteration.consecutive_clean_required = 2
        settings.get_timeout_for_cli.return_value = 900
        return settings

    @pytest.fixture
    def mock_adapters(self):
        """Create mock CLI adapters."""
        adapters = {}
        for name in ["claude", "codex", "gemini"]:
            adapter = MagicMock()
            adapter.name = name
            adapter.is_available = True
            adapter.check_auth = AsyncMock(return_value=True)
            adapter.invoke = AsyncMock(return_value=CLIResult(
                cli_name=name,
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout=f"Plan from {name}",
                stderr="",
            ))
            adapters[name] = adapter
        return adapters

    @pytest.mark.asyncio
    async def test_get_available_clis_for_phase(self, mock_settings, mock_adapters):
        """Test getting available CLIs respects circuit breakers."""
        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters

            # All CLIs should be available initially
            available = orchestrator._get_available_clis_for_phase("planning")
            assert "claude" in available
            assert "codex" in available
            assert "gemini" in available

            # Trip circuit breaker for codex (set state and recent failure time)
            from datetime import datetime, UTC
            orchestrator.circuit_breakers["codex"].state = CircuitState.OPEN
            orchestrator.circuit_breakers["codex"].last_failure_time = datetime.now(UTC)

            available = orchestrator._get_available_clis_for_phase("planning")
            assert "claude" in available
            assert "codex" not in available  # Should be excluded (circuit open)
            assert "gemini" in available

    @pytest.mark.asyncio
    async def test_invoke_cli_with_retry_success(self, mock_settings, mock_adapters):
        """Test successful CLI invocation."""
        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            result = await orchestrator._invoke_cli_with_retry(
                "claude",
                "Test prompt",
            )

            assert result.success is True
            assert result.cli_name == "claude"
            assert result.result is not None

    @pytest.mark.asyncio
    async def test_invoke_cli_with_retry_failure(self, mock_settings, mock_adapters):
        """Test CLI invocation failure."""
        # Make claude fail
        mock_adapters["claude"].invoke = AsyncMock(return_value=CLIResult(
            cli_name="claude",
            status=CLIStatus.ERROR,
            exit_code=1,
            stdout="",
            stderr="Error occurred",
        ))

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            result = await orchestrator._invoke_cli_with_retry(
                "claude",
                "Test prompt",
                max_retries=0,  # No retries
            )

            assert result.success is False
            assert "Error occurred" in result.error

    @pytest.mark.asyncio
    async def test_invoke_clis_concurrent(self, mock_settings, mock_adapters):
        """Test concurrent CLI invocation."""
        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
                max_concurrent_clis=3,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            results = await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
            )

            assert len(results) == 3
            assert all(r.success for r in results)

            # Verify all adapters were called
            for adapter in mock_adapters.values():
                adapter.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_clis_concurrent_partial_failure(self, mock_settings, mock_adapters):
        """Test concurrent invocation with partial failure."""
        # Make codex fail
        mock_adapters["codex"].invoke = AsyncMock(return_value=CLIResult(
            cli_name="codex",
            status=CLIStatus.ERROR,
            exit_code=1,
            stdout="",
            stderr="Codex failed",
        ))

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            results = await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
                min_successes=1,
            )

            assert len(results) == 3
            successes = [r for r in results if r.success]
            failures = [r for r in results if not r.success]

            assert len(successes) == 2  # claude and gemini
            assert len(failures) == 1   # codex

    @pytest.mark.asyncio
    async def test_invoke_with_fallback(self, mock_settings, mock_adapters):
        """Test fallback invocation."""
        # Make claude fail, codex should be tried next
        mock_adapters["claude"].invoke = AsyncMock(return_value=CLIResult(
            cli_name="claude",
            status=CLIStatus.ERROR,
            exit_code=1,
            stdout="",
            stderr="Claude failed",
        ))

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            result = await orchestrator._invoke_with_fallback(
                "planning",
                "Test prompt",
            )

            # Should succeed with codex (first fallback)
            assert result.success is True
            assert result.cli_name == "codex"

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent(self, mock_settings, mock_adapters):
        """Test that semaphore limits concurrent invocations."""
        # Track concurrent calls
        max_concurrent_observed = 0
        current_concurrent = 0

        async def slow_invoke(*args, **kwargs):
            nonlocal max_concurrent_observed, current_concurrent
            current_concurrent += 1
            max_concurrent_observed = max(max_concurrent_observed, current_concurrent)
            await asyncio.sleep(0.05)  # Simulate work
            current_concurrent -= 1
            return CLIResult(
                cli_name="test",
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout="Done",
                stderr="",
            )

        for adapter in mock_adapters.values():
            adapter.invoke = slow_invoke

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            # Limit to 2 concurrent
            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
                max_concurrent_clis=2,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
            )

            # Should never exceed semaphore limit
            assert max_concurrent_observed <= 2


class TestOrchestratorPhases:
    """Tests for orchestrator phase handlers with concurrent execution."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.enabled_clis = ["claude", "codex"]
        settings.default_cli = "claude"
        settings.iteration.max_iterations = 4
        settings.iteration.consecutive_clean_required = 2
        settings.get_timeout_for_cli.return_value = 900
        return settings

    @pytest.mark.asyncio
    async def test_planning_phase_multi_planner(self, mock_settings):
        """Test planning phase uses multiple planners."""
        mock_adapters = {}
        for name in ["claude", "codex"]:
            adapter = MagicMock()
            adapter.name = name
            adapter.is_available = True
            adapter.check_auth = AsyncMock(return_value=True)
            adapter.invoke = AsyncMock(return_value=CLIResult(
                cli_name=name,
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout=f"Plan from {name}",
                stderr="",
            ))
            mock_adapters[name] = adapter

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters
            with patch("ai_orchestrator.project.loader.load_project_context") as mock_load:
                mock_context = MagicMock()
                mock_context.instructions_content = None
                mock_context.summary.return_value = "Test project"
                mock_context.discovery_method = "auto"
                mock_load.return_value = mock_context

                orchestrator = Orchestrator(
                    Path("/tmp/test"),
                    settings=mock_settings,
                )
                orchestrator.adapters = mock_adapters

                # Initialize state
                from ai_orchestrator.core.workflow_phases import WorkflowState
                orchestrator.state = WorkflowState(
                    prompt="Test task",
                    project_path="/tmp/test",
                )
                orchestrator.project_context = mock_context

                # Run planning phase
                await orchestrator._phase_planning()

                # Should have generated plans from both CLIs
                assert len(orchestrator.state.plans) == 2
                assert orchestrator.state.synthesized_plan is not None


class TestFeedbackClassifierIntegration:
    """Tests for feedback classifier integration in orchestrator."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.enabled_clis = ["claude", "gemini"]
        settings.default_cli = "claude"
        settings.iteration.max_iterations = 4
        settings.iteration.consecutive_clean_required = 2
        settings.get_timeout_for_cli.return_value = 900
        return settings

    @pytest.fixture
    def orchestrator(self, mock_settings, tmp_path):
        """Create orchestrator with mocked dependencies."""
        mock_adapters = {}
        for name in ["claude", "gemini"]:
            adapter = MagicMock()
            adapter.name = name
            adapter.is_available = True
            adapter.check_auth = AsyncMock(return_value=True)
            mock_adapters[name] = adapter

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters
            orch = Orchestrator(
                tmp_path,
                settings=mock_settings,
            )
            orch.adapters = mock_adapters
            return orch

    def test_classify_critical_security_feedback(self, orchestrator):
        """Test classification of critical security feedback."""
        raw_output = """
        CRITICAL: SQL injection vulnerability in db.py:42
        The user input is not sanitized before being used in query.
        """

        feedback = orchestrator._classify_review_output(raw_output, "claude")

        assert len(feedback) >= 1
        # At least one item should be critical
        critical_items = [f for f in feedback if f.severity.value == "critical"]
        assert len(critical_items) >= 1
        assert any(f.is_blocker for f in feedback)

    def test_classify_lgtm_response(self, orchestrator):
        """Test classification of LGTM response."""
        # Note: Avoid "critical" word as classifier uses keyword matching
        raw_output = "LGTM - the code looks good to me. Approved for merge."

        feedback = orchestrator._classify_review_output(raw_output, "gemini")

        assert len(feedback) >= 1
        # LGTM response should result in low severity approval message
        # The _classify_review_output handles LGTM detection specially
        assert not any(f.is_blocker for f in feedback)

    def test_classify_multiple_issues(self, orchestrator):
        """Test classification of multiple issues."""
        raw_output = """
        1. CRITICAL: XSS vulnerability in user input handling at frontend/form.js:15
        2. HIGH: Memory leak in cache.py - objects not being released
        3. Medium: Consider adding more unit tests
        4. Suggestion: Rename variable 'x' for clarity
        """

        feedback = orchestrator._classify_review_output(raw_output, "claude")

        assert len(feedback) >= 3

        # Check severity distribution
        severities = [f.severity.value for f in feedback]
        assert "critical" in severities
        assert "high" in severities

        # Should have blockers
        assert any(f.is_blocker for f in feedback)

    def test_classify_empty_output(self, orchestrator):
        """Test classification of empty output."""
        feedback = orchestrator._classify_review_output("", "claude")

        # Empty output should return empty list
        assert len(feedback) == 0

    def test_classify_performance_feedback(self, orchestrator):
        """Test classification of performance-related feedback."""
        raw_output = """
        Performance issue: The query in get_users() causes N+1 problem.
        Suggest using batch loading or prefetch to improve efficiency.
        """

        feedback = orchestrator._classify_review_output(raw_output, "gemini")

        assert len(feedback) >= 1
        # Should detect performance category
        categories = [f.category.value for f in feedback]
        assert "performance" in categories

    def test_reviewer_name_preserved(self, orchestrator):
        """Test that reviewer name is preserved in feedback."""
        raw_output = "Bug: Error handling missing in auth module"

        feedback = orchestrator._classify_review_output(raw_output, "test_reviewer")

        assert len(feedback) >= 1
        assert all(f.reviewer == "test_reviewer" for f in feedback)

    @pytest.mark.asyncio
    async def test_reviewing_phase_classifies_feedback(self, orchestrator, mock_settings):
        """Test that reviewing phase uses classifier."""
        from ai_orchestrator.core.workflow_phases import (
            Plan,
            WorkflowPhase,
            WorkflowState,
        )

        # Setup state with a plan to review
        orchestrator.state = WorkflowState(
            prompt="Test task",
            project_path=str(orchestrator.project_path),
        )
        orchestrator.state.synthesized_plan = Plan(
            source_cli="claude",
            content="Test plan content",
        )
        orchestrator.state.current_phase = WorkflowPhase.REVIEWING

        # Mock adapter responses with feedback
        orchestrator.adapters["claude"].invoke = AsyncMock(return_value=CLIResult(
            cli_name="claude",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="CRITICAL: Security issue in auth.py:10",
            stderr="",
        ))
        orchestrator.adapters["gemini"].invoke = AsyncMock(return_value=CLIResult(
            cli_name="gemini",
            status=CLIStatus.SUCCESS,
            exit_code=0,
            stdout="Bug: Missing null check in utils.py",
            stderr="",
        ))

        # Run reviewing phase
        await orchestrator._phase_reviewing()

        # Should have classified feedback
        assert len(orchestrator.state.review_rounds) >= 1
        if orchestrator.state.review_rounds:
            round_feedback = orchestrator.state.review_rounds[-1].feedback
            assert len(round_feedback) >= 1
            # Check that feedback has proper severity classification
            assert any(f.severity.value in ("critical", "high") for f in round_feedback)


class TestPerTaskTimeouts:
    """Tests for per-task timeout in concurrent CLI invocation."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.enabled_clis = ["claude", "codex", "gemini"]
        settings.default_cli = "claude"
        settings.iteration.max_iterations = 4
        settings.iteration.consecutive_clean_required = 2
        settings.get_timeout_for_cli.return_value = 900
        return settings

    @pytest.fixture
    def mock_adapters(self):
        """Create mock CLI adapters."""
        adapters = {}
        for name in ["claude", "codex", "gemini"]:
            adapter = MagicMock()
            adapter.name = name
            adapter.is_available = True
            adapter.check_auth = AsyncMock(return_value=True)
            adapter.invoke = AsyncMock(return_value=CLIResult(
                cli_name=name,
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout=f"Response from {name}",
                stderr="",
            ))
            adapters[name] = adapter
        return adapters

    @pytest.mark.asyncio
    async def test_per_task_timeout_returns_failure(self, mock_settings, mock_adapters):
        """Test that per-task timeout returns failure without blocking others."""
        # Make codex hang (never return)
        async def hanging_invoke(*args, **kwargs):
            await asyncio.sleep(10)  # Hang for 10 seconds
            return CLIResult(
                cli_name="codex",
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout="Should never see this",
                stderr="",
            )

        mock_adapters["codex"].invoke = hanging_invoke

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            # Use very short per-task timeout
            results = await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
                per_task_timeout=0.1,  # 100ms timeout
            )

            # Should complete without waiting for hanging task
            assert len(results) == 3

            # claude and gemini should succeed
            claude_result = next(r for r in results if r.cli_name == "claude")
            gemini_result = next(r for r in results if r.cli_name == "gemini")
            assert claude_result.success is True
            assert gemini_result.success is True

            # codex should fail with timeout
            codex_result = next(r for r in results if r.cli_name == "codex")
            assert codex_result.success is False
            assert "timeout" in codex_result.error.lower()

    @pytest.mark.asyncio
    async def test_per_task_timeout_records_in_circuit_breaker(self, mock_settings, mock_adapters):
        """Test that per-task timeout is recorded in circuit breaker."""
        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(10)
            return CLIResult(
                cli_name="codex",
                status=CLIStatus.SUCCESS,
                exit_code=0,
                stdout="",
                stderr="",
            )

        mock_adapters["codex"].invoke = slow_invoke

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            # Initial state
            breaker = orchestrator.circuit_breakers["codex"]
            initial_failures = breaker.failure_count

            # Run with short timeout
            await orchestrator._invoke_clis_concurrent(
                ["codex"],
                "Test prompt",
                per_task_timeout=0.1,
            )

            # Circuit breaker should have recorded the timeout
            assert breaker.failure_count > initial_failures
            assert breaker.last_error_type == ErrorType.TIMEOUT

    @pytest.mark.asyncio
    async def test_concurrent_completes_within_timeout(self, mock_settings, mock_adapters):
        """Test that concurrent invocation completes within max per-task timeout."""
        import time

        # Make all adapters fast
        for adapter in mock_adapters.values():
            async def fast_invoke(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms
                return CLIResult(
                    cli_name="test",
                    status=CLIStatus.SUCCESS,
                    exit_code=0,
                    stdout="Fast response",
                    stderr="",
                )
            adapter.invoke = fast_invoke

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            start = time.time()
            results = await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
                per_task_timeout=60.0,  # 60 second timeout
            )
            elapsed = time.time() - start

            # Should complete quickly, not wait for full timeout
            assert elapsed < 5.0  # Should be < 5 seconds
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_exception_in_task_handled_gracefully(self, mock_settings, mock_adapters):
        """Test that exceptions in individual tasks don't crash gather."""
        async def error_invoke(*args, **kwargs):
            raise RuntimeError("Unexpected error in CLI")

        mock_adapters["codex"].invoke = error_invoke

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = mock_adapters

            orchestrator = Orchestrator(
                Path("/tmp/test"),
                settings=mock_settings,
            )
            orchestrator.adapters = mock_adapters
            orchestrator.project_path = Path("/tmp/test")

            # Should not raise - return_exceptions=True handles it
            results = await orchestrator._invoke_clis_concurrent(
                ["claude", "codex", "gemini"],
                "Test prompt",
            )

            # All results should be present
            assert len(results) == 3

            # codex should have failed
            codex_result = next(r for r in results if r.cli_name == "codex")
            assert codex_result.success is False
            # The error might be wrapped, but should contain the message
            assert "error" in codex_result.error.lower() or "unexpected" in codex_result.error.lower()


class TestErrorTypeMapping:
    """Tests for CLI status to error type mapping."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create minimal orchestrator for testing."""
        mock_settings = MagicMock()
        mock_settings.enabled_clis = []
        mock_settings.default_cli = "claude"
        mock_settings.iteration.max_iterations = 4
        mock_settings.iteration.consecutive_clean_required = 2

        with patch("ai_orchestrator.core.orchestrator.get_available_adapters") as mock_get:
            mock_get.return_value = {}
            return Orchestrator(tmp_path, settings=mock_settings)

    def test_timeout_maps_correctly(self, orchestrator):
        """Test TIMEOUT status maps to TIMEOUT error type."""
        error_type = orchestrator._cli_status_to_error_type(CLIStatus.TIMEOUT)
        assert error_type == ErrorType.TIMEOUT

    def test_rate_limited_maps_correctly(self, orchestrator):
        """Test RATE_LIMITED status maps to RATE_LIMITED error type."""
        error_type = orchestrator._cli_status_to_error_type(CLIStatus.RATE_LIMITED)
        assert error_type == ErrorType.RATE_LIMITED

    def test_auth_error_maps_correctly(self, orchestrator):
        """Test AUTH_ERROR status maps to AUTH_ERROR error type."""
        error_type = orchestrator._cli_status_to_error_type(CLIStatus.AUTH_ERROR)
        assert error_type == ErrorType.AUTH_ERROR

    def test_error_maps_to_server_error(self, orchestrator):
        """Test generic ERROR status maps to SERVER_ERROR type."""
        error_type = orchestrator._cli_status_to_error_type(CLIStatus.ERROR)
        assert error_type == ErrorType.SERVER_ERROR

    def test_unknown_status_maps_to_unknown(self, orchestrator):
        """Test unknown status maps to UNKNOWN error type."""
        # Create a mock status that's not in the mapping
        mock_status = MagicMock()
        error_type = orchestrator._cli_status_to_error_type(mock_status)
        assert error_type == ErrorType.UNKNOWN
