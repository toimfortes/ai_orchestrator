"""Tests for metrics and observability."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
import tempfile

from ai_orchestrator.core.workflow_phases import WorkflowPhase
from ai_orchestrator.metrics.observability import (
    MetricsCollector,
    Timer,
    WorkflowMetrics,
    create_collector,
)


class TestWorkflowMetrics:
    """Tests for WorkflowMetrics dataclass."""

    def test_cache_hit_ratio(self):
        """Test cache hit ratio calculation."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
            cache_hits=80,
            cache_misses=20,
        )

        assert metrics.cache_hit_ratio == pytest.approx(0.8)

    def test_cache_hit_ratio_zero_total(self):
        """Test cache hit ratio with no cache activity."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
            cache_hits=0,
            cache_misses=0,
        )

        assert metrics.cache_hit_ratio == 0.0

    def test_cli_success_rate(self):
        """Test CLI success rate calculation."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
        )
        metrics.cli_invocations["claude"] = 10
        metrics.cli_successes["claude"] = 8
        metrics.cli_invocations["codex"] = 5
        metrics.cli_successes["codex"] = 5

        assert metrics.cli_success_rate == pytest.approx(13 / 15)

    def test_issue_reduction_rate(self):
        """Test issue reduction rate calculation."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
            critical_issues_initial=10,
            critical_issues_final=2,
        )

        assert metrics.issue_reduction_rate == pytest.approx(0.8)

    def test_issue_reduction_rate_zero_initial(self):
        """Test issue reduction with no initial issues."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
            critical_issues_initial=0,
            critical_issues_final=0,
        )

        assert metrics.issue_reduction_rate == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = WorkflowMetrics(
            workflow_id="test123",
            started_at=datetime.now(UTC),
            total_iterations=3,
            convergence_achieved=True,
        )

        d = metrics.to_dict()

        assert d["workflow_id"] == "test123"
        assert d["total_iterations"] == 3
        assert d["convergence_achieved"] is True
        assert "started_at" in d

    def test_to_summary(self):
        """Test summary generation."""
        metrics = WorkflowMetrics(
            workflow_id="test",
            started_at=datetime.now(UTC),
            total_duration_seconds=120.5,
            total_iterations=2,
        )
        metrics.cli_invocations["claude"] = 5
        metrics.cli_successes["claude"] = 4

        summary = metrics.to_summary()

        assert "WORKFLOW METRICS" in summary
        assert "120.5s" in summary
        assert "claude" in summary


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_phase_timing(self):
        """Test phase start/end timing."""
        collector = MetricsCollector(workflow_id="test")

        collector.start_phase(WorkflowPhase.PLANNING)
        # Simulate some work (50ms for reliable timing on Windows)
        import time
        time.sleep(0.05)
        collector.end_phase(WorkflowPhase.PLANNING)

        assert WorkflowPhase.PLANNING.value in collector.metrics.phase_durations
        assert collector.metrics.phase_durations[WorkflowPhase.PLANNING.value] >= 0.01

    def test_cli_invocation_recording(self):
        """Test recording CLI invocations."""
        collector = MetricsCollector(workflow_id="test")

        collector.record_cli_invocation(
            cli_name="claude",
            success=True,
            duration_seconds=5.0,
            tokens_used=1000,
            cache_hit=True,
        )
        collector.record_cli_invocation(
            cli_name="claude",
            success=False,
            duration_seconds=2.0,
            tokens_used=500,
            cache_hit=False,
        )

        assert collector.metrics.cli_invocations["claude"] == 2
        assert collector.metrics.cli_successes["claude"] == 1
        assert collector.metrics.cli_failures["claude"] == 1
        assert collector.metrics.cache_hits == 1
        assert collector.metrics.cache_misses == 1
        assert collector.metrics.estimated_tokens_used == 1500

    def test_iteration_recording(self):
        """Test recording iterations."""
        collector = MetricsCollector(workflow_id="test")

        collector.record_iteration(1, critical_count=5)
        collector.record_iteration(2, critical_count=2)
        collector.record_iteration(3, critical_count=0)

        assert collector.metrics.total_iterations == 3
        assert collector.metrics.critical_issues_initial == 5
        assert collector.metrics.critical_issues_final == 0

    def test_human_decision_recording(self):
        """Test recording human decisions."""
        collector = MetricsCollector(workflow_id="test")

        collector.record_human_decision(approved=True, wait_time_seconds=10.0)
        collector.record_human_decision(approved=False, wait_time_seconds=5.0)

        assert collector.metrics.human_decisions_requested == 2
        assert collector.metrics.human_decisions_approved == 1
        assert collector.metrics.human_decisions_rejected == 1
        assert collector.metrics.human_decision_wait_time == pytest.approx(15.0)

    def test_error_recording(self):
        """Test recording errors."""
        collector = MetricsCollector(workflow_id="test")

        collector.record_error("Something went wrong")
        collector.record_error("Another error")

        assert len(collector.metrics.errors) == 2
        assert "Something went wrong" in collector.metrics.errors

    def test_complete(self):
        """Test completing metrics collection."""
        collector = MetricsCollector(workflow_id="test")

        metrics = collector.complete()

        assert metrics.completed_at is not None
        assert metrics.total_duration_seconds >= 0

    def test_save_metrics(self):
        """Test saving metrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            collector = MetricsCollector(
                workflow_id="test123",
                output_dir=output_dir,
            )

            collector.record_cli_invocation("claude", True, 1.0)
            collector.complete()

            metrics_file = output_dir / "metrics_test123.json"
            assert metrics_file.exists()


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_duration(self):
        """Test timer measures duration."""
        import time

        with Timer("test_op") as timer:
            time.sleep(0.05)  # Longer sleep for Windows timer resolution

        assert timer.duration >= 0.03  # Looser bound for timing variability

    def test_timer_with_collector(self):
        """Test timer records to collector."""
        collector = MetricsCollector(workflow_id="test")

        with Timer("my_operation", collector=collector):
            pass

        # Should have recorded a metric point
        assert len(collector._points) > 0


class TestCreateCollector:
    """Tests for create_collector factory."""

    def test_create_with_defaults(self):
        """Test creating collector with defaults."""
        collector = create_collector()

        assert collector.metrics.workflow_id.startswith("wf_")
        assert collector.output_dir is None

    def test_create_with_custom_id(self):
        """Test creating collector with custom workflow ID."""
        collector = create_collector(workflow_id="my_workflow")

        assert collector.metrics.workflow_id == "my_workflow"

    def test_create_with_output_dir(self):
        """Test creating collector with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = create_collector(output_dir=tmpdir)

            assert collector.output_dir == Path(tmpdir)
