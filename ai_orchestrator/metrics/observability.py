"""Observability and metrics tracking for the orchestrator."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

from ai_orchestrator.core.workflow_phases import WorkflowPhase

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Type of metric being tracked."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class WorkflowMetrics:
    """Aggregated metrics for a workflow run."""

    workflow_id: str
    started_at: datetime
    completed_at: datetime | None = None
    total_duration_seconds: float = 0.0

    # Phase metrics
    phase_durations: dict[str, float] = field(default_factory=dict)
    current_phase: WorkflowPhase | None = None

    # CLI metrics
    cli_invocations: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cli_successes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cli_failures: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cli_durations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # Iteration metrics
    total_iterations: int = 0
    convergence_achieved: bool = False
    critical_issues_initial: int = 0
    critical_issues_final: int = 0

    # Token/cost estimation (for subscription efficiency tracking)
    estimated_tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Human-in-loop metrics
    human_decisions_requested: int = 0
    human_decisions_approved: int = 0
    human_decisions_rejected: int = 0
    human_decision_wait_time: float = 0.0

    # Error tracking
    errors: list[str] = field(default_factory=list)

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def cli_success_rate(self) -> float:
        """Calculate overall CLI success rate."""
        total_success = sum(self.cli_successes.values())
        total_invocations = sum(self.cli_invocations.values())
        return total_success / total_invocations if total_invocations > 0 else 0.0

    @property
    def issue_reduction_rate(self) -> float:
        """Calculate critical issue reduction rate."""
        if self.critical_issues_initial == 0:
            return 1.0
        return 1.0 - (self.critical_issues_final / self.critical_issues_initial)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "phase_durations": self.phase_durations,
            "cli_invocations": dict(self.cli_invocations),
            "cli_successes": dict(self.cli_successes),
            "cli_failures": dict(self.cli_failures),
            "total_iterations": self.total_iterations,
            "convergence_achieved": self.convergence_achieved,
            "critical_issues_initial": self.critical_issues_initial,
            "critical_issues_final": self.critical_issues_final,
            "cache_hit_ratio": self.cache_hit_ratio,
            "cli_success_rate": self.cli_success_rate,
            "issue_reduction_rate": self.issue_reduction_rate,
            "human_decisions": {
                "requested": self.human_decisions_requested,
                "approved": self.human_decisions_approved,
                "rejected": self.human_decisions_rejected,
                "wait_time": self.human_decision_wait_time,
            },
            "errors": self.errors,
        }

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "WORKFLOW METRICS SUMMARY",
            "=" * 60,
            f"Workflow ID: {self.workflow_id}",
            f"Duration: {self.total_duration_seconds:.1f}s",
            "",
            "Phase Durations:",
        ]

        for phase, duration in self.phase_durations.items():
            lines.append(f"  {phase}: {duration:.1f}s")

        lines.extend([
            "",
            "CLI Performance:",
            f"  Total Invocations: {sum(self.cli_invocations.values())}",
            f"  Success Rate: {self.cli_success_rate:.1%}",
        ])

        for cli, count in self.cli_invocations.items():
            success = self.cli_successes.get(cli, 0)
            failure = self.cli_failures.get(cli, 0)
            lines.append(f"  {cli}: {success}/{count} successful ({failure} failed)")

        lines.extend([
            "",
            "Iteration Metrics:",
            f"  Total Iterations: {self.total_iterations}",
            f"  Convergence: {'Yes' if self.convergence_achieved else 'No'}",
            f"  Issue Reduction: {self.issue_reduction_rate:.1%}",
            "",
            "Efficiency:",
            f"  Cache Hit Ratio: {self.cache_hit_ratio:.1%}",
            f"  Estimated Tokens: {self.estimated_tokens_used:,}",
        ])

        if self.human_decisions_requested > 0:
            lines.extend([
                "",
                "Human Decisions:",
                f"  Requested: {self.human_decisions_requested}",
                f"  Approved: {self.human_decisions_approved}",
                f"  Rejected: {self.human_decisions_rejected}",
                f"  Wait Time: {self.human_decision_wait_time:.1f}s",
            ])

        if self.errors:
            lines.extend([
                "",
                f"Errors: {len(self.errors)}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


class MetricsCollector:
    """
    Collect and track workflow metrics.

    Provides:
    - Phase timing
    - CLI invocation tracking
    - Iteration metrics
    - Token/efficiency estimation
    - Human decision tracking
    """

    def __init__(
        self,
        workflow_id: str,
        output_dir: Path | None = None,
    ) -> None:
        """
        Initialize metrics collector.

        Args:
            workflow_id: Unique identifier for this workflow run.
            output_dir: Directory to save metrics files.
        """
        self.metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            started_at=datetime.now(UTC),
        )
        self.output_dir = output_dir
        self._phase_start_times: dict[str, float] = {}
        self._points: list[MetricPoint] = []

    def start_phase(self, phase: WorkflowPhase) -> None:
        """Record start of a workflow phase."""
        self.metrics.current_phase = phase
        self._phase_start_times[phase.value] = time.monotonic()

        logger.info("Phase started: %s", phase.value)

    def end_phase(self, phase: WorkflowPhase) -> None:
        """Record end of a workflow phase."""
        start_time = self._phase_start_times.get(phase.value)
        if start_time:
            duration = time.monotonic() - start_time
            self.metrics.phase_durations[phase.value] = duration

            logger.info("Phase ended: %s (%.1fs)", phase.value, duration)

    def record_cli_invocation(
        self,
        cli_name: str,
        success: bool,
        duration_seconds: float,
        tokens_used: int = 0,
        cache_hit: bool = False,
    ) -> None:
        """Record a CLI invocation."""
        self.metrics.cli_invocations[cli_name] += 1

        if success:
            self.metrics.cli_successes[cli_name] += 1
        else:
            self.metrics.cli_failures[cli_name] += 1

        self.metrics.cli_durations[cli_name].append(duration_seconds)
        self.metrics.estimated_tokens_used += tokens_used

        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

        self._add_point(
            "cli_invocation",
            1.0,
            MetricType.COUNTER,
            {"cli": cli_name, "success": str(success)},
        )

    def record_iteration(
        self,
        iteration_number: int,
        critical_count: int,
    ) -> None:
        """Record an iteration completion."""
        self.metrics.total_iterations = iteration_number

        if iteration_number == 1:
            self.metrics.critical_issues_initial = critical_count

        self.metrics.critical_issues_final = critical_count

        self._add_point(
            "iteration_critical_count",
            float(critical_count),
            MetricType.GAUGE,
            {"iteration": str(iteration_number)},
        )

    def record_convergence(self, achieved: bool) -> None:
        """Record whether convergence was achieved."""
        self.metrics.convergence_achieved = achieved

    def record_human_decision(
        self,
        approved: bool,
        wait_time_seconds: float,
    ) -> None:
        """Record a human decision."""
        self.metrics.human_decisions_requested += 1

        if approved:
            self.metrics.human_decisions_approved += 1
        else:
            self.metrics.human_decisions_rejected += 1

        self.metrics.human_decision_wait_time += wait_time_seconds

    def record_error(self, error_message: str) -> None:
        """Record an error."""
        self.metrics.errors.append(error_message)

        self._add_point(
            "error",
            1.0,
            MetricType.COUNTER,
            {"message": error_message[:100]},
        )

    def complete(self) -> WorkflowMetrics:
        """Mark workflow as complete and return final metrics."""
        self.metrics.completed_at = datetime.now(UTC)
        self.metrics.total_duration_seconds = (
            self.metrics.completed_at - self.metrics.started_at
        ).total_seconds()

        # Save metrics if output directory configured
        if self.output_dir:
            self._save_metrics()

        logger.info(
            "Workflow completed: %s (%.1fs)",
            self.metrics.workflow_id,
            self.metrics.total_duration_seconds,
        )

        return self.metrics

    def _add_point(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Add a metric data point."""
        point = MetricPoint(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
        )
        self._points.append(point)

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary metrics
        metrics_file = self.output_dir / f"metrics_{self.metrics.workflow_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        # Save detailed points
        points_file = self.output_dir / f"points_{self.metrics.workflow_id}.jsonl"
        with open(points_file, "w") as f:
            for point in self._points:
                f.write(json.dumps({
                    "name": point.name,
                    "value": point.value,
                    "type": point.metric_type.value,
                    "labels": point.labels,
                    "timestamp": point.timestamp.isoformat(),
                }) + "\n")

        logger.info("Metrics saved to %s", metrics_file)

    def print_summary(self) -> None:
        """Print metrics summary to console."""
        print(self.metrics.to_summary())


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str, collector: MetricsCollector | None = None) -> None:
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed.
            collector: Optional metrics collector to record duration.
        """
        self.name = name
        self.collector = collector
        self._start_time: float = 0.0
        self._duration: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and record duration."""
        self._duration = time.monotonic() - self._start_time

        if self.collector:
            self.collector._add_point(
                f"timer_{self.name}",
                self._duration,
                MetricType.TIMER,
            )

    @property
    def duration(self) -> float:
        """Get the recorded duration."""
        return self._duration


def create_collector(
    workflow_id: str | None = None,
    output_dir: Path | str | None = None,
) -> MetricsCollector:
    """
    Create a metrics collector.

    Args:
        workflow_id: Optional workflow ID (auto-generated if not provided).
        output_dir: Optional directory to save metrics.

    Returns:
        Configured MetricsCollector.
    """
    import uuid

    wf_id = workflow_id or f"wf_{uuid.uuid4().hex[:8]}"
    out_dir = Path(output_dir) if output_dir else None

    return MetricsCollector(workflow_id=wf_id, output_dir=out_dir)
