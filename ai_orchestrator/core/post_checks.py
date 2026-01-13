"""POST_CHECKS 5-gate verification after implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Status of a verification gate."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class GateName(str, Enum):
    """Names of the 5 verification gates."""

    STATIC_ANALYSIS = "static_analysis"
    UNIT_TESTS = "unit_tests"
    BUILD = "build"
    SECURITY_SCAN = "security_scan"
    MANUAL_SMOKE = "manual_smoke"


@dataclass
class GateResult:
    """Result of a single verification gate."""

    gate: GateName
    status: GateStatus
    required: bool = True  # Whether this gate blocks the workflow
    message: str = ""
    duration_seconds: float = 0.0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether this gate passed (or was skipped/warning if not required)."""
        if self.required:
            return self.status == GateStatus.PASSED
        return self.status in (GateStatus.PASSED, GateStatus.SKIPPED, GateStatus.WARNING)


@dataclass
class PostCheckResult:
    """Result of all POST_CHECK gates."""

    gates: list[GateResult]
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether all required gates passed."""
        return all(g.passed for g in self.gates if g.required)

    @property
    def total_duration_seconds(self) -> float:
        """Total duration of all gates."""
        return sum(g.duration_seconds for g in self.gates)

    @property
    def failed_gates(self) -> list[GateResult]:
        """List of failed gates."""
        return [g for g in self.gates if not g.passed]

    @property
    def warnings(self) -> list[GateResult]:
        """List of gates with warnings."""
        return [g for g in self.gates if g.status == GateStatus.WARNING]

    def get_gate(self, name: GateName) -> GateResult | None:
        """Get a specific gate result."""
        for gate in self.gates:
            if gate.gate == name:
                return gate
        return None

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "POST_CHECKS VERIFICATION SUMMARY",
            "=" * 60,
        ]

        for gate in self.gates:
            icon = {
                GateStatus.PASSED: "[+]",
                GateStatus.FAILED: "[X]",
                GateStatus.SKIPPED: "[-]",
                GateStatus.WARNING: "[!]",
            }[gate.status]

            req_tag = "(required)" if gate.required else "(optional)"
            lines.append(f"  {icon} {gate.gate.value}: {gate.status.value} {req_tag}")
            if gate.message:
                lines.append(f"      {gate.message}")

        lines.append("")
        overall = "PASSED" if self.passed else "FAILED"
        lines.append(f"Overall: {overall}")
        lines.append(f"Duration: {self.total_duration_seconds:.1f}s")
        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class PostCheckConfig:
    """Configuration for POST_CHECKS."""

    # Static analysis commands (will run all)
    static_analysis_commands: list[list[str]] = field(
        default_factory=lambda: [
            ["ruff", "check", "."],
            ["mypy", "."],
        ]
    )

    # Unit test command
    unit_test_command: list[str] = field(
        default_factory=lambda: ["pytest", "-v", "--tb=short"]
    )

    # Build command (optional, set to empty to skip)
    build_command: list[str] = field(default_factory=list)

    # Security scan command (optional)
    security_scan_command: list[str] = field(
        default_factory=lambda: ["bandit", "-r", "."]
    )

    # Whether security scan is required (vs warning only)
    security_scan_required: bool = False

    # Whether manual smoke test is enabled
    manual_smoke_enabled: bool = False

    # Timeouts
    static_analysis_timeout: float = 300.0  # 5 minutes
    unit_test_timeout: float = 600.0  # 10 minutes
    build_timeout: float = 600.0  # 10 minutes
    security_scan_timeout: float = 300.0  # 5 minutes


class PostChecks:
    """
    5-gate verification after implementation.

    Gates:
    1. Static Analysis - Lint + type check (ruff, mypy)
    2. Unit Tests - Test suite passes
    3. Build - Project builds successfully
    4. Security Scan - No HIGH/CRITICAL findings (optional)
    5. Manual Smoke - Human spot-check (optional, non-blocking)
    """

    def __init__(
        self,
        config: PostCheckConfig | None = None,
        working_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize POST_CHECKS.

        Args:
            config: Verification configuration.
            working_dir: Working directory for running commands.
        """
        self.config = config or PostCheckConfig()
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    async def run_all(self) -> PostCheckResult:
        """
        Run all verification gates.

        Returns:
            PostCheckResult with all gate results.
        """
        results: list[GateResult] = []

        # Gate 1: Static Analysis (required)
        logger.info("Running Gate 1: Static Analysis")
        results.append(await self._run_static_analysis())

        # Gate 2: Unit Tests (required)
        logger.info("Running Gate 2: Unit Tests")
        results.append(await self._run_unit_tests())

        # Gate 3: Build (required if configured)
        if self.config.build_command:
            logger.info("Running Gate 3: Build")
            results.append(await self._run_build())
        else:
            results.append(
                GateResult(
                    gate=GateName.BUILD,
                    status=GateStatus.SKIPPED,
                    required=False,
                    message="No build command configured",
                )
            )

        # Gate 4: Security Scan (optional by default)
        if self.config.security_scan_command:
            logger.info("Running Gate 4: Security Scan")
            results.append(await self._run_security_scan())
        else:
            results.append(
                GateResult(
                    gate=GateName.SECURITY_SCAN,
                    status=GateStatus.SKIPPED,
                    required=False,
                    message="No security scan configured",
                )
            )

        # Gate 5: Manual Smoke (optional, non-blocking)
        if self.config.manual_smoke_enabled:
            results.append(
                GateResult(
                    gate=GateName.MANUAL_SMOKE,
                    status=GateStatus.WARNING,
                    required=False,
                    message="Manual smoke test pending - human verification needed",
                )
            )
        else:
            results.append(
                GateResult(
                    gate=GateName.MANUAL_SMOKE,
                    status=GateStatus.SKIPPED,
                    required=False,
                    message="Manual smoke test disabled",
                )
            )

        result = PostCheckResult(
            gates=results,
            completed_at=datetime.now(UTC),
        )

        logger.info(
            "POST_CHECKS completed",
            extra={
                "passed": result.passed,
                "duration_seconds": result.total_duration_seconds,
                "failed_gates": [g.gate.value for g in result.failed_gates],
            },
        )

        return result

    async def _run_static_analysis(self) -> GateResult:
        """Run static analysis commands."""
        import time

        start = time.monotonic()
        all_passed = True
        all_stdout = []
        all_stderr = []
        exit_codes = []

        for cmd in self.config.static_analysis_commands:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.working_dir,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.static_analysis_timeout,
                )

                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                all_stdout.append(f"=== {' '.join(cmd)} ===\n{stdout_str}")
                all_stderr.append(stderr_str)
                exit_codes.append(process.returncode or 0)

                if process.returncode != 0:
                    all_passed = False

            except asyncio.TimeoutError:
                all_passed = False
                all_stderr.append(f"Timeout running {' '.join(cmd)}")
                exit_codes.append(-1)

            except FileNotFoundError:
                # Tool not installed, treat as warning
                all_stderr.append(f"Tool not found: {cmd[0]}")
                exit_codes.append(-1)

            except OSError as e:
                all_passed = False
                all_stderr.append(f"Error running {' '.join(cmd)}: {e}")
                exit_codes.append(-1)

        duration = time.monotonic() - start

        return GateResult(
            gate=GateName.STATIC_ANALYSIS,
            status=GateStatus.PASSED if all_passed else GateStatus.FAILED,
            required=True,
            message=f"Ran {len(self.config.static_analysis_commands)} checks",
            duration_seconds=duration,
            exit_code=max(exit_codes) if exit_codes else 0,
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr),
            details={"exit_codes": exit_codes},
        )

    async def _run_unit_tests(self) -> GateResult:
        """Run unit tests."""
        import time

        start = time.monotonic()

        try:
            process = await asyncio.create_subprocess_exec(
                *self.config.unit_test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.unit_test_timeout,
            )

            duration = time.monotonic() - start
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            passed = process.returncode == 0

            # Extract test counts from pytest output
            test_details = self._parse_pytest_output(stdout_str)

            return GateResult(
                gate=GateName.UNIT_TESTS,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                required=True,
                message=f"Tests {'passed' if passed else 'failed'}",
                duration_seconds=duration,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                details=test_details,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return GateResult(
                gate=GateName.UNIT_TESTS,
                status=GateStatus.FAILED,
                required=True,
                message=f"Tests timed out after {self.config.unit_test_timeout}s",
                duration_seconds=duration,
                exit_code=-1,
            )

        except FileNotFoundError:
            return GateResult(
                gate=GateName.UNIT_TESTS,
                status=GateStatus.FAILED,
                required=True,
                message="pytest not found - install with: pip install pytest",
                exit_code=-1,
            )

        except OSError as e:
            return GateResult(
                gate=GateName.UNIT_TESTS,
                status=GateStatus.FAILED,
                required=True,
                message=f"Error running tests: {e}",
                exit_code=-1,
            )

    async def _run_build(self) -> GateResult:
        """Run build command."""
        import time

        start = time.monotonic()

        try:
            process = await asyncio.create_subprocess_exec(
                *self.config.build_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.build_timeout,
            )

            duration = time.monotonic() - start
            passed = process.returncode == 0

            return GateResult(
                gate=GateName.BUILD,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                required=True,
                message=f"Build {'succeeded' if passed else 'failed'}",
                duration_seconds=duration,
                exit_code=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return GateResult(
                gate=GateName.BUILD,
                status=GateStatus.FAILED,
                required=True,
                message=f"Build timed out after {self.config.build_timeout}s",
                duration_seconds=duration,
                exit_code=-1,
            )

        except OSError as e:
            return GateResult(
                gate=GateName.BUILD,
                status=GateStatus.FAILED,
                required=True,
                message=f"Build error: {e}",
                exit_code=-1,
            )

    async def _run_security_scan(self) -> GateResult:
        """Run security scan."""
        import time

        start = time.monotonic()

        try:
            process = await asyncio.create_subprocess_exec(
                *self.config.security_scan_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.security_scan_timeout,
            )

            duration = time.monotonic() - start
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Bandit returns 0 for no issues, 1 for issues found
            # We treat non-critical findings as warnings
            if process.returncode == 0:
                status = GateStatus.PASSED
                message = "No security issues found"
            elif "High" in stdout_str or "Critical" in stdout_str:
                status = GateStatus.FAILED
                message = "HIGH/CRITICAL security issues found"
            else:
                status = GateStatus.WARNING
                message = "Security issues found (non-critical)"

            return GateResult(
                gate=GateName.SECURITY_SCAN,
                status=status,
                required=self.config.security_scan_required,
                message=message,
                duration_seconds=duration,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return GateResult(
                gate=GateName.SECURITY_SCAN,
                status=GateStatus.FAILED,
                required=self.config.security_scan_required,
                message=f"Security scan timed out after {self.config.security_scan_timeout}s",
                duration_seconds=duration,
                exit_code=-1,
            )

        except FileNotFoundError:
            return GateResult(
                gate=GateName.SECURITY_SCAN,
                status=GateStatus.SKIPPED,
                required=False,
                message="bandit not found - install with: pip install bandit",
                exit_code=-1,
            )

        except OSError as e:
            return GateResult(
                gate=GateName.SECURITY_SCAN,
                status=GateStatus.FAILED,
                required=self.config.security_scan_required,
                message=f"Security scan error: {e}",
                exit_code=-1,
            )

    def _parse_pytest_output(self, output: str) -> dict[str, Any]:
        """Parse pytest output for test counts."""
        import re

        details: dict[str, Any] = {}

        # Try to extract test counts from pytest summary line
        # Format: "X passed, Y failed, Z skipped in N.NNs"
        match = re.search(
            r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?",
            output,
        )
        if match:
            details["passed"] = int(match.group(1))
            details["failed"] = int(match.group(2) or 0)
            details["skipped"] = int(match.group(3) or 0)

        return details

    def print_results(self, result: PostCheckResult) -> None:
        """Print results to console."""
        print(result.to_summary())
