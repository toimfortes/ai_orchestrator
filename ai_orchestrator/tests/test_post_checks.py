"""Tests for POST_CHECKS verification system."""

import pytest

from ai_orchestrator.core.post_checks import (
    GateName,
    GateResult,
    GateStatus,
    PostCheckConfig,
    PostCheckResult,
    PostChecks,
)


class TestGateResult:
    """Tests for GateResult."""

    def test_passed_required_gate(self):
        """Test passed property for required gate."""
        result = GateResult(
            gate=GateName.STATIC_ANALYSIS,
            status=GateStatus.PASSED,
            required=True,
        )
        assert result.passed is True

    def test_failed_required_gate(self):
        """Test passed property for failed required gate."""
        result = GateResult(
            gate=GateName.UNIT_TESTS,
            status=GateStatus.FAILED,
            required=True,
        )
        assert result.passed is False

    def test_skipped_optional_gate(self):
        """Test passed property for skipped optional gate."""
        result = GateResult(
            gate=GateName.SECURITY_SCAN,
            status=GateStatus.SKIPPED,
            required=False,
        )
        assert result.passed is True

    def test_warning_optional_gate(self):
        """Test passed property for warning optional gate."""
        result = GateResult(
            gate=GateName.MANUAL_SMOKE,
            status=GateStatus.WARNING,
            required=False,
        )
        assert result.passed is True

    def test_warning_required_gate_fails(self):
        """Test passed property for warning on required gate."""
        result = GateResult(
            gate=GateName.SECURITY_SCAN,
            status=GateStatus.WARNING,
            required=True,
        )
        assert result.passed is False


class TestPostCheckResult:
    """Tests for PostCheckResult."""

    def test_all_gates_passed(self):
        """Test passed property when all gates pass."""
        result = PostCheckResult(
            gates=[
                GateResult(gate=GateName.STATIC_ANALYSIS, status=GateStatus.PASSED, required=True),
                GateResult(gate=GateName.UNIT_TESTS, status=GateStatus.PASSED, required=True),
                GateResult(gate=GateName.BUILD, status=GateStatus.SKIPPED, required=False),
                GateResult(gate=GateName.SECURITY_SCAN, status=GateStatus.WARNING, required=False),
            ]
        )
        assert result.passed is True
        assert len(result.failed_gates) == 0

    def test_required_gate_failed(self):
        """Test passed property when required gate fails."""
        result = PostCheckResult(
            gates=[
                GateResult(gate=GateName.STATIC_ANALYSIS, status=GateStatus.PASSED, required=True),
                GateResult(gate=GateName.UNIT_TESTS, status=GateStatus.FAILED, required=True),
                GateResult(gate=GateName.BUILD, status=GateStatus.SKIPPED, required=False),
            ]
        )
        assert result.passed is False
        assert len(result.failed_gates) == 1
        assert result.failed_gates[0].gate == GateName.UNIT_TESTS

    def test_total_duration(self):
        """Test total duration calculation."""
        result = PostCheckResult(
            gates=[
                GateResult(
                    gate=GateName.STATIC_ANALYSIS,
                    status=GateStatus.PASSED,
                    duration_seconds=10.5,
                ),
                GateResult(
                    gate=GateName.UNIT_TESTS,
                    status=GateStatus.PASSED,
                    duration_seconds=25.3,
                ),
            ]
        )
        assert result.total_duration_seconds == pytest.approx(35.8)

    def test_get_gate(self):
        """Test getting specific gate result."""
        result = PostCheckResult(
            gates=[
                GateResult(gate=GateName.STATIC_ANALYSIS, status=GateStatus.PASSED),
                GateResult(gate=GateName.UNIT_TESTS, status=GateStatus.FAILED),
            ]
        )

        static = result.get_gate(GateName.STATIC_ANALYSIS)
        assert static is not None
        assert static.status == GateStatus.PASSED

        unit = result.get_gate(GateName.UNIT_TESTS)
        assert unit is not None
        assert unit.status == GateStatus.FAILED

        build = result.get_gate(GateName.BUILD)
        assert build is None

    def test_warnings_list(self):
        """Test warnings property."""
        result = PostCheckResult(
            gates=[
                GateResult(gate=GateName.STATIC_ANALYSIS, status=GateStatus.PASSED),
                GateResult(gate=GateName.SECURITY_SCAN, status=GateStatus.WARNING),
                GateResult(gate=GateName.MANUAL_SMOKE, status=GateStatus.WARNING),
            ]
        )
        assert len(result.warnings) == 2

    def test_to_summary(self):
        """Test summary generation."""
        result = PostCheckResult(
            gates=[
                GateResult(
                    gate=GateName.STATIC_ANALYSIS,
                    status=GateStatus.PASSED,
                    required=True,
                    message="Ran 2 checks",
                    duration_seconds=5.0,
                ),
                GateResult(
                    gate=GateName.UNIT_TESTS,
                    status=GateStatus.FAILED,
                    required=True,
                    message="Tests failed",
                    duration_seconds=10.0,
                ),
            ]
        )

        summary = result.to_summary()
        assert "POST_CHECKS" in summary
        assert "FAILED" in summary
        assert "static_analysis" in summary
        assert "unit_tests" in summary
        assert "[+]" in summary  # Passed indicator
        assert "[X]" in summary  # Failed indicator


class TestPostCheckConfig:
    """Tests for PostCheckConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PostCheckConfig()

        assert len(config.static_analysis_commands) == 2
        assert config.unit_test_command == ["pytest", "-v", "--tb=short"]
        assert config.build_command == []
        assert config.security_scan_required is False
        assert config.manual_smoke_enabled is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PostCheckConfig(
            static_analysis_commands=[["ruff", "check", "src/"]],
            unit_test_command=["pytest", "tests/", "-v"],
            build_command=["npm", "run", "build"],
            security_scan_required=True,
            manual_smoke_enabled=True,
        )

        assert len(config.static_analysis_commands) == 1
        assert config.build_command == ["npm", "run", "build"]
        assert config.security_scan_required is True
        assert config.manual_smoke_enabled is True


class TestPostChecks:
    """Tests for PostChecks class."""

    @pytest.mark.asyncio
    async def test_skipped_build_when_not_configured(self):
        """Test that build is skipped when not configured."""
        config = PostCheckConfig(
            static_analysis_commands=[],  # Skip to test faster
            unit_test_command=["python", "-c", "pass"],  # Quick pass
            build_command=[],  # Not configured
            security_scan_command=[],  # Not configured
        )

        checks = PostChecks(config=config)
        result = await checks.run_all()

        build_gate = result.get_gate(GateName.BUILD)
        assert build_gate is not None
        assert build_gate.status == GateStatus.SKIPPED
        assert build_gate.required is False

    @pytest.mark.asyncio
    async def test_manual_smoke_warning_when_enabled(self):
        """Test manual smoke shows as warning when enabled."""
        config = PostCheckConfig(
            static_analysis_commands=[],
            unit_test_command=["python", "-c", "pass"],
            security_scan_command=[],
            manual_smoke_enabled=True,
        )

        checks = PostChecks(config=config)
        result = await checks.run_all()

        smoke_gate = result.get_gate(GateName.MANUAL_SMOKE)
        assert smoke_gate is not None
        assert smoke_gate.status == GateStatus.WARNING
        assert smoke_gate.required is False
        assert "human verification" in smoke_gate.message.lower()

    def test_print_results(self, capsys):
        """Test that print_results outputs summary."""
        result = PostCheckResult(
            gates=[
                GateResult(
                    gate=GateName.STATIC_ANALYSIS,
                    status=GateStatus.PASSED,
                    required=True,
                ),
            ]
        )

        checks = PostChecks()
        checks.print_results(result)

        captured = capsys.readouterr()
        assert "POST_CHECKS" in captured.out
        assert "static_analysis" in captured.out
