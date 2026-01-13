"""Tests for project discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_orchestrator.project.context import ProjectContext, ProjectConfig
from ai_orchestrator.project.discovery import ProjectContextDiscovery


class TestProjectContextDiscovery:
    """Tests for project discovery."""

    @pytest.mark.asyncio
    async def test_discover_empty_project(self, temp_project_dir: Path) -> None:
        """Test discovery on empty project."""
        discovery = ProjectContextDiscovery()
        context = await discovery.discover(temp_project_dir)

        assert context.root == temp_project_dir.resolve()
        assert context.instructions_path is None
        assert context.debug_script_path is None
        assert context.discovery_method == "auto"

    @pytest.mark.asyncio
    async def test_discover_with_claude_md(
        self, temp_project_with_claude_md: Path
    ) -> None:
        """Test discovery finds CLAUDE.md."""
        discovery = ProjectContextDiscovery()
        context = await discovery.discover(temp_project_with_claude_md)

        assert context.instructions_path is not None
        assert context.instructions_path.name == "CLAUDE.md"
        assert context.has_instructions is True

    @pytest.mark.asyncio
    async def test_discovery_caching(self, temp_project_dir: Path) -> None:
        """Test that discovery results are cached."""
        discovery = ProjectContextDiscovery()

        context1 = await discovery.discover(temp_project_dir)
        context2 = await discovery.discover(temp_project_dir)

        # Should return same instance (cached)
        assert context1 is context2

        # Clear cache and get new instance
        discovery.clear_cache()
        context3 = await discovery.discover(temp_project_dir)
        assert context3 is not context1


class TestProjectContext:
    """Tests for ProjectContext."""

    def test_context_properties(self, temp_project_dir: Path) -> None:
        """Test context property methods."""
        context = ProjectContext(root=temp_project_dir)

        assert context.has_instructions is False
        assert context.has_debug_script is False
        assert context.has_patterns is False
        assert context.has_registry is False

    def test_context_with_paths(self, temp_project_dir: Path) -> None:
        """Test context with discovered paths."""
        claude_md = temp_project_dir / "CLAUDE.md"
        claude_md.write_text("# Instructions")

        context = ProjectContext(
            root=temp_project_dir,
            instructions_path=claude_md,
        )

        assert context.has_instructions is True

    def test_summary(self, temp_project_dir: Path) -> None:
        """Test summary generation."""
        context = ProjectContext(root=temp_project_dir)
        summary = context.summary()

        assert temp_project_dir.name in summary
        assert "auto" in summary

    def test_get_verification_commands(self, temp_project_dir: Path) -> None:
        """Test getting verification commands."""
        context = ProjectContext(root=temp_project_dir)
        commands = context.get_verification_commands()

        assert "static_analysis" in commands
        assert "unit_tests" in commands
        assert commands["unit_tests"] == "pytest tests/ -v"


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ProjectConfig()

        assert config.version == "1.0"
        assert config.verification.unit_tests == "pytest tests/ -v"
        assert len(config.verification.static_analysis) == 2

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ProjectConfig(
            version="2.0",
            name="Custom Project",
        )

        assert config.version == "2.0"
        assert config.name == "Custom Project"
