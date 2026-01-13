"""Auto-discovery of project conventions."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import logging

from ai_orchestrator.project.context import ProjectContext

logger = logging.getLogger(__name__)


class ProjectContextDiscovery:
    """Auto-discover project-specific conventions."""

    DISCOVERY_PATTERNS: dict[str, list[str]] = {
        # Coding standards / instructions
        "instructions": [
            "CLAUDE.md",
            "claude.md",
            "agents.md",
            "AGENTS.md",
            ".claude/instructions.md",
            "docs/CLAUDE.md",
        ],
        # Debug/verification scripts
        "debug_script": [
            "scripts/debug_everything.py",
            "scripts/debug.py",
            "tools/debug.py",
            "bin/debug",
        ],
        # Pattern libraries
        "patterns": [
            "best_practices/",
            "patterns/",
            ".patterns/",
        ],
        # Code registries
        "registry": [
            "data/code_catalog.json",
            "catalog.json",
            ".code_registry.json",
        ],
        # Blast radius analysis
        "blast_radius": [
            "scripts/measure_blast_radius.py",
            "tools/blast_radius.py",
        ],
        # Pre-commit hooks (for verification)
        "pre_commit": [".pre-commit-config.yaml"],
        # Test commands
        "tests": ["pytest.ini", "pyproject.toml", "setup.cfg"],
    }

    def __init__(self) -> None:
        self._cache: dict[str, ProjectContext] = {}

    async def discover(self, project_root: Path) -> ProjectContext:
        """
        Scan project and return discovered conventions.

        Args:
            project_root: Path to the project root directory.

        Returns:
            ProjectContext with discovered paths and configuration.
        """
        # Check cache
        cache_key = str(project_root.resolve())
        if cache_key in self._cache:
            logger.debug("Using cached project context for %s", project_root)
            return self._cache[cache_key]

        context = ProjectContext(root=project_root.resolve())
        context.discovered_at = datetime.now(UTC).isoformat()

        # Discover each category
        for category, patterns in self.DISCOVERY_PATTERNS.items():
            discovered_path = self._find_first_match(project_root, patterns)
            if discovered_path:
                self._set_context_path(context, category, discovered_path)
                logger.info("Discovered %s: %s", category, discovered_path)

        # Cache and return
        self._cache[cache_key] = context
        return context

    def _find_first_match(self, root: Path, patterns: list[str]) -> Path | None:
        """Find the first matching pattern in the project."""
        for pattern in patterns:
            path = root / pattern

            # Handle directory patterns (ending with /)
            if pattern.endswith("/"):
                check_path = root / pattern.rstrip("/")
                if check_path.is_dir():
                    return check_path
            elif path.exists():
                return path

        return None

    def _set_context_path(
        self, context: ProjectContext, category: str, path: Path
    ) -> None:
        """Set the appropriate context attribute for a category."""
        mapping = {
            "instructions": "instructions_path",
            "debug_script": "debug_script_path",
            "patterns": "patterns_dir",
            "registry": "registry_path",
            "blast_radius": "blast_radius_script",
            "pre_commit": "pre_commit_config",
            "tests": "test_config_path",
        }

        attr_name = mapping.get(category)
        if attr_name:
            setattr(context, attr_name, path)

    def clear_cache(self) -> None:
        """Clear the discovery cache."""
        self._cache.clear()


# Convenience function
async def discover_project(project_root: Path) -> ProjectContext:
    """Discover project conventions (convenience wrapper)."""
    discovery = ProjectContextDiscovery()
    return await discovery.discover(project_root)
