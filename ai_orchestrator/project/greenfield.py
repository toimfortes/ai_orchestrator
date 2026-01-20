"""Greenfield (new/empty) project detection.

Detects project maturity level to determine if foundation scaffolding is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ProjectMaturity(str, Enum):
    """Classification of project maturity based on source file count."""

    GREENFIELD = "greenfield"  # Empty/near-empty (0-5 source files)
    NASCENT = "nascent"  # Has structure, minimal code (5-20 files)
    ESTABLISHED = "established"  # Significant codebase (20+ files)


@dataclass
class GreenfieldAnalysis:
    """Results of greenfield analysis."""

    maturity: ProjectMaturity
    source_file_count: int
    has_readme: bool
    has_tests_folder: bool
    has_git: bool
    has_claude_md: bool
    has_scripts_folder: bool
    needs_scaffolding: bool

    @property
    def summary(self) -> str:
        """Human-readable summary of analysis."""
        return (
            f"Project maturity: {self.maturity.value} "
            f"({self.source_file_count} source files, "
            f"scaffolding {'needed' if self.needs_scaffolding else 'not needed'})"
        )


class GreenfieldDetector:
    """Detects if a project is greenfield and needs foundation scaffolding."""

    # Maximum source files to be considered greenfield
    GREENFIELD_THRESHOLD = 5

    # Maximum source files to be considered nascent
    NASCENT_THRESHOLD = 20

    # Directories to EXCLUDE from source file count
    EXCLUDE_DIRS = {
        ".git",
        "node_modules",
        "vendor",
        "docs",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "egg-info",
        ".tox",
        ".nox",
        "coverage",
        ".coverage",
    }

    # Source file extensions to count
    SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".scala",
        ".rb",
        ".php",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
    }

    async def analyze(self, project_root: Path) -> GreenfieldAnalysis:
        """
        Analyze a project to determine its maturity level.

        Args:
            project_root: Path to the project root directory.

        Returns:
            GreenfieldAnalysis with detection results.
        """
        project_root = project_root.resolve()

        source_count = self._count_source_files(project_root)
        maturity = self._classify_maturity(source_count)

        has_claude_md = (project_root / "CLAUDE.md").exists()
        has_scripts = (project_root / "scripts").is_dir()

        # Needs scaffolding if greenfield AND doesn't already have foundations
        needs_scaffolding = (
            maturity == ProjectMaturity.GREENFIELD
            and not has_claude_md
            and not has_scripts
        )

        analysis = GreenfieldAnalysis(
            maturity=maturity,
            source_file_count=source_count,
            has_readme=(project_root / "README.md").exists(),
            has_tests_folder=(project_root / "tests").is_dir(),
            has_git=(project_root / ".git").is_dir(),
            has_claude_md=has_claude_md,
            has_scripts_folder=has_scripts,
            needs_scaffolding=needs_scaffolding,
        )

        logger.info("Greenfield analysis: %s", analysis.summary)
        return analysis

    def _count_source_files(self, root: Path) -> int:
        """
        Count source files in project, excluding non-source directories.

        Args:
            root: Project root path.

        Returns:
            Number of source files found.
        """
        count = 0

        try:
            for path in root.rglob("*"):
                # Skip if in excluded directory
                if any(excl in path.parts for excl in self.EXCLUDE_DIRS):
                    continue

                # Count if it's a source file
                if path.is_file() and path.suffix.lower() in self.SOURCE_EXTENSIONS:
                    count += 1

        except PermissionError as e:
            logger.warning("Permission denied while scanning: %s", e)
        except OSError as e:
            logger.warning("OS error while scanning: %s", e)

        return count

    def _classify_maturity(self, source_count: int) -> ProjectMaturity:
        """
        Classify project maturity based on source file count.

        Args:
            source_count: Number of source files.

        Returns:
            ProjectMaturity classification.
        """
        if source_count <= self.GREENFIELD_THRESHOLD:
            return ProjectMaturity.GREENFIELD
        elif source_count <= self.NASCENT_THRESHOLD:
            return ProjectMaturity.NASCENT
        else:
            return ProjectMaturity.ESTABLISHED
