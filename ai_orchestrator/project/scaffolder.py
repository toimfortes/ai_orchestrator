"""Foundation scaffolder for greenfield projects.

Scaffolds workflow scripts, best practices, and templates into new projects
with rollback support on failure.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from ai_orchestrator.project.foundation_registry import FoundationRegistry

logger = logging.getLogger(__name__)


class ScaffoldingError(Exception):
    """Error during scaffolding process."""

    pass


class FoundationScaffolder:
    """
    Scaffolds foundation tools into new projects with rollback support.

    Creates:
    - scripts/ directory with workflow scripts
    - best_practices/ directory with patterns
    - data/ directory for code catalog
    - CLAUDE.md with project-specific values
    """

    def __init__(
        self,
        project_root: Path,
        orchestrator_root: Path | None = None,
    ) -> None:
        """
        Initialize scaffolder.

        Args:
            project_root: Target project to scaffold into.
            orchestrator_root: Source of foundation files. If None, auto-detected.
        """
        self.project_root = project_root.resolve()
        self.orchestrator_root = (
            orchestrator_root or FoundationRegistry.get_orchestrator_root()
        )
        self.created_paths: list[Path] = []  # Track for rollback

    async def scaffold_all(
        self,
        project_name: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Scaffold all foundation files with rollback on failure.

        Args:
            project_name: Name to use in templates. Defaults to directory name.
            dry_run: If True, only report what would be done.

        Returns:
            Dict with scaffolding results.

        Raises:
            ScaffoldingError: If scaffolding fails and rollback completes.
        """
        project_name = project_name or self.project_root.name
        results: dict[str, Any] = {
            "project_name": project_name,
            "project_root": str(self.project_root),
            "dry_run": dry_run,
            "created_files": [],
            "created_dirs": [],
            "skipped": [],
            "errors": [],
        }

        if dry_run:
            return self._dry_run_report(project_name)

        try:
            # Create directories first
            await self._create_directories(results)

            # Copy scripts
            await self._copy_scripts(results)

            # Copy best practices
            await self._copy_best_practices(results)

            # Generate CLAUDE.md from template
            await self._generate_claude_md(project_name, results)

            logger.info(
                "Foundation scaffolding complete: %d files, %d dirs created",
                len(results["created_files"]),
                len(results["created_dirs"]),
            )

            return results

        except Exception as e:
            logger.error("Scaffolding failed, rolling back: %s", e, exc_info=True)
            await self._rollback()
            results["errors"].append(str(e))
            results["rolled_back"] = True
            raise ScaffoldingError(f"Scaffolding failed: {e}") from e

    async def _create_directories(self, results: dict[str, Any]) -> None:
        """Create required directories."""
        for dir_name in FoundationRegistry.DIRECTORIES:
            dir_path = self.project_root / dir_name

            if dir_path.exists():
                results["skipped"].append(f"{dir_name}/ (exists)")
                continue

            dir_path.mkdir(parents=True, exist_ok=True)
            self.created_paths.append(dir_path)
            results["created_dirs"].append(dir_name)
            logger.debug("Created directory: %s", dir_path)

    async def _copy_scripts(self, results: dict[str, Any]) -> None:
        """Copy workflow scripts from live source."""
        scripts_dir = self.project_root / "scripts"

        for src_path, dst_name in FoundationRegistry.get_script_paths():
            dst_path = scripts_dir / dst_name

            if dst_path.exists():
                results["skipped"].append(f"scripts/{dst_name} (exists)")
                continue

            shutil.copy2(src_path, dst_path)
            self.created_paths.append(dst_path)
            results["created_files"].append(f"scripts/{dst_name}")
            logger.debug("Copied script: %s", dst_name)

    async def _copy_best_practices(self, results: dict[str, Any]) -> None:
        """Copy best practices patterns from live source."""
        bp_dir = self.project_root / "best_practices"

        for src_path, dst_name in FoundationRegistry.get_best_practices_paths():
            dst_path = bp_dir / dst_name

            if dst_path.exists():
                results["skipped"].append(f"best_practices/{dst_name} (exists)")
                continue

            shutil.copy2(src_path, dst_path)
            self.created_paths.append(dst_path)
            results["created_files"].append(f"best_practices/{dst_name}")
            logger.debug("Copied best practice: %s", dst_name)

    async def _generate_claude_md(
        self,
        project_name: str,
        results: dict[str, Any],
    ) -> None:
        """Generate project-specific CLAUDE.md from template."""
        dst_path = self.project_root / "CLAUDE.md"

        if dst_path.exists():
            results["skipped"].append("CLAUDE.md (exists)")
            return

        template_path = FoundationRegistry.get_template_path("CLAUDE.md")

        if template_path is None:
            # Use built-in fallback template
            content = self._get_fallback_claude_md(project_name)
        else:
            template = template_path.read_text(encoding="utf-8")
            content = template.format(project_name=project_name)

        dst_path.write_text(content, encoding="utf-8")
        self.created_paths.append(dst_path)
        results["created_files"].append("CLAUDE.md")
        logger.debug("Generated CLAUDE.md")

    async def _rollback(self) -> None:
        """Remove all created files/directories on failure."""
        logger.info("Rolling back %d created paths", len(self.created_paths))

        # Process in reverse order (files before directories)
        for path in reversed(self.created_paths):
            try:
                if path.is_file():
                    path.unlink()
                    logger.debug("Rolled back file: %s", path)
                elif path.is_dir():
                    # Only remove if empty
                    if not any(path.iterdir()):
                        path.rmdir()
                        logger.debug("Rolled back directory: %s", path)
            except Exception as e:
                logger.warning("Rollback failed for %s: %s", path, e)

    def _dry_run_report(self, project_name: str) -> dict[str, Any]:
        """Generate report of what would be scaffolded."""
        results: dict[str, Any] = {
            "project_name": project_name,
            "project_root": str(self.project_root),
            "dry_run": True,
            "would_create_files": [],
            "would_create_dirs": [],
            "would_skip": [],
        }

        # Check directories
        for dir_name in FoundationRegistry.DIRECTORIES:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                results["would_skip"].append(f"{dir_name}/ (exists)")
            else:
                results["would_create_dirs"].append(dir_name)

        # Check scripts
        for src_path, dst_name in FoundationRegistry.get_script_paths():
            dst_path = self.project_root / "scripts" / dst_name
            if dst_path.exists():
                results["would_skip"].append(f"scripts/{dst_name} (exists)")
            else:
                results["would_create_files"].append(f"scripts/{dst_name}")

        # Check best practices
        for src_path, dst_name in FoundationRegistry.get_best_practices_paths():
            dst_path = self.project_root / "best_practices" / dst_name
            if dst_path.exists():
                results["would_skip"].append(f"best_practices/{dst_name} (exists)")
            else:
                results["would_create_files"].append(f"best_practices/{dst_name}")

        # Check CLAUDE.md
        if (self.project_root / "CLAUDE.md").exists():
            results["would_skip"].append("CLAUDE.md (exists)")
        else:
            results["would_create_files"].append("CLAUDE.md")

        return results

    def _get_fallback_claude_md(self, project_name: str) -> str:
        """Get fallback CLAUDE.md content if template not found."""
        return f'''# AI Coding Standards - {project_name}

## Purpose
- Governs code generation for the {project_name} project
- Enforces: SOLID principles, composition + DI, modularity, tests, observability

## Pre-Implementation Workflow ("Measure Twice, Cut Once")

Before making changes, run these checks:

### 1. Update Code Catalog
```bash
python scripts/build_code_registry.py
```

### 2. Measure Blast Radius
```bash
python scripts/measure_blast_radius.py <file_to_change>
```

### 3. Check Patterns
```bash
python scripts/check_patterns.py <file_to_change>
```

## Post-Implementation Workflow

After making changes:

### 1. Run Full Validation
```bash
python scripts/post_implementation_check.py --all
```

### 2. Debug Any Issues
```bash
python scripts/debug_everything.py
```

## Quick Rules

### Never
- Hardcoded configuration values in code files
- Tests outside tests/ directory
- Log errors without traceback (use exc_info=True)
- Block event loops with sync calls in async code

### Always
- Use structured logging with context
- Type hints on all public functions
- Pydantic validation for inputs
- Circuit breakers for external API calls
'''
