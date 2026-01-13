"""Load explicit project configuration from .ai_orchestrator.yaml."""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

import yaml
from pydantic import ValidationError

from ai_orchestrator.project.context import ProjectConfig, ProjectContext
from ai_orchestrator.project.discovery import ProjectContextDiscovery

logger = logging.getLogger(__name__)

CONFIG_FILENAME = ".ai_orchestrator.yaml"


def load_project_config(project_root: Path) -> ProjectConfig | None:
    """
    Load project configuration from .ai_orchestrator.yaml if it exists.

    Args:
        project_root: Path to the project root directory.

    Returns:
        ProjectConfig if file exists and is valid, None otherwise.
    """
    config_path = project_root / CONFIG_FILENAME

    if not config_path.exists():
        logger.debug("No %s found in %s", CONFIG_FILENAME, project_root)
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            logger.warning("Empty %s in %s", CONFIG_FILENAME, project_root)
            return None

        # Handle nested 'project' key if present
        if "project" in data:
            project_data = data.pop("project")
            data.update(project_data)

        config = ProjectConfig.model_validate(data)
        logger.info("Loaded project config from %s", config_path)
        return config

    except yaml.YAMLError as e:
        logger.error("Failed to parse %s: %s", config_path, e)
        return None
    except ValidationError as e:
        logger.error("Invalid config in %s: %s", config_path, e)
        return None


async def load_project_context(
    project_root: Path,
    prefer_config: bool = True,
) -> ProjectContext:
    """
    Load project context, preferring explicit config over auto-discovery.

    Args:
        project_root: Path to the project root directory.
        prefer_config: If True, use .ai_orchestrator.yaml when available.

    Returns:
        ProjectContext with discovered or configured conventions.
    """
    project_root = project_root.resolve()

    # Try loading explicit config first
    config = load_project_config(project_root) if prefer_config else None

    # Auto-discover conventions
    discovery = ProjectContextDiscovery()
    context = await discovery.discover(project_root)

    # Merge explicit config if available
    if config:
        context.config = config
        context.discovery_method = "config"

        # Override discovered paths with explicit config
        if config.instructions:
            path = project_root / config.instructions
            if path.exists():
                context.instructions_path = path

        if config.patterns_dir:
            path = project_root / config.patterns_dir
            if path.is_dir():
                context.patterns_dir = path

        if config.registry:
            path = project_root / config.registry
            if path.exists():
                context.registry_path = path

    # Load instructions content if available
    if context.instructions_path and context.instructions_path.exists():
        try:
            context.instructions_content = context.instructions_path.read_text(
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Failed to read instructions: %s", e)

    return context


def create_default_config(project_root: Path) -> Path:
    """
    Create a default .ai_orchestrator.yaml file.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Path to the created config file.
    """
    config_path = project_root / CONFIG_FILENAME

    default_config: dict[str, Any] = {
        "version": "1.0",
        "project": {
            "name": project_root.name,
            "instructions": "CLAUDE.md",
            "patterns_dir": "best_practices/",
            "registry": "data/code_catalog.json",
        },
        "verification": {
            "debug_script": "scripts/debug_everything.py --quick",
            "static_analysis": ["ruff check .", "mypy ."],
            "unit_tests": "pytest tests/ -v",
        },
        "context_injection": {
            "include_files": ["CLAUDE.md"],
            "summarize_dirs": ["src/", "backend/"],
        },
        "scoring": {
            "security": 20,
            "performance": 10,
            "maintainability": 15,
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(default_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Created default config at %s", config_path)
    return config_path
