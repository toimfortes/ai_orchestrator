"""Project discovery and context module."""

from ai_orchestrator.project.context import ProjectContext
from ai_orchestrator.project.discovery import ProjectContextDiscovery
from ai_orchestrator.project.loader import load_project_config

__all__ = ["ProjectContext", "ProjectContextDiscovery", "load_project_config"]
