"""
AI Orchestrator - Multi-AI Code Generation and Review System

A project-agnostic orchestration infrastructure for code generation
and review workflows using multiple AI coding CLIs.
"""

__version__ = "0.1.0"
__author__ = "Auto Coder Multi Agents"

from ai_orchestrator.core.orchestrator import Orchestrator
from ai_orchestrator.project.context import ProjectContext
from ai_orchestrator.project.discovery import ProjectContextDiscovery

__all__ = [
    "Orchestrator",
    "ProjectContext",
    "ProjectContextDiscovery",
]
