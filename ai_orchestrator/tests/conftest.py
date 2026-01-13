"""Test fixtures for AI Orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
import tempfile
import shutil

import pytest


@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory."""
    temp_dir = tempfile.mkdtemp(prefix="ai_orchestrator_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_project_with_claude_md(temp_project_dir: Path) -> Path:
    """Create a temp project with CLAUDE.md."""
    claude_md = temp_project_dir / "CLAUDE.md"
    claude_md.write_text("# Project Instructions\n\nTest project instructions.")
    return temp_project_dir


@pytest.fixture
def temp_project_with_config(temp_project_dir: Path) -> Path:
    """Create a temp project with .ai_orchestrator.yaml."""
    config = temp_project_dir / ".ai_orchestrator.yaml"
    config.write_text("""
version: "1.0"
project:
  name: "Test Project"
  instructions: "CLAUDE.md"
verification:
  unit_tests: "pytest tests/ -v"
""")

    # Also create CLAUDE.md
    claude_md = temp_project_dir / "CLAUDE.md"
    claude_md.write_text("# Test Instructions")

    return temp_project_dir
