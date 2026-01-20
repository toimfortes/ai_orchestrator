"""Base class for CLI adapters."""

from __future__ import annotations

import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any


def find_npm_executable(name: str) -> str | None:
    """
    Find an npm-installed CLI executable, handling Windows .cmd files.

    Args:
        name: The CLI name (e.g., "claude", "gemini", "codex")

    Returns:
        Full path to executable, or None if not found.
    """
    # Try standard lookup first
    exe = shutil.which(name)
    if exe:
        return exe

    # On Windows, npm installs create .cmd wrapper files
    if sys.platform == "win32":
        # Try with .cmd extension
        exe = shutil.which(f"{name}.cmd")
        if exe:
            return exe

        # Check common npm global path
        npm_path = Path(os.environ.get("APPDATA", "")) / "npm" / f"{name}.cmd"
        if npm_path.exists():
            return str(npm_path)

    return None


class CLIStatus(str, Enum):
    """Status of a CLI invocation."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    AUTH_ERROR = "auth_error"


@dataclass
class CLIResult:
    """Result from a CLI invocation."""

    cli_name: str
    status: CLIStatus
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the invocation was successful."""
        return self.status == CLIStatus.SUCCESS and self.exit_code == 0

    @property
    def output(self) -> str:
        """Primary output (stdout, or stderr if stdout is empty)."""
        return self.stdout.strip() or self.stderr.strip()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cli_name": self.cli_name,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class CLIAdapter(ABC):
    """
    Abstract base class for CLI adapters.

    Provides a unified interface for invoking different AI coding CLIs.
    """

    def __init__(
        self,
        name: str,
        default_timeout: float = 900.0,  # 15 minutes
    ) -> None:
        self.name = name
        self.default_timeout = default_timeout

    @abstractmethod
    async def invoke(
        self,
        prompt: str,
        *,
        continue_session: bool = False,
        session_id: str | None = None,
        planning_mode: bool = False,
        timeout_seconds: float | None = None,
        working_dir: str | None = None,
        extra_args: list[str] | None = None,
    ) -> CLIResult:
        """
        Invoke the CLI with a prompt.

        Args:
            prompt: The prompt to send to the CLI.
            continue_session: Whether to continue a previous session.
            session_id: ID of session to continue (if continue_session=True).
            planning_mode: Whether to run in planning mode.
            timeout_seconds: Override default timeout.
            working_dir: Working directory for the CLI.
            extra_args: Additional CLI arguments.

        Returns:
            CLIResult with the invocation result.
        """
        pass

    @abstractmethod
    async def check_auth(self) -> bool:
        """
        Check if the CLI is authenticated.

        Returns:
            True if authenticated, False otherwise.
        """
        pass

    @abstractmethod
    def get_auth_command(self) -> str:
        """
        Get the command to authenticate this CLI.

        Returns:
            Command string for authentication.
        """
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this CLI is available on the system."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
