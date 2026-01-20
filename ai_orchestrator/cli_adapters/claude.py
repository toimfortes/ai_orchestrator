"""Claude Code CLI adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from ai_orchestrator.cli_adapters.base import (
    CLIAdapter,
    CLIResult,
    CLIStatus,
    find_npm_executable,
)
from ai_orchestrator.utils.sanitization import PromptSanitizer

logger = logging.getLogger(__name__)


class ClaudeAdapter(CLIAdapter):
    """
    Adapter for Claude Code CLI.

    Uses browser-based OAuth authentication (no API keys).
    Supports JSON output, session resumption, and planning mode.
    """

    CLI_NAME = "claude"
    CONFIG_PATH = Path.home() / ".claude"

    def __init__(
        self,
        default_timeout: float = 900.0,
        sanitizer: PromptSanitizer | None = None,
    ) -> None:
        super().__init__(name=self.CLI_NAME, default_timeout=default_timeout)
        self.sanitizer = sanitizer or PromptSanitizer()
        self._available: bool | None = None
        self._executable: str | None = None

    @property
    def executable(self) -> str:
        """Get the claude executable path."""
        if self._executable is None:
            self._executable = find_npm_executable(self.CLI_NAME) or self.CLI_NAME
        return self._executable

    @property
    def is_available(self) -> bool:
        """Check if claude CLI is available on the system."""
        if self._available is None:
            self._available = find_npm_executable(self.CLI_NAME) is not None
        return self._available

    async def check_auth(self) -> bool:
        """Check if Claude CLI is authenticated."""
        if not self.is_available:
            return False

        try:
            # Run 'claude --version' to check if authenticated
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (asyncio.TimeoutError, OSError) as e:
            logger.warning("Claude auth check failed: %s", e)
            return False

    def get_auth_command(self) -> str:
        """Get the command to authenticate Claude CLI."""
        return "claude login"

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
        Invoke Claude CLI with a prompt.

        Args:
            prompt: The prompt to send to Claude.
            continue_session: Whether to continue a previous session.
            session_id: ID of session to continue.
            planning_mode: Whether to run in planning mode.
            timeout_seconds: Override default timeout.
            working_dir: Working directory for the CLI.
            extra_args: Additional CLI arguments.

        Returns:
            CLIResult with the invocation result.
        """
        if not self.is_available:
            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr="Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            )

        # Validate and sanitize prompt
        validated_prompt = self.sanitizer.validate_prompt(prompt)

        # Build command arguments
        args = self._build_args(
            validated_prompt,
            continue_session=continue_session,
            session_id=session_id,
            planning_mode=planning_mode,
            extra_args=extra_args,
        )

        timeout = timeout_seconds or self.default_timeout
        start_time = time.monotonic()

        try:
            # SECURITY: Use create_subprocess_exec with argv list (NOT shell=True)
            process = await asyncio.create_subprocess_exec(
                self.executable,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            duration = time.monotonic() - start_time
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # Determine status
            status = self._determine_status(process.returncode, stderr)

            return CLIResult(
                cli_name=self.name,
                status=status,
                exit_code=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                metadata={
                    "planning_mode": planning_mode,
                    "continue_session": continue_session,
                    "session_id": session_id,
                },
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.warning(
                "Claude CLI timed out after %.1f seconds",
                duration,
            )
            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {timeout} seconds",
                duration_seconds=duration,
            )

        except OSError as e:
            logger.error("Claude CLI execution error: %s", e, exc_info=True)
            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    def _build_args(
        self,
        prompt: str,
        *,
        continue_session: bool,
        session_id: str | None,
        planning_mode: bool,
        extra_args: list[str] | None,
    ) -> list[str]:
        """Build CLI arguments."""
        args = ["-p", prompt]

        # Output format
        args.extend(["--output-format", "stream-json"])

        # Session handling
        if continue_session and session_id:
            args.extend(["--resume", session_id])

        # Planning mode (if supported)
        # Note: Claude CLI planning mode via --plan flag if available

        # Extra arguments
        if extra_args:
            args.extend(extra_args)

        return args

    def _determine_status(self, return_code: int | None, stderr: str) -> CLIStatus:
        """Determine CLI status from return code and stderr."""
        if return_code == 0:
            return CLIStatus.SUCCESS

        stderr_lower = stderr.lower()

        if "rate limit" in stderr_lower or "429" in stderr_lower:
            return CLIStatus.RATE_LIMITED

        if "auth" in stderr_lower or "login" in stderr_lower or "unauthorized" in stderr_lower:
            return CLIStatus.AUTH_ERROR

        return CLIStatus.ERROR

    def parse_stream_json(self, output: str) -> list[dict[str, Any]]:
        """
        Parse stream-json output from Claude CLI.

        Claude CLI outputs newline-delimited JSON objects.

        Args:
            output: Raw stdout from CLI.

        Returns:
            List of parsed JSON objects.
        """
        results = []
        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip non-JSON lines (progress messages, etc.)
                continue
        return results

    def extract_final_response(self, output: str) -> str:
        """
        Extract the final response text from stream-json output.

        Args:
            output: Raw stdout from CLI.

        Returns:
            Final response text.
        """
        parsed = self.parse_stream_json(output)

        # Look for the final message content
        for item in reversed(parsed):
            if "result" in item:
                return str(item["result"])
            if "content" in item:
                return str(item["content"])
            if "message" in item:
                return str(item["message"])

        # Fallback: return raw output
        return output
