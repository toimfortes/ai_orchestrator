"""OpenAI Codex CLI adapter."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

from ai_orchestrator.cli_adapters.base import CLIAdapter, CLIResult, CLIStatus
from ai_orchestrator.utils.sanitization import PromptSanitizer

logger = logging.getLogger(__name__)


class CodexAdapter(CLIAdapter):
    """
    Adapter for OpenAI Codex CLI.

    Uses browser-based OAuth authentication (no API keys).
    Supports JSON output, session resumption, and full-auto mode.

    CLI Reference:
        codex exec "prompt"           # Non-interactive
        codex exec --json "prompt"    # JSON output
        codex exec --full-auto "prompt"  # Full automation
        codex exec resume ID "prompt"    # Resume session
    """

    CLI_NAME = "codex"
    CONFIG_PATH = Path.home() / ".codex"

    def __init__(
        self,
        default_timeout: float = 600.0,  # 10 minutes
        sanitizer: PromptSanitizer | None = None,
    ) -> None:
        super().__init__(name=self.CLI_NAME, default_timeout=default_timeout)
        self.sanitizer = sanitizer or PromptSanitizer()
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Check if codex CLI is available on the system."""
        if self._available is None:
            self._available = shutil.which("codex") is not None
        return self._available

    async def check_auth(self) -> bool:
        """Check if Codex CLI is authenticated."""
        if not self.is_available:
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                "codex",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (asyncio.TimeoutError, OSError) as e:
            logger.warning("Codex auth check failed: %s", e)
            return False

    def get_auth_command(self) -> str:
        """Get the command to authenticate Codex CLI."""
        return "codex login"

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
        Invoke Codex CLI with a prompt.

        Args:
            prompt: The prompt to send to Codex.
            continue_session: Whether to continue a previous session.
            session_id: ID of session to continue.
            planning_mode: Whether to run in planning mode (not used by Codex).
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
                stderr="Codex CLI not found. Install with: npm i -g @openai/codex",
            )

        # Validate and sanitize prompt
        validated_prompt = self.sanitizer.validate_prompt(prompt)

        # Build command arguments
        args = self._build_args(
            validated_prompt,
            continue_session=continue_session,
            session_id=session_id,
            extra_args=extra_args,
        )

        timeout = timeout_seconds or self.default_timeout
        start_time = time.monotonic()

        try:
            process = await asyncio.create_subprocess_exec(
                "codex",
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

            status = self._determine_status(process.returncode, stderr)

            return CLIResult(
                cli_name=self.name,
                status=status,
                exit_code=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                metadata={
                    "continue_session": continue_session,
                    "session_id": session_id,
                },
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.warning("Codex CLI timed out after %.1f seconds", duration)
            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {timeout} seconds",
                duration_seconds=duration,
            )

        except OSError as e:
            logger.error("Codex CLI execution error: %s", e)
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
        extra_args: list[str] | None,
    ) -> list[str]:
        """Build CLI arguments."""
        # Base command: codex exec
        if continue_session and session_id:
            # Resume session: codex exec resume SESSION_ID "prompt"
            args = ["exec", "resume", session_id, prompt]
        else:
            # New session: codex exec "prompt"
            args = ["exec", prompt]

        # Add JSON output
        args.insert(1, "--json")

        # Full automation (no interactive prompts)
        args.insert(1, "--full-auto")

        # Extra arguments
        if extra_args:
            args.extend(extra_args)

        return args

    def _determine_status(self, return_code: int | None, stderr: str) -> CLIStatus:
        """Determine CLI status from return code and stderr."""
        if return_code == 0:
            return CLIStatus.SUCCESS

        if return_code == 124:  # Timeout exit code
            return CLIStatus.TIMEOUT

        stderr_lower = stderr.lower()

        if "rate limit" in stderr_lower or "429" in stderr_lower:
            return CLIStatus.RATE_LIMITED

        if "auth" in stderr_lower or "login" in stderr_lower:
            return CLIStatus.AUTH_ERROR

        return CLIStatus.ERROR
