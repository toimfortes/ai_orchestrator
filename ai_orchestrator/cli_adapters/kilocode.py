"""Kilocode CLI adapter for OpenRouter models."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ai_orchestrator.cli_adapters.base import (
    CLIAdapter,
    CLIResult,
    CLIStatus,
    find_npm_executable,
)
from ai_orchestrator.utils.sanitization import PromptSanitizer

logger = logging.getLogger(__name__)


class KilocodeAdapter(CLIAdapter):
    """
    Adapter for Kilocode CLI (OpenRouter integration).

    Uses browser-based OAuth authentication.
    Supports multiple models via OpenRouter, including free models.

    CLI Reference:
        kilocode --auto "task"              # Autonomous mode
        kilocode --auto --json "task"       # JSON output (requires --auto)
        kilocode --auto --timeout 300 "task"  # With timeout

    Free Models (via OpenRouter):
        mistralai/devstral-2512:free
        qwen/qwen3-coder:free
        deepseek/deepseek-r1-0528:free
        nvidia/nemotron-3-nano-30b-a3b:free
    """

    CLI_NAME = "kilocode"
    CONFIG_PATH = Path.home() / ".kilocode"

    # Default model tiers
    DEFAULT_MODEL = "mistralai/devstral-2512:free"  # Best free coding model

    def __init__(
        self,
        default_timeout: float = 900.0,  # 15 minutes
        model: str | None = None,
        sanitizer: PromptSanitizer | None = None,
    ) -> None:
        super().__init__(name=self.CLI_NAME, default_timeout=default_timeout)
        self.model = model or self.DEFAULT_MODEL
        self.sanitizer = sanitizer or PromptSanitizer()
        self._available: bool | None = None
        self._executable: str | None = None

    @property
    def executable(self) -> str:
        """Get the kilocode executable path."""
        if self._executable is None:
            self._executable = find_npm_executable(self.CLI_NAME) or self.CLI_NAME
        return self._executable

    @property
    def is_available(self) -> bool:
        """Check if kilocode CLI is available on the system."""
        if self._available is None:
            self._available = find_npm_executable(self.CLI_NAME) is not None
        return self._available

    async def check_auth(self) -> bool:
        """Check if Kilocode CLI is authenticated."""
        if not self.is_available:
            return False

        # Check if secrets.json exists (required for auto mode)
        secrets_path = self.CONFIG_PATH / "secrets.json"
        if not secrets_path.exists():
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=10)
            return process.returncode == 0
        except (asyncio.TimeoutError, OSError) as e:
            logger.warning("Kilocode auth check failed: %s", e)
            return False

    def get_auth_command(self) -> str:
        """Get the command to authenticate Kilocode CLI."""
        return "kilocode auth"

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
        Invoke Kilocode CLI with a prompt.

        Args:
            prompt: The prompt/task to send to Kilocode.
            continue_session: Not supported by Kilocode (ignored).
            session_id: Not supported by Kilocode (ignored).
            planning_mode: Not used.
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
                stderr="Kilocode CLI not found. Install from: https://kilo.ai",
            )

        # Validate and sanitize prompt
        validated_prompt = self.sanitizer.validate_prompt(prompt)

        # Build command arguments
        args = self._build_args(
            validated_prompt,
            timeout_seconds=timeout_seconds,
            extra_args=extra_args,
        )

        timeout = timeout_seconds or self.default_timeout
        start_time = time.monotonic()

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                self.executable,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout + 30,  # Add buffer for CLI's own timeout
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
                metadata={"model": self.model},
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.warning(
                "Kilocode CLI timed out after %.1f seconds - potential infinite loop",
                duration,
            )

            # Kill process to prevent infinite loops (known Kilocode issue)
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                    logger.info("Killed timed-out Kilocode process")
                except OSError as kill_error:
                    logger.debug("Process kill error (may already be dead): %s", kill_error)

            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.TIMEOUT,
                exit_code=124,  # Standard timeout exit code
                stdout="",
                stderr=f"Timeout after {timeout} seconds - process terminated",
                duration_seconds=duration,
            )

        except OSError as e:
            logger.error("Kilocode CLI execution error: %s", e, exc_info=True)
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
        timeout_seconds: float | None,
        extra_args: list[str] | None,
    ) -> list[str]:
        """Build CLI arguments."""
        # Autonomous mode is required for non-interactive use
        args = ["--auto"]

        # JSON output (requires --auto)
        args.append("--json")

        # Timeout (in seconds)
        if timeout_seconds:
            args.extend(["--timeout", str(int(timeout_seconds))])

        # Model selection (if supported)
        # Note: Model may be configured via config file or environment
        # args.extend(["--model", self.model])

        # The task/prompt comes last
        args.append(prompt)

        # Extra arguments
        if extra_args:
            # Insert before prompt
            args = args[:-1] + extra_args + [args[-1]]

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

        if "auth" in stderr_lower or "token" in stderr_lower:
            return CLIStatus.AUTH_ERROR

        if "connection" in stderr_lower or "refused" in stderr_lower:
            return CLIStatus.ERROR

        return CLIStatus.ERROR

    def set_model(self, model: str) -> None:
        """Change the model used by this adapter."""
        self.model = model
        logger.info("Kilocode model changed to: %s", model)
