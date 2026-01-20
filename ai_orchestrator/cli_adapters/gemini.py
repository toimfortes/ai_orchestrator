"""Google Gemini CLI adapter.

Gemini vs Claude Prompting:
    Claude handles implicit reasoning well - "do a deep critique" triggers internal
    chain-of-thought. Gemini "favors directness over persuasion and logic over
    verbosity" and is less verbose by default. This adapter includes depth-enhancing
    prompt prefixes to match Claude's thoroughness.

    Key techniques applied:
    1. Explicit reasoning decomposition (chain-of-thought)
    2. Self-critique mechanism requests
    3. Verbosity level instructions
    4. Structured output formatting

    For persistent depth, configure ~/.gemini/GEMINI.md or project GEMINI.md with
    depth standards. See: https://geminicli.com/docs/cli/gemini-md/
"""

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


# Prompt prefixes to enhance Gemini response depth (research-backed)
DEPTH_ENHANCEMENT_PREFIX = """Before responding, follow these reasoning steps:
1. Parse the goal into distinct sub-tasks
2. Check if you have complete information for each sub-task
3. Create a structured outline of your response
4. Provide thorough, detailed analysis for each section
5. After drafting, self-critique: Did you answer the intent or just literal words?

Provide comprehensive, in-depth responses rather than concise summaries.
"""

PLANNING_MODE_PREFIX = """You are creating a detailed implementation plan. Be thorough and comprehensive.

Before providing your plan:
1. Identify all components that need to be created or modified
2. Consider security implications, testing strategy, and potential risks
3. Break down complex changes into ordered steps
4. Identify both obvious and non-obvious considerations

Provide a complete plan with all sections fully developed, not brief summaries.
"""


class GeminiAdapter(CLIAdapter):
    """
    Adapter for Google Gemini CLI with depth-enhancing prompts.

    Uses browser-based OAuth authentication (Google account).
    Supports JSON output, model selection, and yolo mode.

    Gemini is less verbose by default than Claude, so this adapter includes
    depth-enhancement prefixes to request thorough analysis.

    CLI Reference:
        gemini -p "prompt"                       # Non-interactive
        gemini -p "prompt" --output-format json  # JSON output
        gemini -p "prompt" --yolo                # Auto-approve all actions
        gemini -p "prompt" -m gemini-2.5-flash   # Specify model

    GEMINI.md Setup (Recommended for persistent depth):
        Create ~/.gemini/GEMINI.md or project-level GEMINI.md:
        ```markdown
        ## Response Depth Standards
        - Always provide thorough, in-depth analysis
        - Break complex problems into sub-tasks before answering
        - Explain your reasoning step-by-step
        - Identify both obvious and non-obvious considerations
        - Self-critique your response before finalizing
        ```
    """

    CLI_NAME = "gemini"
    CONFIG_PATH = Path.home() / ".gemini"

    def __init__(
        self,
        default_timeout: float = 600.0,  # 10 minutes
        model: str | None = None,
        sanitizer: PromptSanitizer | None = None,
        enhance_depth: bool = True,  # Add depth-enhancement prefixes
    ) -> None:
        """
        Initialize Gemini adapter.

        Args:
            default_timeout: Default timeout in seconds.
            model: Model to use (e.g., "gemini-2.5-pro", "gemini-2.5-flash").
            sanitizer: Prompt sanitizer instance.
            enhance_depth: Whether to add depth-enhancement prefixes to prompts.
                          Gemini is less verbose by default than Claude, so this
                          adds explicit instructions for thorough analysis.
                          Set to False if using GEMINI.md for persistent depth config.
        """
        super().__init__(name=self.CLI_NAME, default_timeout=default_timeout)
        self.model = model  # e.g., "gemini-2.5-flash", "gemini-2.5-pro"
        self.sanitizer = sanitizer or PromptSanitizer()
        self.enhance_depth = enhance_depth
        self._available: bool | None = None
        self._executable: str | None = None

    @property
    def executable(self) -> str:
        """Get the gemini executable path."""
        if self._executable is None:
            self._executable = find_npm_executable(self.CLI_NAME) or self.CLI_NAME
        return self._executable

    @property
    def is_available(self) -> bool:
        """Check if gemini CLI is available on the system."""
        if self._available is None:
            self._available = find_npm_executable(self.CLI_NAME) is not None
        return self._available

    async def check_auth(self) -> bool:
        """Check if Gemini CLI is authenticated."""
        if not self.is_available:
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
            logger.warning("Gemini auth check failed: %s", e)
            return False

    def get_auth_command(self) -> str:
        """Get the command to authenticate Gemini CLI."""
        return "gemini auth"

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
        Invoke Gemini CLI with a prompt.

        Args:
            prompt: The prompt to send to Gemini.
            continue_session: Not supported by Gemini CLI (ignored).
            session_id: Not supported by Gemini CLI (ignored).
            planning_mode: Whether to run in planning mode (not used).
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
                stderr="Gemini CLI not found. Install with: npm install -g @google/gemini-cli",
            )

        # Validate and sanitize prompt
        validated_prompt = self.sanitizer.validate_prompt(prompt)

        # Apply depth enhancement if enabled
        # Gemini is less verbose by default than Claude, so we add explicit
        # instructions for thorough analysis
        if self.enhance_depth:
            if planning_mode:
                validated_prompt = f"{PLANNING_MODE_PREFIX}\n\n{validated_prompt}"
            else:
                validated_prompt = f"{DEPTH_ENHANCEMENT_PREFIX}\n\n{validated_prompt}"
            logger.debug("Applied depth enhancement prefix to Gemini prompt")

        # Build command arguments
        args = self._build_args(validated_prompt, extra_args=extra_args)

        timeout = timeout_seconds or self.default_timeout
        start_time = time.monotonic()

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
                metadata={"model": self.model},
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.warning("Gemini CLI timed out after %.1f seconds", duration)
            return CLIResult(
                cli_name=self.name,
                status=CLIStatus.TIMEOUT,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {timeout} seconds",
                duration_seconds=duration,
            )

        except OSError as e:
            logger.error("Gemini CLI execution error: %s", e, exc_info=True)
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
        extra_args: list[str] | None,
    ) -> list[str]:
        """Build CLI arguments."""
        args = ["-p", prompt]

        # Output format
        args.extend(["--output-format", "json"])

        # Auto-approve all actions (yolo mode)
        args.append("--yolo")

        # Model selection
        if self.model:
            args.extend(["-m", self.model])

        # Extra arguments
        if extra_args:
            args.extend(extra_args)

        return args

    def _determine_status(self, return_code: int | None, stderr: str) -> CLIStatus:
        """Determine CLI status from return code and stderr."""
        if return_code == 0:
            return CLIStatus.SUCCESS

        stderr_lower = stderr.lower()

        if "rate limit" in stderr_lower or "429" in stderr_lower or "quota" in stderr_lower:
            return CLIStatus.RATE_LIMITED

        if "auth" in stderr_lower or "login" in stderr_lower or "credentials" in stderr_lower:
            return CLIStatus.AUTH_ERROR

        return CLIStatus.ERROR
