"""Prompt sanitization and validation utilities."""

from __future__ import annotations

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the maximum length."""

    pass


class PathTraversalError(Exception):
    """Raised when a path attempts directory traversal."""

    pass


class PromptSanitizer:
    """
    Validate and sanitize prompts before CLI invocation.

    SECURITY NOTE: We use create_subprocess_exec() with argv list,
    NOT shell=True. This means shell metacharacters (|, ;, &&, etc.)
    are passed as literal strings, not interpreted. No injection possible.

    shlex.quote() is NOT needed because we're not invoking a shell.
    """

    MAX_PROMPT_LENGTH = 100_000  # 100KB

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root.resolve() if project_root else None

    def validate_prompt(self, prompt: str) -> str:
        """
        Validate prompt for CLI argument passing.

        Args:
            prompt: The raw prompt string.

        Returns:
            Validated prompt string.

        Raises:
            PromptTooLongError: If prompt exceeds maximum length.
        """
        # 1. Remove null bytes (could truncate strings in C-based CLIs)
        validated = prompt.replace("\x00", "")

        # 2. Limit length (prevent memory exhaustion)
        if len(validated) > self.MAX_PROMPT_LENGTH:
            raise PromptTooLongError(
                f"Prompt exceeds {self.MAX_PROMPT_LENGTH} characters "
                f"(got {len(validated)})"
            )

        # 3. Log warning for suspicious patterns (for monitoring, not blocking)
        self._log_suspicious_patterns(validated)

        return validated

    def sanitize_file_path(self, path: str) -> Path:
        """
        Sanitize file paths to prevent directory traversal.

        Args:
            path: The raw path string.

        Returns:
            Resolved Path object.

        Raises:
            PathTraversalError: If path escapes project root.
        """
        if self.project_root is None:
            # No project root set, just resolve
            return Path(path).resolve()

        resolved = Path(path).resolve()

        # Prevent escaping project directory
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            raise PathTraversalError(
                f"Path escapes project root: {path} -> {resolved}"
            )

        return resolved

    def _log_suspicious_patterns(self, text: str) -> None:
        """Log warnings for suspicious patterns (for monitoring)."""
        suspicious = [
            ("$(", "command substitution"),
            ("`", "backtick command"),
            ("&&", "command chaining"),
            ("||", "command chaining"),
            ("|", "pipe"),
            (";", "command separator"),
            (">", "redirect"),
            ("<", "redirect"),
        ]

        for pattern, description in suspicious:
            if pattern in text:
                logger.debug(
                    "Prompt contains '%s' (%s) - safe with exec mode",
                    pattern,
                    description,
                )
