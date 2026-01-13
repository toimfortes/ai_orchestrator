"""Utility module for JSON parsing, sanitization, and concurrency."""

from ai_orchestrator.utils.json_parser import (
    RobustJSONParser,
    JSONParseError,
    parse_json,
    parse_json_safe,
)
from ai_orchestrator.utils.sanitization import PromptSanitizer


def truncate_with_marker(text: str, max_length: int, marker: str = "[...truncated]") -> str:
    """
    Truncate text and add marker if it exceeds max_length.

    Args:
        text: The text to truncate.
        max_length: Maximum length before truncation.
        marker: Marker to append when truncated.

    Returns:
        Original text if within limit, otherwise truncated with marker.
    """
    if len(text) <= max_length:
        return text
    # Reserve space for marker
    truncate_at = max_length - len(marker)
    return text[:truncate_at] + marker


__all__ = [
    "RobustJSONParser",
    "JSONParseError",
    "PromptSanitizer",
    "truncate_with_marker",
    "parse_json",
    "parse_json_safe",
]
