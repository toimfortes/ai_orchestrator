"""Robust JSON parsing with multiple fallback strategies."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JSONParseError(Exception):
    """Raised when all JSON parsing strategies fail."""

    def __init__(
        self,
        message: str,
        response_preview: str = "",
        strategies_tried: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.response_preview = response_preview
        self.strategies_tried = strategies_tried or []


class RobustJSONParser:
    """
    Parse JSON from LLM output with multiple fallback strategies.

    Strategy order:
    1. Direct parse (fastest)
    2. Strip markdown code blocks
    3. Extract JSON object/array with regex
    4. Use json_repair library
    5. Return error with details
    """

    # Per-CLI parser hints
    CLI_PARSER_CONFIG: dict[str, dict[str, Any]] = {
        "claude": {
            "common_wrappers": ["```json", "Here's the JSON:"],
            "typical_issues": ["trailing_comma", "single_quotes"],
        },
        "codex": {
            "common_wrappers": ["```json"],
            "typical_issues": ["incomplete_output"],
        },
        "gemini": {
            "common_wrappers": ["```json", "```"],
            "typical_issues": ["markdown_in_strings"],
        },
        "kilocode": {
            "common_wrappers": ["```json"],
            "typical_issues": ["varies_by_model"],
        },
    }

    def parse(
        self,
        response: str,
        expected_type: type = dict,
        cli_hint: str | None = None,
    ) -> Any:
        """
        Parse JSON with ordered fallback strategies.

        Args:
            response: Raw response string from CLI.
            expected_type: Expected type (dict or list).
            cli_hint: Optional hint about which CLI produced this output.

        Returns:
            Parsed JSON data.

        Raises:
            JSONParseError: If all strategies fail.
        """
        errors: list[str] = []

        # Strategy 1: Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            errors.append(f"Direct parse failed: {e}")

        # Strategy 2: Strip markdown code blocks
        stripped = self._strip_markdown(response)
        if stripped != response:
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as e:
                errors.append(f"Markdown strip failed: {e}")

        # Strategy 3: Regex extraction
        extracted = self._extract_json_regex(response, expected_type)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError as e:
                errors.append(f"Regex extraction failed: {e}")

        # Strategy 4: json_repair library
        if HAS_JSON_REPAIR:
            try:
                repaired = repair_json(response)
                return json.loads(repaired)
            except Exception as e:
                errors.append(f"json_repair failed: {e}")
        else:
            errors.append("json_repair not available")

        # All strategies failed
        logger.error(
            "All JSON parse strategies failed",
            extra={
                "response_preview": response[:500],
                "errors": errors,
            },
        )

        raise JSONParseError(
            message="Could not parse JSON from response",
            response_preview=response[:200],
            strategies_tried=errors,
        )

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown code block wrappers."""
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return text

    def _extract_json_regex(self, text: str, expected_type: type) -> str | None:
        """Extract JSON object or array using regex."""
        if expected_type == list:
            # Look for array
            match = re.search(r"\[[\s\S]*\]", text)
        else:
            # Look for object (greedy, outermost braces)
            match = re.search(r"\{[\s\S]*\}", text)

        return match.group(0) if match else None

    def parse_safe(
        self,
        response: str,
        default: T,
        expected_type: type = dict,
    ) -> Any | T:
        """
        Parse JSON, returning default on failure.

        Args:
            response: Raw response string.
            default: Default value if parsing fails.
            expected_type: Expected type (dict or list).

        Returns:
            Parsed JSON or default value.
        """
        try:
            return self.parse(response, expected_type)
        except JSONParseError:
            return default


# Singleton instance
_parser = RobustJSONParser()


def parse_json(response: str, expected_type: type = dict) -> Any:
    """Parse JSON from response (convenience function)."""
    return _parser.parse(response, expected_type)


def parse_json_safe(response: str, default: T, expected_type: type = dict) -> Any | T:
    """Parse JSON from response, returning default on failure."""
    return _parser.parse_safe(response, default, expected_type)
