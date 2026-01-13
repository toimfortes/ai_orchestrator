"""Robust JSON parsing with multiple fallback strategies and Pydantic validation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar, TYPE_CHECKING

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None  # type: ignore
    ValidationError = Exception  # type: ignore

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _truncate(text: str, max_len: int, marker: str = "[...truncated]") -> str:
    """Truncate text with marker if exceeds max_len."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(marker)] + marker


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
                "response_preview": _truncate(response, 500),
                "errors": errors,
            },
        )

        raise JSONParseError(
            message="Could not parse JSON from response",
            response_preview=_truncate(response, 200),
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
        """Extract JSON object or array using balanced bracket matching."""
        if expected_type == list:
            start_char, end_char = "[", "]"
        else:
            start_char, end_char = "{", "}"

        # Find first occurrence of start character
        start_idx = text.find(start_char)
        if start_idx == -1:
            return None

        # Find matching end bracket using bracket counting
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

        return None  # Unbalanced brackets

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

    def parse_with_schema(
        self,
        response: str,
        schema: type[BaseModel],  # type: ignore[valid-type]
        cli_hint: str | None = None,
    ) -> BaseModel:  # type: ignore[valid-type]
        """
        Parse JSON and validate against a Pydantic schema.

        This provides type-safe parsing with automatic validation and
        clear error messages when the LLM output doesn't match expectations.

        Args:
            response: Raw response string from CLI.
            schema: Pydantic model class to validate against.
            cli_hint: Optional hint about which CLI produced this output.

        Returns:
            Validated Pydantic model instance.

        Raises:
            JSONParseError: If parsing fails.
            ValidationError: If validation against schema fails.
            RuntimeError: If Pydantic is not installed.

        Example:
            class PlanResponse(BaseModel):
                steps: list[str]
                estimated_complexity: int

            result = parser.parse_with_schema(response, PlanResponse)
        """
        if not HAS_PYDANTIC:
            raise RuntimeError(
                "Pydantic is required for schema validation. "
                "Install with: pip install pydantic>=2.0"
            )

        # First, parse the JSON
        data = self.parse(response, expected_type=dict, cli_hint=cli_hint)

        # Then validate against schema
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            logger.warning(
                "JSON parsed but schema validation failed: %s",
                e.errors(),
            )
            raise

    def parse_with_schema_safe(
        self,
        response: str,
        schema: type[BaseModel],  # type: ignore[valid-type]
        default: T,
        cli_hint: str | None = None,
    ) -> BaseModel | T:  # type: ignore[valid-type]
        """
        Parse and validate JSON, returning default on any failure.

        Args:
            response: Raw response string.
            schema: Pydantic model class to validate against.
            default: Default value if parsing or validation fails.
            cli_hint: Optional hint about which CLI produced this output.

        Returns:
            Validated Pydantic model instance or default value.
        """
        try:
            return self.parse_with_schema(response, schema, cli_hint)
        except (JSONParseError, ValidationError, RuntimeError):
            return default


# Singleton instance
_parser = RobustJSONParser()


def parse_json(response: str, expected_type: type = dict) -> Any:
    """Parse JSON from response (convenience function)."""
    return _parser.parse(response, expected_type)


def parse_json_safe(response: str, default: T, expected_type: type = dict) -> Any | T:
    """Parse JSON from response, returning default on failure."""
    return _parser.parse_safe(response, default, expected_type)
