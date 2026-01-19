"""Bridge to MCP deep research tools.

This module provides async wrappers for the MCP deep research tools.
The actual MCP tools are injected at runtime when running within an
MCP-enabled environment (like Claude Code).

For standalone usage, this module provides mock implementations
that can be replaced with direct API calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if we're in an MCP environment
_MCP_AVAILABLE = os.environ.get("MCP_DEEP_RESEARCH_AVAILABLE", "false").lower() == "true"


async def start_deep_research(
    query: str,
    provider: str = "gemini",
) -> dict[str, Any]:
    """
    Start a deep research job.

    Args:
        query: Research query/prompt.
        provider: Provider to use ("gemini" or "openai").

    Returns:
        Dict with job_id and status.
    """
    if _MCP_AVAILABLE:
        # In MCP environment, tools are called directly
        # This would be handled by the MCP runtime
        raise NotImplementedError(
            "Direct MCP calls should be handled by the MCP runtime. "
            "Use the CLI wrapper for standalone execution."
        )

    # Standalone mode: Use CLI wrapper
    return await _cli_start_research(query, provider)


async def check_deep_research(
    job_id: str,
    include_result: bool = False,
) -> dict[str, Any]:
    """
    Check status of a deep research job.

    Args:
        job_id: The job ID from start_deep_research.
        include_result: Whether to include full result inline.

    Returns:
        Dict with status and optionally result.
    """
    if _MCP_AVAILABLE:
        raise NotImplementedError("Use MCP runtime for direct calls")

    return await _cli_check_research(job_id, include_result)


async def get_research_result(
    job_id: str,
    format: str = "text",
) -> dict[str, Any]:
    """
    Get the result of a completed research job.

    Args:
        job_id: The job ID.
        format: Output format ("text" or "json").

    Returns:
        Dict with result and metadata.
    """
    if _MCP_AVAILABLE:
        raise NotImplementedError("Use MCP runtime for direct calls")

    return await _cli_get_result(job_id, format)


# CLI-based implementation for standalone usage
# These call the deep-research MCP server via command line

async def _cli_start_research(query: str, provider: str) -> dict[str, Any]:
    """Start research via CLI."""
    # This would call the deep-research server
    # For now, return a mock response indicating the tool isn't available
    logger.warning(
        "Deep research CLI not configured. "
        "Set MCP_DEEP_RESEARCH_AVAILABLE=true in MCP environment."
    )
    return {
        "status": "error",
        "error": "Deep research not available in standalone mode",
        "job_id": None,
    }


async def _cli_check_research(job_id: str, include_result: bool) -> dict[str, Any]:
    """Check research status via CLI."""
    logger.warning("Deep research CLI not configured")
    return {
        "status": "error",
        "error": "Deep research not available in standalone mode",
    }


async def _cli_get_result(job_id: str, format: str) -> dict[str, Any]:
    """Get research result via CLI."""
    logger.warning("Deep research CLI not configured")
    return {
        "status": "error",
        "error": "Deep research not available in standalone mode",
        "result": "",
    }


# Alternative: Direct API integration (future enhancement)
# This would allow running without MCP by calling APIs directly

class DirectAPIBridge:
    """
    Direct API bridge for deep research without MCP.

    This is a placeholder for future implementation that would
    call Gemini/OpenAI APIs directly for deep research.
    """

    def __init__(self) -> None:
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

    async def research_gemini(self, query: str) -> dict[str, Any]:
        """Run Gemini deep research via direct API."""
        if not self.gemini_api_key:
            return {
                "status": "error",
                "error": "GEMINI_API_KEY not set",
            }

        # TODO: Implement direct Gemini API call
        # This would use the Gemini API's research capabilities
        raise NotImplementedError("Direct Gemini API not yet implemented")

    async def research_openai(self, query: str) -> dict[str, Any]:
        """Run OpenAI deep research via direct API."""
        if not self.openai_api_key:
            return {
                "status": "error",
                "error": "OPENAI_API_KEY not set",
            }

        # TODO: Implement direct OpenAI API call
        raise NotImplementedError("Direct OpenAI API not yet implemented")
