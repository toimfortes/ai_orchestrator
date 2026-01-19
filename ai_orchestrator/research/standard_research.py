"""Standard (middle tier) research with 2-provider sequential comparison."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

from ai_orchestrator.research.comparison import (
    compare_findings,
    extract_findings,
    check_no_issues_response,
)
from ai_orchestrator.research.models import (
    ComparisonResult,
    Finding,
    ProviderResult,
    ResearchFocus,
    StandardResearchResult,
)
from ai_orchestrator.research.prompts import build_research_prompt

logger = logging.getLogger(__name__)


class ResearchProvider:
    """
    Base class for research providers.

    Wraps deep research APIs (Gemini, OpenAI, etc.) with consistent interface.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    async def research(self, prompt: str) -> ProviderResult:
        """
        Execute deep research with the given prompt.

        Args:
            prompt: The research prompt.

        Returns:
            ProviderResult with findings and raw output.
        """
        raise NotImplementedError("Subclasses must implement research()")


class GeminiResearchProvider(ResearchProvider):
    """Gemini deep research provider using MCP."""

    def __init__(self) -> None:
        super().__init__("gemini")
        self._mcp_available = True  # Assume available, fail gracefully

    async def research(self, prompt: str) -> ProviderResult:
        """Execute Gemini deep research via MCP."""
        start_time = time.monotonic()

        try:
            # Import MCP tools - these are injected at runtime
            from ai_orchestrator.research._mcp_bridge import (
                start_deep_research,
                check_deep_research,
                get_research_result,
            )

            # Start research job
            job_result = await start_deep_research(prompt, provider="gemini")
            job_id = job_result.get("job_id")

            if not job_id:
                return ProviderResult(
                    provider=self.name,
                    raw_output="",
                    success=False,
                    error="Failed to start Gemini research job",
                    duration_seconds=time.monotonic() - start_time,
                )

            logger.info("Started Gemini research job: %s", job_id)

            # Poll for completion (max 15 minutes)
            max_wait = 900  # 15 minutes
            poll_interval = 10  # Check every 10 seconds
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status = await check_deep_research(job_id)
                state = status.get("status", "unknown")

                if state == "completed":
                    # Get full result
                    result = await get_research_result(job_id)
                    raw_output = result.get("result", "")

                    # Extract findings
                    if check_no_issues_response(raw_output):
                        findings = []
                    else:
                        findings = extract_findings(raw_output)

                    return ProviderResult(
                        provider=self.name,
                        raw_output=raw_output,
                        findings=findings,
                        summary=self._extract_summary(raw_output),
                        duration_seconds=time.monotonic() - start_time,
                        success=True,
                    )

                elif state == "failed":
                    return ProviderResult(
                        provider=self.name,
                        raw_output="",
                        success=False,
                        error=status.get("error", "Research job failed"),
                        duration_seconds=time.monotonic() - start_time,
                    )

                logger.debug("Gemini research status: %s (elapsed: %ds)", state, elapsed)

            # Timeout
            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error=f"Research job timed out after {max_wait}s",
                duration_seconds=time.monotonic() - start_time,
            )

        except ImportError:
            logger.warning("MCP bridge not available for Gemini research")
            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error="MCP bridge not available",
                duration_seconds=time.monotonic() - start_time,
            )

        except Exception as e:
            logger.error("Gemini research failed: %s", e, exc_info=True)
            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error=str(e),
                duration_seconds=time.monotonic() - start_time,
            )

    def _extract_summary(self, raw_output: str) -> str:
        """Extract summary section from output."""
        import re

        match = re.search(r"###?\s*Summary\s*\n(.+?)(?:\n###?|\Z)", raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


class OpenAIResearchProvider(ResearchProvider):
    """OpenAI deep research provider using MCP."""

    def __init__(self) -> None:
        super().__init__("openai")

    async def research(self, prompt: str) -> ProviderResult:
        """Execute OpenAI deep research via MCP."""
        start_time = time.monotonic()

        try:
            from ai_orchestrator.research._mcp_bridge import (
                start_deep_research,
                check_deep_research,
                get_research_result,
            )

            # Start research job
            job_result = await start_deep_research(prompt, provider="openai")
            job_id = job_result.get("job_id")

            if not job_id:
                return ProviderResult(
                    provider=self.name,
                    raw_output="",
                    success=False,
                    error="Failed to start OpenAI research job",
                    duration_seconds=time.monotonic() - start_time,
                )

            logger.info("Started OpenAI research job: %s", job_id)

            # Poll for completion
            max_wait = 900
            poll_interval = 10
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                status = await check_deep_research(job_id)
                state = status.get("status", "unknown")

                if state == "completed":
                    result = await get_research_result(job_id)
                    raw_output = result.get("result", "")

                    if check_no_issues_response(raw_output):
                        findings = []
                    else:
                        findings = extract_findings(raw_output)

                    return ProviderResult(
                        provider=self.name,
                        raw_output=raw_output,
                        findings=findings,
                        summary=self._extract_summary(raw_output),
                        duration_seconds=time.monotonic() - start_time,
                        success=True,
                    )

                elif state == "failed":
                    return ProviderResult(
                        provider=self.name,
                        raw_output="",
                        success=False,
                        error=status.get("error", "Research job failed"),
                        duration_seconds=time.monotonic() - start_time,
                    )

            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error=f"Research job timed out after {max_wait}s",
                duration_seconds=time.monotonic() - start_time,
            )

        except ImportError:
            logger.warning("MCP bridge not available for OpenAI research")
            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error="MCP bridge not available",
                duration_seconds=time.monotonic() - start_time,
            )

        except Exception as e:
            logger.error("OpenAI research failed: %s", e, exc_info=True)
            return ProviderResult(
                provider=self.name,
                raw_output="",
                success=False,
                error=str(e),
                duration_seconds=time.monotonic() - start_time,
            )

    def _extract_summary(self, raw_output: str) -> str:
        """Extract summary section from output."""
        import re

        match = re.search(r"###?\s*Summary\s*\n(.+?)(?:\n###?|\Z)", raw_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""


# Provider registry - typed as Callable since subclasses have parameterless __init__
PROVIDERS: dict[str, Callable[[], ResearchProvider]] = {
    "gemini": GeminiResearchProvider,
    "openai": OpenAIResearchProvider,
}


class StandardResearch:
    """
    Standard (middle tier) research with 2-provider sequential comparison.

    Flow:
    1. Build focused prompt from code context and question
    2. Run first provider (e.g., Gemini)
    3. Run second provider (e.g., OpenAI) - sequential to avoid rate limits
    4. Extract findings from both outputs
    5. Compare findings to identify consensus and disagreements
    6. Return structured result with confidence score

    Why 2 providers:
    - Catches single-model blind spots
    - Agreement = high confidence
    - Disagreement = needs human review

    Why sequential (not parallel):
    - Avoids rate limit issues (especially Gemini)
    - Easier to debug
    - Total time ~8-12 min is acceptable for this use case
    """

    DEFAULT_PROVIDERS = ["gemini", "openai"]

    def __init__(
        self,
        providers: list[str] | None = None,
    ) -> None:
        """
        Initialize StandardResearch.

        Args:
            providers: List of provider names to use (default: gemini, openai).
        """
        provider_names = providers or self.DEFAULT_PROVIDERS

        if len(provider_names) < 2:
            raise ValueError("StandardResearch requires at least 2 providers")

        self.provider_instances: list[ResearchProvider] = []
        for name in provider_names[:2]:  # Only use first 2
            if name not in PROVIDERS:
                raise ValueError(f"Unknown provider: {name}")
            self.provider_instances.append(PROVIDERS[name]())

    async def run(
        self,
        code_context: str,
        question: str,
        focus: ResearchFocus = ResearchFocus.GENERAL,
    ) -> StandardResearchResult:
        """
        Run standard research with 2 providers.

        Args:
            code_context: Code to analyze (formatted with file paths).
            question: Specific question to answer.
            focus: Focus area for the research.

        Returns:
            StandardResearchResult with findings and comparison.
        """
        start_time = time.monotonic()

        # Build prompt
        prompt = build_research_prompt(code_context, question, focus)

        logger.info(
            "Starting standard research: focus=%s, providers=%s",
            focus.value,
            [p.name for p in self.provider_instances],
        )

        # Run provider 1
        logger.info("Running provider 1: %s", self.provider_instances[0].name)
        result_1 = await self.provider_instances[0].research(prompt)
        logger.info(
            "Provider 1 complete: success=%s, findings=%d, duration=%.1fs",
            result_1.success,
            len(result_1.findings),
            result_1.duration_seconds,
        )

        # Run provider 2 (sequential)
        logger.info("Running provider 2: %s", self.provider_instances[1].name)
        result_2 = await self.provider_instances[1].research(prompt)
        logger.info(
            "Provider 2 complete: success=%s, findings=%d, duration=%.1fs",
            result_2.success,
            len(result_2.findings),
            result_2.duration_seconds,
        )

        # Compare findings
        if result_1.success and result_2.success:
            comparison = compare_findings(
                result_1.findings,
                result_2.findings,
                self.provider_instances[0].name,
                self.provider_instances[1].name,
            )
        elif result_1.success:
            # Only provider 1 succeeded - all findings are "disagreements"
            for f in result_1.findings:
                f.source = f"{self.provider_instances[0].name}_only"
            comparison = ComparisonResult(
                agreed_findings=[],
                disagreed_findings=result_1.findings,
                confidence_score=0.5,  # Lower confidence with single provider
            )
        elif result_2.success:
            # Only provider 2 succeeded
            for f in result_2.findings:
                f.source = f"{self.provider_instances[1].name}_only"
            comparison = ComparisonResult(
                agreed_findings=[],
                disagreed_findings=result_2.findings,
                confidence_score=0.5,
            )
        else:
            # Both failed
            comparison = ComparisonResult(
                agreed_findings=[],
                disagreed_findings=[],
                confidence_score=0.0,
            )

        total_duration = time.monotonic() - start_time

        logger.info(
            "Standard research complete: agreed=%d, disagreed=%d, confidence=%.0f%%, duration=%.1fs",
            len(comparison.agreed_findings),
            len(comparison.disagreed_findings),
            comparison.confidence_score * 100,
            total_duration,
        )

        return StandardResearchResult(
            question=question,
            focus=focus,
            provider_1=result_1,
            provider_2=result_2,
            comparison=comparison,
            total_duration_seconds=total_duration,
        )


async def quick_research(
    code_context: str,
    question: str,
    focus: ResearchFocus = ResearchFocus.GENERAL,
    provider: str = "gemini",
) -> ProviderResult:
    """
    Quick (single provider) research for fast results.

    Args:
        code_context: Code to analyze.
        question: Question to answer.
        focus: Focus area.
        provider: Provider to use (default: gemini).

    Returns:
        ProviderResult with findings.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    prompt = build_research_prompt(code_context, question, focus)
    provider_instance = PROVIDERS[provider]()

    logger.info("Starting quick research: provider=%s, focus=%s", provider, focus.value)

    result = await provider_instance.research(prompt)

    logger.info(
        "Quick research complete: success=%s, findings=%d, duration=%.1fs",
        result.success,
        len(result.findings),
        result.duration_seconds,
    )

    return result


def format_code_context(files: dict[str, str], max_lines: int = 2000) -> str:
    """
    Format code files into a code context string for prompts.

    Args:
        files: Dict mapping file paths to content.
        max_lines: Maximum total lines to include.

    Returns:
        Formatted code context string.
    """
    sections = []
    total_lines = 0

    for path, content in files.items():
        lines = content.split("\n")
        if total_lines + len(lines) > max_lines:
            # Truncate this file
            remaining = max_lines - total_lines
            if remaining > 50:  # Only include if meaningful
                lines = lines[:remaining]
                sections.append(f"### {path} (truncated)\n```python\n{chr(10).join(lines)}\n```")
            break

        sections.append(f"### {path}\n```python\n{content}\n```")
        total_lines += len(lines)

    return "\n\n".join(sections)
