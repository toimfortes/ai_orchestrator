#!/usr/bin/env python
"""Script to invoke CLI adapters for code critique."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_orchestrator.cli_adapters.codex import CodexAdapter
from ai_orchestrator.cli_adapters.gemini import GeminiAdapter


CRITIQUE_PROMPT = """You are a senior software architect reviewing an AI Orchestrator implementation.

Review the following key implementation files and provide a thorough critique:

1. **orchestrator.py** - Main orchestration engine with:
   - Concurrent multi-CLI execution via asyncio.gather()
   - Custom CircuitBreaker class (closed/open/half-open states)
   - Semaphore-limited concurrent invocations
   - Fallback chains for graceful degradation
   - FeedbackClassifier integration for review classification

2. **feedback_classifier.py** - Keyword-based feedback classification:
   - Severity levels: CRITICAL/HIGH/MEDIUM/LOW
   - Category detection via keyword sets
   - File/line extraction via regex patterns
   - is_blocker binary flag

3. **post_checks.py** - 5-gate verification system:
   - Static analysis (ruff, mypy)
   - Unit tests (pytest)
   - Build verification
   - Security scan (bandit)
   - Manual smoke test (optional)

**Critique Focus Areas:**
1. Circuit breaker implementation - Is the custom implementation robust enough vs using pybreaker/tenacity?
2. Feedback classification - Keyword matching vs LLM-based classification tradeoffs
3. Concurrent execution - Is asyncio.gather with semaphore the right pattern?
4. Post-checks gates - Are there critical gates missing?
5. Error handling - Retry/backoff strategies
6. Context management - How should state be passed between phases?

**Provide:**
- CRITICAL issues (must fix)
- HIGH severity issues (should fix)
- MEDIUM issues (nice to have)
- Specific code/design recommendations

Be thorough and specific. Reference industry patterns (Microsoft AI Agents, LangGraph, etc).
"""


async def invoke_codex() -> str:
    """Invoke Codex CLI for critique."""
    print("Invoking Codex CLI...")
    adapter = CodexAdapter(default_timeout=300.0)

    if not adapter.is_available:
        return "ERROR: Codex CLI not available"

    result = await adapter.invoke(
        CRITIQUE_PROMPT,
        timeout_seconds=300.0,
        working_dir=str(Path(__file__).parent.parent),
    )

    print(f"Codex result: status={result.status.value}, exit_code={result.exit_code}")

    if result.status.value == "success":
        return result.stdout
    else:
        return f"ERROR: {result.stderr or result.status.value}"


async def invoke_gemini() -> str:
    """Invoke Gemini CLI for critique."""
    print("Invoking Gemini CLI...")
    adapter = GeminiAdapter(default_timeout=300.0, enhance_depth=True)

    if not adapter.is_available:
        return "ERROR: Gemini CLI not available"

    result = await adapter.invoke(
        CRITIQUE_PROMPT,
        planning_mode=False,
        timeout_seconds=300.0,
        working_dir=str(Path(__file__).parent.parent),
    )

    print(f"Gemini result: status={result.status.value}, exit_code={result.exit_code}")

    if result.status.value == "success":
        return result.stdout
    else:
        return f"ERROR: {result.stderr or result.status.value}"


async def main():
    """Run both critiques."""
    print("=" * 60)
    print("AI ORCHESTRATOR CRITIQUE VIA CLI ADAPTERS")
    print("=" * 60)

    # Run both in parallel
    codex_task = asyncio.create_task(invoke_codex())
    gemini_task = asyncio.create_task(invoke_gemini())

    codex_result, gemini_result = await asyncio.gather(codex_task, gemini_task)

    print("\n" + "=" * 60)
    print("CODEX CRITIQUE")
    print("=" * 60)
    print(codex_result[:5000] if len(codex_result) > 5000 else codex_result)

    print("\n" + "=" * 60)
    print("GEMINI CRITIQUE")
    print("=" * 60)
    print(gemini_result[:5000] if len(gemini_result) > 5000 else gemini_result)


if __name__ == "__main__":
    asyncio.run(main())
