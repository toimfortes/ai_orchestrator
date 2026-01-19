"""Prompt templates for research tasks."""

from __future__ import annotations

from ai_orchestrator.research.models import ResearchFocus


# Base template - kept simple per critique recommendations
BASE_RESEARCH_PROMPT = """## Role
Senior {focus_description} reviewing Python code.

## Code
{code_context}

## Question
{question}

## Instructions
1. Identify specific issues (not theoretical concerns)
2. For each issue provide:
   - Severity: CRITICAL, HIGH, MEDIUM, or LOW
   - Location: file.py:line_number (be specific)
   - Issue: Clear description of the problem
   - Evidence: Code snippet showing the issue
   - Fix: Concrete suggestion to resolve it
3. If no real issues found, say "No significant issues found"
4. Focus on {focus_area} concerns

## Output Format
Use this markdown structure:

### Findings

- **[SEVERITY]** file.py:123 - Brief issue title
  - Issue: Full description of the problem
  - Evidence: `relevant code snippet`
  - Fix: Specific fix suggestion

### Summary
1-2 sentence overall assessment.
"""

# Focus-specific descriptions
FOCUS_DESCRIPTIONS: dict[ResearchFocus, dict[str, str]] = {
    ResearchFocus.GENERAL: {
        "focus_description": "software engineer",
        "focus_area": "correctness, reliability, and maintainability",
    },
    ResearchFocus.SECURITY: {
        "focus_description": "security engineer specializing in OWASP Top 10, "
        "command injection, and async security patterns",
        "focus_area": "security vulnerabilities including injection attacks, "
        "authentication issues, data exposure, and unsafe subprocess usage",
    },
    ResearchFocus.PERFORMANCE: {
        "focus_description": "performance engineer specializing in async Python "
        "and distributed systems optimization",
        "focus_area": "performance bottlenecks, resource leaks, inefficient patterns, "
        "and scalability issues",
    },
    ResearchFocus.ARCHITECTURE: {
        "focus_description": "software architect specializing in distributed systems, "
        "async patterns, and clean architecture",
        "focus_area": "architectural issues including coupling, SOLID violations, "
        "error handling patterns, and extensibility concerns",
    },
    ResearchFocus.RELIABILITY: {
        "focus_description": "site reliability engineer specializing in "
        "fault tolerance and recovery patterns",
        "focus_area": "reliability issues including race conditions, error recovery, "
        "circuit breaker patterns, and state management",
    },
}


def build_research_prompt(
    code_context: str,
    question: str,
    focus: ResearchFocus = ResearchFocus.GENERAL,
) -> str:
    """
    Build a research prompt for the given focus area.

    Args:
        code_context: The code to analyze (formatted with file paths).
        question: The specific question to answer.
        focus: The focus area for the research.

    Returns:
        Formatted prompt string.
    """
    focus_config = FOCUS_DESCRIPTIONS.get(focus, FOCUS_DESCRIPTIONS[ResearchFocus.GENERAL])

    return BASE_RESEARCH_PROMPT.format(
        focus_description=focus_config["focus_description"],
        focus_area=focus_config["focus_area"],
        code_context=code_context,
        question=question,
    )


# Specific question templates for common research tasks
QUESTION_TEMPLATES = {
    "architecture_review": (
        "Review the architecture of this code. Focus on:\n"
        "1. Error handling and recovery patterns\n"
        "2. Concurrency and race conditions\n"
        "3. State management and persistence\n"
        "4. Extensibility and modularity\n"
        "5. Integration points and failure modes"
    ),
    "security_review": (
        "Perform a security review of this code. Check for:\n"
        "1. Injection vulnerabilities (command, prompt, path)\n"
        "2. Authentication and authorization issues\n"
        "3. Sensitive data exposure in logs or errors\n"
        "4. Unsafe subprocess or file operations\n"
        "5. Resource exhaustion or DoS vectors"
    ),
    "reliability_review": (
        "Review this code for reliability issues. Focus on:\n"
        "1. Circuit breaker and retry logic\n"
        "2. Timeout handling and cancellation\n"
        "3. State persistence and recovery\n"
        "4. Graceful degradation patterns\n"
        "5. Error propagation and logging"
    ),
    "performance_review": (
        "Review this code for performance issues. Check for:\n"
        "1. Blocking operations in async code\n"
        "2. Unnecessary serialization or copying\n"
        "3. N+1 patterns or redundant operations\n"
        "4. Resource leaks or inefficient cleanup\n"
        "5. Concurrency bottlenecks"
    ),
}


def get_question_template(template_name: str) -> str | None:
    """Get a predefined question template by name."""
    return QUESTION_TEMPLATES.get(template_name)
