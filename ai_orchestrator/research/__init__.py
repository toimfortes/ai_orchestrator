"""Research module for deep research and code analysis.

Provides tiered research capabilities:
- Quick: Single provider, fast (~2-5 min)
- Standard: 2 providers with comparison (~8-12 min)
- Council: 4 providers with full consensus (future)
"""

from ai_orchestrator.research.models import (
    ComparisonResult,
    Finding,
    ProviderResult,
    ResearchFocus,
    ResearchTier,
    Severity,
    StandardResearchResult,
)
from ai_orchestrator.research.prompts import (
    build_research_prompt,
    get_question_template,
    QUESTION_TEMPLATES,
)
from ai_orchestrator.research.comparison import (
    compare_findings,
    extract_findings,
    check_no_issues_response,
)
from ai_orchestrator.research.standard_research import (
    StandardResearch,
    quick_research,
    format_code_context,
    PROVIDERS,
)

__all__ = [
    # Models
    "ComparisonResult",
    "Finding",
    "ProviderResult",
    "ResearchFocus",
    "ResearchTier",
    "Severity",
    "StandardResearchResult",
    # Prompts
    "build_research_prompt",
    "get_question_template",
    "QUESTION_TEMPLATES",
    # Comparison
    "compare_findings",
    "extract_findings",
    "check_no_issues_response",
    # Research
    "StandardResearch",
    "quick_research",
    "format_code_context",
    "PROVIDERS",
]
