"""CLI adapter module for interfacing with AI coding CLIs.

Each CLI has different prompting characteristics:

    Claude:   Best at implicit reasoning. Vague prompts like "deep critique" work well.
              Thorough responses by default.

    Codex:    Good at following specifications. Be explicit about requirements.
              Moderate default verbosity.

    Gemini:   Less verbose by default. Needs explicit depth instructions for thorough
              analysis. The GeminiAdapter auto-adds depth enhancement prefixes.
              For persistent config, use GEMINI.md files.

    Kilocode: Varies by underlying model (OpenRouter). Follow best practices for
              whichever model you're using (claude, gpt, gemini, etc).

See docs/cli_prompting_comparison.md for detailed prompting guidance.
See docs/gemini_prompting_guide.md for Gemini-specific depth techniques.
"""

from ai_orchestrator.cli_adapters.base import CLIAdapter, CLIResult, CLIStatus
from ai_orchestrator.cli_adapters.claude import ClaudeAdapter
from ai_orchestrator.cli_adapters.codex import CodexAdapter
from ai_orchestrator.cli_adapters.gemini import GeminiAdapter
from ai_orchestrator.cli_adapters.kilocode import KilocodeAdapter

__all__ = [
    # Base classes and types
    "CLIAdapter",
    "CLIResult",
    "CLIStatus",
    # Adapters
    "ClaudeAdapter",
    "CodexAdapter",
    "GeminiAdapter",
    "KilocodeAdapter",
    # Factory functions
    "get_adapter",
    "get_available_adapters",
]


def get_adapter(cli_name: str) -> CLIAdapter:
    """
    Get a CLI adapter by name.

    Args:
        cli_name: Name of the CLI (claude, codex, gemini, kilocode).

    Returns:
        Appropriate CLIAdapter instance.

    Raises:
        ValueError: If CLI name is not recognized.
    """
    adapters = {
        "claude": ClaudeAdapter,
        "codex": CodexAdapter,
        "gemini": GeminiAdapter,
        "kilocode": KilocodeAdapter,
    }

    adapter_class = adapters.get(cli_name.lower())
    if adapter_class is None:
        raise ValueError(f"Unknown CLI: {cli_name}. Available: {list(adapters.keys())}")

    return adapter_class()


def get_available_adapters() -> dict[str, CLIAdapter]:
    """
    Get all available CLI adapters.

    Returns:
        Dictionary of CLI name to adapter instance (only available CLIs).
    """
    all_adapters = {
        "claude": ClaudeAdapter(),
        "codex": CodexAdapter(),
        "gemini": GeminiAdapter(),
        "kilocode": KilocodeAdapter(),
    }

    return {name: adapter for name, adapter in all_adapters.items() if adapter.is_available}
