"""Pydantic settings for AI Orchestrator configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class TimeoutConfig(BaseModel):
    """Timeout configuration per CLI and phase."""

    planning: int = 900  # 15 min
    reviewing: int = 600  # 10 min
    implementing: int = 1800  # 30 min


class CLITimeouts(BaseModel):
    """Timeout configuration for all CLIs."""

    claude: TimeoutConfig = Field(default_factory=TimeoutConfig)
    codex: TimeoutConfig = Field(default_factory=lambda: TimeoutConfig(
        planning=600, reviewing=300, implementing=1200
    ))
    gemini: TimeoutConfig = Field(default_factory=lambda: TimeoutConfig(
        planning=600, reviewing=300, implementing=1200
    ))
    kilocode: TimeoutConfig = Field(default_factory=TimeoutConfig)

    # Global limits
    max_total_workflow: int = 7200  # 2 hours
    max_single_operation: int = 1800  # 30 min


class TimeoutBehavior(BaseModel):
    """Behavior on timeout."""

    action: str = "cancel_and_continue"
    retry_with_shorter: bool = True
    log_level: str = "WARNING"


class ResilienceConfig(BaseModel):
    """Resilience and circuit breaker configuration."""

    max_concurrent_clis: int = 5
    circuit_breaker_fail_max: int = 3
    circuit_breaker_reset_timeout: int = 60
    retry_attempts: int = 2
    retry_backoff: list[int] = Field(default_factory=lambda: [1, 2, 4])


class IterationSettings(BaseModel):
    """Adaptive iteration settings."""

    max_iterations: int = 4
    convergence_check_after: int = 2
    stop_on_zero_critical: bool = True
    stop_on_plateau: bool = True
    consecutive_clean_required: int = 2
    agreement_threshold: float = 0.8


class ConsensusSettings(BaseModel):
    """CONSENSAGENT planning settings."""

    max_divergence_rounds: int = 2
    agreement_threshold: float = 0.8


class HumanLoopSettings(BaseModel):
    """Human-in-the-loop decision point settings."""

    after_plan_synthesis: bool = True
    after_final_iteration: str = "conditional"  # true, false, conditional
    before_implementing: str = "conditional"
    after_post_checks_failure: bool = True


class CachingSettings(BaseModel):
    """Token caching settings."""

    enabled: bool = True
    cache_research_context: bool = True
    cache_best_practices: bool = True
    cache_code_catalog: bool = True


class ModelInfo(BaseModel):
    """Information about an AI model."""

    id: str
    name: str
    tier: str = "standard"  # free, fast, standard, premium


# Codex reasoning levels (from `codex /model` command)
CODEX_REASONING_LEVELS: list[dict[str, str]] = [
    {"id": "low", "name": "Low (Fast)", "description": "Quick responses, minimal reasoning"},
    {"id": "medium", "name": "Medium (Balanced)", "description": "Good balance of speed and depth"},
    {"id": "high", "name": "High (Thorough)", "description": "Detailed reasoning and analysis"},
    {"id": "extra_high", "name": "Extra High (Maximum)", "description": "Maximum reasoning depth"},
]


class ProviderModels(BaseModel):
    """Available models for a provider."""

    models: list[ModelInfo] = Field(default_factory=list)

    def get_default(self) -> str:
        """Get the default (first) model ID."""
        return self.models[0].id if self.models else ""


class AvailableModels(BaseModel):
    """All available models by provider (updated January 2026)."""

    # Anthropic Claude models - from `claude /model` command
    claude: ProviderModels = Field(default_factory=lambda: ProviderModels(models=[
        ModelInfo(id="opus", name="Opus 4.5 (Most Capable, Default)", tier="premium"),
        ModelInfo(id="sonnet", name="Sonnet 4.5 (Best for Everyday)", tier="standard"),
        ModelInfo(id="haiku", name="Haiku 4.5 (Fastest)", tier="fast"),
    ]))

    # OpenAI Codex models - from `codex /model` command
    # Note: Models support reasoning levels (low, medium, high, extra_high)
    codex: ProviderModels = Field(default_factory=lambda: ProviderModels(models=[
        ModelInfo(id="gpt-5.2-codex", name="GPT-5.2 Codex (Latest Agentic)", tier="premium"),
        ModelInfo(id="gpt-5.1-codex-max", name="GPT-5.1 Codex Max (Deep Reasoning)", tier="premium"),
        ModelInfo(id="gpt-5.1-codex-mini", name="GPT-5.1 Codex Mini (Fast)", tier="fast"),
        ModelInfo(id="gpt-5.2", name="GPT-5.2 (Latest Frontier)", tier="premium"),
    ]))

    # Google Gemini models - from `gemini /model` command
    # Note: "auto" modes let CLI pick best model for the task
    gemini: ProviderModels = Field(default_factory=lambda: ProviderModels(models=[
        # Auto modes
        ModelInfo(id="auto-gemini-3", name="Auto (Gemini 3) - CLI picks pro/flash", tier="standard"),
        ModelInfo(id="auto-gemini-2.5", name="Auto (Gemini 2.5) - CLI picks pro/flash", tier="standard"),
        # Manual models
        ModelInfo(id="gemini-3-pro-preview", name="Gemini 3 Pro Preview", tier="premium"),
        ModelInfo(id="gemini-3-flash-preview", name="Gemini 3 Flash Preview", tier="standard"),
        ModelInfo(id="gemini-2.5-pro", name="Gemini 2.5 Pro", tier="premium"),
        ModelInfo(id="gemini-2.5-flash", name="Gemini 2.5 Flash", tier="standard"),
        ModelInfo(id="gemini-2.5-flash-lite", name="Gemini 2.5 Flash Lite (Fast)", tier="fast"),
    ]))

    # Kilocode models (OpenRouter) - user selected
    kilocode: ProviderModels = Field(default_factory=lambda: ProviderModels(models=[
        # Free tier
        ModelInfo(id="x-ai/grok-code-fast-1", name="Grok Code Fast 1 (Free)", tier="free"),
        ModelInfo(id="mistralai/devstral-2512", name="Devstral 2512 (Free)", tier="free"),
        ModelInfo(id="giga-potato", name="Giga Potato (Free)", tier="free"),
        # Paid
        ModelInfo(id="x-ai/grok-4.1-fast", name="Grok 4.1 Fast", tier="standard"),
        ModelInfo(id="deepseek/deepseek-v3.2", name="DeepSeek V3.2", tier="standard"),
        ModelInfo(id="moonshotai/kimi-k2-thinking", name="Kimi K2 Thinking", tier="premium"),
    ]))

    def get_provider(self, provider: str) -> ProviderModels | None:
        """Get models for a specific provider."""
        return getattr(self, provider, None)

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Convert to dictionary format for API."""
        result = {}
        for provider in ["claude", "codex", "gemini", "kilocode"]:
            provider_models = getattr(self, provider)
            result[provider] = [m.model_dump() for m in provider_models.models]
        return result


class CLIConfig(BaseModel):
    """Configuration for a single CLI."""

    name: str
    enabled: bool = True
    timeout_multiplier: float = 1.0
    extra_args: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Main settings for AI Orchestrator."""

    model_config = SettingsConfigDict(
        env_prefix="AI_ORCHESTRATOR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # General
    debug: bool = False
    verbose: bool = False
    log_level: str = "INFO"

    # CLIs
    enabled_clis: list[str] = Field(default_factory=lambda: ["claude"])
    default_cli: str = "claude"

    # Timeouts
    timeouts: CLITimeouts = Field(default_factory=CLITimeouts)
    timeout_behavior: TimeoutBehavior = Field(default_factory=TimeoutBehavior)

    # Resilience
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)

    # Iteration
    iteration: IterationSettings = Field(default_factory=IterationSettings)

    # Consensus
    consensus: ConsensusSettings = Field(default_factory=ConsensusSettings)

    # Human-in-loop
    human_in_loop: HumanLoopSettings = Field(default_factory=HumanLoopSettings)

    # Caching
    caching: CachingSettings = Field(default_factory=CachingSettings)

    # Available models per provider
    available_models: AvailableModels = Field(default_factory=AvailableModels)

    # Paths
    state_dir: str = ".ai_orchestrator"

    def get_timeout_for_cli(self, cli: str, phase: str) -> int:
        """Get timeout in seconds for a CLI and phase."""
        cli_timeouts = getattr(self.timeouts, cli, self.timeouts.claude)
        timeout = getattr(cli_timeouts, phase, cli_timeouts.planning)
        return min(timeout, self.timeouts.max_single_operation)

    def get_enabled_clis(self) -> list[str]:
        """Get list of enabled CLI names."""
        return self.enabled_clis


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_settings_from_yaml(yaml_path: Path) -> Settings:
    """Load settings from a YAML file."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Settings.model_validate(data or {})


def get_default_config() -> dict[str, Any]:
    """Get default configuration as a dictionary."""
    return Settings().model_dump()
