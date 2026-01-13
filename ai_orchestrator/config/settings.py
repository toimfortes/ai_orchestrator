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
