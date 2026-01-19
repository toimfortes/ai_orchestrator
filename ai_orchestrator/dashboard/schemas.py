"""API schemas for the AI Orchestrator Dashboard."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Roles an agent can perform."""
    PLANNER = "planner"
    REVIEWER = "reviewer"
    IMPLEMENTER = "implementer"
    RESEARCHER = "researcher"


class AgentStatus(str, Enum):
    """Current status of an agent."""
    AVAILABLE = "available"
    BUSY = "busy"
    DISABLED = "disabled"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class ResearchProvider(str, Enum):
    """Available deep research providers (via MCP browsing)."""
    GEMINI = "gemini"  # gemini.google.com
    CHATGPT = "chatgpt"  # chatgpt.com (OpenAI)
    CLAUDE = "claude"  # claude.com (Anthropic)
    PERPLEXITY = "perplexity"  # perplexity.ai
    NONE = "none"


class WebSearchProvider(str, Enum):
    """Available web search providers."""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    SERPER = "serper"
    TAVILY = "tavily"
    NONE = "none"


# === Agent Configuration ===

class AgentConfig(BaseModel):
    """Configuration for a single AI agent."""
    name: str
    display_name: str
    enabled: bool = True
    roles: list[AgentRole] = Field(default_factory=list)
    priority: int = 1  # Lower is higher priority
    model: str | None = None  # Specific model to use
    extra_args: list[str] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.AVAILABLE
    last_used: datetime | None = None
    success_rate: float = 1.0
    avg_response_time: float = 0.0


class AgentAssignment(BaseModel):
    """Assignment of agents to workflow roles."""
    planners: list[str] = Field(default_factory=lambda: ["claude"])
    reviewers: list[str] = Field(default_factory=lambda: ["claude"])
    implementers: list[str] = Field(default_factory=lambda: ["claude"])
    researchers: list[str] = Field(default_factory=list)


# === Timeout Configuration ===

class PhaseTimeouts(BaseModel):
    """Timeouts for each workflow phase (in seconds)."""
    planning: int = 900
    reviewing: int = 600
    implementing: int = 1800
    researching: int = 1200
    post_checks: int = 300


class TimeoutConfig(BaseModel):
    """Complete timeout configuration."""
    claude: PhaseTimeouts = Field(default_factory=PhaseTimeouts)
    codex: PhaseTimeouts = Field(default_factory=lambda: PhaseTimeouts(
        planning=600, reviewing=300, implementing=1200
    ))
    gemini: PhaseTimeouts = Field(default_factory=lambda: PhaseTimeouts(
        planning=600, reviewing=300, implementing=1200
    ))
    kilocode: PhaseTimeouts = Field(default_factory=PhaseTimeouts)
    max_total_workflow: int = 7200
    max_single_operation: int = 1800
    timeout_action: str = "cancel_and_continue"
    retry_on_timeout: bool = True


# === Research Configuration ===

class ResearchConfig(BaseModel):
    """Deep research configuration."""
    enabled: bool = False
    # Multiple providers can be selected (additive, not exclusive)
    providers: list[ResearchProvider] = Field(
        default_factory=lambda: [ResearchProvider.GEMINI]
    )
    # Legacy single provider field for backward compatibility
    provider: ResearchProvider | None = None
    timeout: int = 1200
    max_sources: int = 10
    include_code_examples: bool = True
    search_depth: str = "comprehensive"  # quick, moderate, comprehensive
    api_key: str | None = None  # Masked in responses

    def model_post_init(self, __context: Any) -> None:
        """Migrate legacy provider to providers list."""
        if self.provider and self.provider not in self.providers:
            self.providers.append(self.provider)


class WebSearchConfig(BaseModel):
    """Web search configuration."""
    enabled: bool = False
    provider: WebSearchProvider = WebSearchProvider.GOOGLE
    max_results: int = 5
    include_snippets: bool = True
    safe_search: bool = True
    api_key: str | None = None  # Masked in responses


# === Prompt Configuration ===

class PromptEnhancement(BaseModel):
    """Prompt amelioration/enhancement settings."""
    enabled: bool = True
    add_reasoning_steps: bool = True
    add_verification_prompts: bool = True
    inject_best_practices: bool = True
    include_code_context: bool = True
    max_context_files: int = 10
    gemini_depth_enhancement: bool = True
    planning_mode_prefix: bool = True
    custom_prefix: str | None = None
    custom_suffix: str | None = None


# === Workflow Configuration ===

class IterationConfig(BaseModel):
    """Workflow iteration settings."""
    max_iterations: int = 4
    convergence_check_after: int = 2
    stop_on_zero_critical: bool = True
    stop_on_plateau: bool = True
    consecutive_clean_required: int = 2
    agreement_threshold: float = 0.8


class HumanLoopConfig(BaseModel):
    """Human-in-the-loop configuration."""
    after_plan_synthesis: bool = True
    after_final_iteration: str = "conditional"
    before_implementing: str = "conditional"
    after_post_checks_failure: bool = True
    auto_approve_timeout: int | None = None  # Auto-approve after N seconds


class ResilienceConfig(BaseModel):
    """Resilience and circuit breaker settings."""
    max_concurrent_clis: int = 5
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_timeout: int = 60
    retry_attempts: int = 2
    retry_backoff: list[int] = Field(default_factory=lambda: [1, 2, 4])
    enable_fallback: bool = True
    fallback_chains: dict[str, list[str]] = Field(default_factory=lambda: {
        "planning": ["claude", "codex", "gemini"],
        "reviewing": ["claude", "gemini", "codex"],
        "implementing": ["claude", "codex", "gemini"],
    })


class IncrementalReviewConfig(BaseModel):
    """Incremental review settings."""
    enabled: bool = False
    granularity: str = "file"  # file, commit, hunk, time
    threshold_lines: int = 10
    review_agent: str = "gemini"


class PostChecksConfig(BaseModel):
    """Post-implementation verification settings."""
    static_analysis: bool = True
    unit_tests: bool = True
    build_check: bool = True
    security_scan: bool = False
    manual_smoke_test: bool = False
    custom_commands: list[str] = Field(default_factory=list)


# === Complete Dashboard Configuration ===

class DashboardConfig(BaseModel):
    """Complete configuration for the AI Orchestrator dashboard."""
    # General
    debug: bool = False
    verbose: bool = False
    log_level: str = "INFO"

    # Agents
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    agent_assignment: AgentAssignment = Field(default_factory=AgentAssignment)

    # Timeouts
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)

    # Research
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)

    # Prompts
    prompt_enhancement: PromptEnhancement = Field(default_factory=PromptEnhancement)

    # Workflow
    iteration: IterationConfig = Field(default_factory=IterationConfig)
    human_loop: HumanLoopConfig = Field(default_factory=HumanLoopConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    incremental_review: IncrementalReviewConfig = Field(default_factory=IncrementalReviewConfig)
    post_checks: PostChecksConfig = Field(default_factory=PostChecksConfig)


# === Workflow Status Models ===

class PhaseStatus(BaseModel):
    """Status of a workflow phase."""
    phase: str
    status: str  # pending, in_progress, completed, failed, skipped
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    agent_used: str | None = None
    error: str | None = None


class WorkflowStatus(BaseModel):
    """Current workflow status for monitoring."""
    workflow_id: str
    prompt: str
    project_path: str
    current_phase: str
    phases: list[PhaseStatus] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 4
    convergence_status: str = "improving"
    critical_issues: int = 0
    files_modified: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None


class AgentMetrics(BaseModel):
    """Metrics for a single agent."""
    name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    avg_duration_seconds: float = 0.0
    total_tokens_used: int = 0
    cache_hits: int = 0
    last_error: str | None = None
    circuit_breaker_state: str = "closed"


class DashboardMetrics(BaseModel):
    """Aggregate metrics for the dashboard."""
    total_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    active_workflows: int = 0
    agents: dict[str, AgentMetrics] = Field(default_factory=dict)
    total_duration_seconds: float = 0.0
    avg_workflow_duration: float = 0.0


# === API Request/Response Models ===

class RunWorkflowRequest(BaseModel):
    """Request to start a new workflow."""
    prompt: str
    project_path: str | None = None
    dry_run: bool = False
    plan_only: bool = False
    config_overrides: dict[str, Any] | None = None


class RunWorkflowResponse(BaseModel):
    """Response from starting a workflow."""
    workflow_id: str
    status: str
    message: str


class UpdateConfigRequest(BaseModel):
    """Request to update configuration."""
    config: DashboardConfig


class UpdateConfigResponse(BaseModel):
    """Response from updating configuration."""
    success: bool
    message: str
    config: DashboardConfig | None = None


class AgentActionRequest(BaseModel):
    """Request to perform an action on an agent."""
    agent: str
    action: str  # enable, disable, reset_circuit_breaker, test


class AgentActionResponse(BaseModel):
    """Response from agent action."""
    success: bool
    message: str
    agent_status: AgentConfig | None = None
