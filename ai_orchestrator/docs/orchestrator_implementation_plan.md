# AI Orchestrator Implementation Plan v2.0

> Synthesized from extensive web research on multi-agent orchestration patterns, production best practices, and failure mode analysis.

## Research Sources

- [Microsoft AI Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [ByteByteGo Agentic Workflow Patterns](https://blog.bytebytego.com/p/top-ai-agentic-workflow-patterns)
- [LangGraph State Machines in Production](https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4)
- [Why Multi-Agent LLM Systems Fail (Galileo)](https://galileo.ai/blog/multi-agent-llm-systems-fail)
- [Retries, Fallbacks, Circuit Breakers (Portkey)](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [Human-in-the-Loop Best Practices (Permit.io)](https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo)
- [AgentCoder Multi-Agent Architecture](https://arxiv.org/html/2312.13010v3)
- [Multi-Agent Orchestration Patterns](https://www.wethinkapp.ai/blog/design-patterns-for-multi-agent-orchestration)

---

## Key Research Findings

### 1. Orchestration Pattern Selection

| Pattern | When to Use | Our Application |
|---------|-------------|-----------------|
| **Sequential** | Linear pipelines, progressive refinement | RESEARCH → PLANNING → REVIEWING → FIXING |
| **Concurrent** | Independent parallel work | Multi-planner, multi-reviewer |
| **Supervisor/Coordinator** | Governance, compliance, control | Main orchestrator coordinates all phases |
| **Handoff** | Dynamic specialist routing | Route security issues to security reviewers |

**Decision**: Use **Coordinator + Sequential hybrid** with concurrent sub-phases.

### 2. Failure Mode Analysis (41-86.7% multi-agent systems fail!)

| Failure Mode | Root Cause | Our Mitigation |
|--------------|------------|----------------|
| Agent coordination breakdown | Role drift, misalignment | Explicit role boundaries in prompts |
| Context loss across handoffs | Truncation, compression | Persistent state with full history |
| Endless loops | Missing termination criteria | Max iterations + convergence detection |
| Runtime coordination failures | Race conditions, bottlenecks | Semaphore limits, circuit breakers |
| Role confusion | Unclear specialization | Single responsibility per CLI |

### 3. Production Best Practices

**State Management**:
- Explicit TypedDict/Pydantic state definitions
- Atomic persistence (temp + rename)
- Checkpoint recovery on crash
- Session tokens for history reconstruction

**Error Handling** (Layered approach):
1. **Retries** - Transient failures (network, rate limits)
2. **Fallbacks** - Route to secondary providers
3. **Circuit Breakers** - Prevent cascading failures

**Convergence Detection**:
- Define clear success criteria
- Max iteration limits (prevent infinite loops)
- Monitor for non-productive responses
- Track fallback frequency as early warning

### 4. Human-in-the-Loop Patterns

**When to require approval** (Decision framework: "Would I be OK if agent did this without asking?"):
- ✅ Access control changes
- ✅ Infrastructure modifications
- ✅ Destructive operations
- ❌ Read-only operations
- ❌ Low-risk transformations

**Implementation**:
- `interrupt()` mechanism for synchronous approval
- Policy-based approval for role enforcement
- Clear context in approval requests (not raw JSON)
- Timeout handling with fallback escalation

---

## Architecture Design

### Orchestration Pattern: Coordinator + Sequential

```
                    ┌─────────────────────┐
                    │   ORCHESTRATOR      │
                    │   (Coordinator)     │
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌───────┐              ┌───────────────┐          ┌───────────────┐
│ INIT  │─────────────▶│DEEP_RESEARCH  │─────────▶│ MEASURE_TWICE │
└───────┘              └───────────────┘          └───────┬───────┘
                                                          │
                       ┌──────────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │ MULTI_PLANNING  │ ◄─── Concurrent: Claude, Codex, Gemini
              │  (CONSENSAGENT) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ PLAN_COMPARISON │
              │  + SYNTHESIS    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐     ┌──────────────────┐
              │ [HUMAN GATE 1]  │────▶│ Approval: Plan   │
              │ After Synthesis │     │ (High-impact)    │
              └────────┬────────┘     └──────────────────┘
                       │ (approved)
                       ▼
              ┌─────────────────┐
              │ MULTI_REVIEWING │ ◄─── Concurrent + Specialist routing
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  CONSOLIDATING  │ ◄─── Merge + Classify feedback
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     FIXING      │ ◄─── Address critical issues
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │  Convergence Check    │
           │  - Zero critical?     │
           │  - Max iterations?    │
           │  - Feedback stale?    │
           └───────────┬───────────┘
                       │
        ┌──────────────┼──────────────┐
        │ (not converged)             │ (converged)
        ▼                             ▼
   Back to REVIEWING          ┌─────────────────┐
                              │ [HUMAN GATE 2]  │
                              │ Before Implement│
                              └────────┬────────┘
                                       │ (approved)
                                       ▼
                              ┌─────────────────┐
                              │  IMPLEMENTING   │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   POST_CHECKS   │ ◄─── 5 gates
                              │ (Static, Tests, │
                              │  Build, Security│
                              │  Manual Smoke)  │
                              └────────┬────────┘
                                       │
        ┌──────────────────────────────┼──────────────────┐
        │ (failed)                     │ (passed)         │
        ▼                              ▼                  │
┌─────────────────┐           ┌─────────────────┐        │
│ [HUMAN GATE 3]  │           │  FINAL_REVIEW   │        │
│ PostCheck Failed│           └────────┬────────┘        │
└─────────────────┘                    │                 │
                                       ▼                 │
                              ┌─────────────────┐        │
                              │   COMPLETED     │◄───────┘
                              └─────────────────┘
```

### State Definition (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class OrchestratorPhase(str, Enum):
    INIT = "init"
    DEEP_RESEARCH = "deep_research"
    MEASURE_TWICE = "measure_twice"
    MULTI_PLANNING = "multi_planning"
    PLAN_COMPARISON = "plan_comparison"
    PLAN_SYNTHESIS = "plan_synthesis"
    HUMAN_APPROVAL_PLAN = "human_approval_plan"
    MULTI_REVIEWING = "multi_reviewing"
    CONSOLIDATING = "consolidating"
    FIXING = "fixing"
    HUMAN_APPROVAL_IMPLEMENT = "human_approval_implement"
    IMPLEMENTING = "implementing"
    POST_CHECKS = "post_checks"
    HUMAN_APPROVAL_POSTCHECK = "human_approval_postcheck"
    FINAL_REVIEW = "final_review"
    COMPLETED = "completed"
    FAILED = "failed"

class ConvergenceStatus(str, Enum):
    IMPROVING = "improving"       # Making progress
    CONVERGED = "converged"       # Zero critical issues
    PLATEAU = "plateau"           # No improvement
    MAX_ITERATIONS = "max_iterations"

class OrchestratorState(BaseModel):
    """Explicit state for crash-safe persistence."""

    # Identity
    workflow_id: str
    task: str
    project_root: str

    # Phase tracking
    current_phase: OrchestratorPhase = OrchestratorPhase.INIT
    phase_history: list[tuple[str, datetime]] = Field(default_factory=list)

    # Research context
    research_context: Optional[str] = None
    blast_radius_report: Optional[str] = None

    # Planning
    plan_candidates: list[dict] = Field(default_factory=list)
    plan_scores: list[dict] = Field(default_factory=list)
    synthesized_plan: Optional[str] = None

    # Reviewing
    current_iteration: int = 0
    review_rounds: list[dict] = Field(default_factory=list)
    classified_feedback: list[dict] = Field(default_factory=list)

    # Convergence
    convergence_status: ConvergenceStatus = ConvergenceStatus.IMPROVING
    consecutive_clean_rounds: int = 0

    # Human decisions
    human_decisions: list[dict] = Field(default_factory=list)
    pending_approval: Optional[str] = None

    # Errors and recovery
    errors: list[str] = Field(default_factory=list)
    last_checkpoint: Optional[datetime] = None
    recovery_attempts: int = 0

    # Metrics
    started_at: datetime = Field(default_factory=lambda: datetime.now())
    completed_at: Optional[datetime] = None
    total_tokens_used: int = 0
    cli_invocations: dict[str, int] = Field(default_factory=dict)
```

---

## Component Design

### 1. Main Orchestrator

```python
class AIOrchestrator:
    """
    Coordinator-pattern orchestrator for multi-CLI code generation.

    Responsibilities:
    - Manage phase transitions
    - Coordinate CLI invocations
    - Handle state persistence
    - Enforce convergence detection
    - Gate human approvals
    """

    def __init__(
        self,
        state_manager: StateManager,
        cli_registry: CLIRegistry,
        iteration_controller: AdaptiveIterationController,
        decision_gates: DecisionGates,
        metrics_collector: MetricsCollector,
        config: OrchestratorConfig,
    ):
        self.state_manager = state_manager
        self.cli_registry = cli_registry
        self.iteration_controller = iteration_controller
        self.decision_gates = decision_gates
        self.metrics = metrics_collector
        self.config = config

        # Circuit breakers per CLI
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Phase handlers
        self._phase_handlers = {
            OrchestratorPhase.INIT: self._handle_init,
            OrchestratorPhase.DEEP_RESEARCH: self._handle_research,
            OrchestratorPhase.MEASURE_TWICE: self._handle_measure_twice,
            OrchestratorPhase.MULTI_PLANNING: self._handle_planning,
            # ... etc
        }

    async def run(self, task: str, project_root: Path) -> OrchestratorResult:
        """Main entry point - run full workflow."""

        # Try to resume from checkpoint
        state = await self.state_manager.load_state()
        if state is None:
            state = OrchestratorState(
                workflow_id=generate_workflow_id(),
                task=task,
                project_root=str(project_root),
            )

        try:
            while state.current_phase != OrchestratorPhase.COMPLETED:
                # Get handler for current phase
                handler = self._phase_handlers.get(state.current_phase)
                if handler is None:
                    raise OrchestratorError(f"No handler for phase: {state.current_phase}")

                # Execute phase
                self.metrics.start_phase(state.current_phase)
                try:
                    state = await handler(state)
                except Exception as e:
                    state = await self._handle_phase_error(state, e)
                finally:
                    self.metrics.end_phase(state.current_phase)

                # Checkpoint after each phase
                await self.state_manager.save_state_atomic(state)

            return OrchestratorResult(
                success=True,
                state=state,
                metrics=self.metrics.complete(),
            )

        except FatalError as e:
            # Unrecoverable - save state and surface to user
            state.errors.append(str(e))
            await self.state_manager.save_state_atomic(state)
            return OrchestratorResult(
                success=False,
                state=state,
                error=str(e),
            )
```

### 2. Phase Handlers

```python
async def _handle_planning(self, state: OrchestratorState) -> OrchestratorState:
    """
    MULTI_PLANNING phase - Generate plans from multiple CLIs.

    Pattern: Concurrent orchestration
    """
    planners = self.cli_registry.get_planners()

    # Build planning prompt with research context
    prompt = self._build_planning_prompt(
        task=state.task,
        research_context=state.research_context,
    )

    # Invoke all planners concurrently
    async def invoke_with_circuit_breaker(cli_name: str) -> Optional[PlanCandidate]:
        breaker = self.circuit_breakers.get(cli_name)
        if breaker and breaker.is_open:
            logger.warning(f"Circuit open for {cli_name}, skipping")
            return None

        try:
            adapter = self.cli_registry.get_adapter(cli_name)
            result = await adapter.invoke(prompt, planning_mode=True)

            if result.status == CLIStatus.SUCCESS:
                return PlanCandidate(
                    planner_name=cli_name,
                    plan=self._parse_plan(result.stdout),
                    cli_result=result,
                )
            else:
                self._record_failure(cli_name, result)
                return None

        except Exception as e:
            self._record_failure(cli_name, e)
            return None

    # Run concurrently with semaphore limit
    async with asyncio.Semaphore(self.config.max_concurrent_clis):
        candidates = await asyncio.gather(
            *[invoke_with_circuit_breaker(p) for p in planners],
            return_exceptions=True,
        )

    # Filter successful candidates
    valid_candidates = [c for c in candidates if isinstance(c, PlanCandidate)]

    if not valid_candidates:
        # All planners failed - graceful degradation
        return await self._handle_total_planning_failure(state, candidates)

    state.plan_candidates = [c.to_dict() for c in valid_candidates]
    state.current_phase = OrchestratorPhase.PLAN_COMPARISON
    return state
```

### 3. Convergence Detection

```python
async def _check_convergence(self, state: OrchestratorState) -> ConvergenceStatus:
    """
    Determine if review iterations should continue.

    Exit conditions:
    1. Zero critical issues for N consecutive rounds → CONVERGED
    2. Max iterations reached → MAX_ITERATIONS
    3. Same issues repeated (no progress) → PLATEAU
    """

    # Count critical issues in latest round
    if state.review_rounds:
        latest = state.review_rounds[-1]
        critical_count = sum(
            1 for f in latest.get("feedback", [])
            if f.get("severity") == "CRITICAL"
        )
    else:
        critical_count = 0

    # Check for consecutive clean rounds
    if critical_count == 0:
        state.consecutive_clean_rounds += 1
        if state.consecutive_clean_rounds >= self.config.clean_rounds_required:
            return ConvergenceStatus.CONVERGED
    else:
        state.consecutive_clean_rounds = 0

    # Check max iterations
    if state.current_iteration >= self.config.max_iterations:
        return ConvergenceStatus.MAX_ITERATIONS

    # Check for plateau (same issues repeating)
    if self._is_feedback_stale(state):
        return ConvergenceStatus.PLATEAU

    return ConvergenceStatus.IMPROVING


def _is_feedback_stale(self, state: OrchestratorState) -> bool:
    """Detect if feedback is repeating without progress."""
    if len(state.review_rounds) < 2:
        return False

    # Compare issue signatures between last two rounds
    current_issues = self._extract_issue_signatures(state.review_rounds[-1])
    previous_issues = self._extract_issue_signatures(state.review_rounds[-2])

    # If 80%+ overlap, consider stale
    overlap = len(current_issues & previous_issues)
    total = len(current_issues | previous_issues)

    if total == 0:
        return False

    return (overlap / total) >= 0.8
```

### 4. Error Recovery

```python
async def _handle_phase_error(
    self,
    state: OrchestratorState,
    error: Exception,
) -> OrchestratorState:
    """
    Layered error recovery strategy.

    Layer 1: Retry with backoff (transient failures)
    Layer 2: Fallback to alternative CLI (provider failure)
    Layer 3: Circuit breaker trip (persistent failure)
    Layer 4: Graceful degradation (all providers down)
    Layer 5: Human escalation (unrecoverable)
    """

    state.errors.append(f"{state.current_phase}: {error}")
    state.recovery_attempts += 1

    # Layer 1: Retry for transient errors
    if self._is_transient_error(error) and state.recovery_attempts <= 3:
        await asyncio.sleep(2 ** state.recovery_attempts)  # Exponential backoff
        return state  # Retry same phase

    # Layer 2: Try fallback CLI
    fallback = self._get_fallback_cli(state.current_phase)
    if fallback:
        logger.info(f"Falling back to {fallback} for {state.current_phase}")
        state.metadata["fallback_cli"] = fallback
        return state  # Retry with fallback

    # Layer 3: Check circuit breakers
    for cli_name in self.cli_registry.get_all():
        if self._should_trip_breaker(cli_name):
            self.circuit_breakers[cli_name].trip()
            logger.warning(f"Circuit breaker tripped for {cli_name}")

    # Layer 4: Graceful degradation
    if self._can_degrade_gracefully(state.current_phase):
        return self._apply_degradation(state)

    # Layer 5: Human escalation
    state.pending_approval = "error_escalation"
    state.current_phase = OrchestratorPhase.FAILED
    return state
```

---

## Configuration

```yaml
# ai_orchestrator/config/orchestrator.yaml

orchestrator:
  # Phase settings
  max_iterations: 4
  clean_rounds_required: 2

  # Concurrency
  max_concurrent_clis: 3
  max_concurrent_reviewers: 5

  # Timeouts (seconds)
  timeouts:
    planning: 900      # 15 min
    reviewing: 600     # 10 min
    implementing: 1800 # 30 min
    post_checks: 300   # 5 min

  # Circuit breakers
  circuit_breaker:
    fail_threshold: 3
    reset_timeout: 60
    half_open_requests: 1

  # Retry policy
  retry:
    max_attempts: 3
    backoff_multiplier: 2
    max_backoff: 30

  # Human gates
  human_gates:
    after_plan_synthesis: true
    before_implementing: conditional  # Only for high-risk changes
    after_post_checks_failure: true

  # Convergence
  convergence:
    stale_threshold: 0.8  # 80% overlap = stale
    min_iterations: 1

  # CLI preferences
  cli_preferences:
    planners: [claude, codex, gemini]
    reviewers:
      required: [claude]
      optional: [gemini, codex]
    implementers: [claude]  # Claude-only for implementation
```

---

## File Structure

```
ai_orchestrator/
├── core/
│   ├── orchestrator.py          # Main coordinator (NEW)
│   ├── phase_handlers/          # Phase-specific logic (NEW)
│   │   ├── __init__.py
│   │   ├── init_handler.py
│   │   ├── research_handler.py
│   │   ├── planning_handler.py
│   │   ├── reviewing_handler.py
│   │   ├── fixing_handler.py
│   │   ├── implementing_handler.py
│   │   └── post_checks_handler.py
│   ├── convergence.py           # Convergence detection (NEW)
│   ├── error_recovery.py        # Layered error handling (NEW)
│   ├── state_manager.py         # (EXISTS)
│   ├── iteration_controller.py  # (EXISTS)
│   └── workflow_phases.py       # (EXISTS)
├── research/
│   ├── __init__.py
│   ├── deep_research_runner.py  # MCP integration (NEW)
│   └── measure_twice_checker.py # Blast radius (NEW)
├── cli_adapters/                # (EXISTS)
├── planning/                    # (EXISTS)
├── reviewing/                   # (EXISTS)
├── human_loop/                  # (EXISTS)
├── metrics/                     # (EXISTS)
└── __main__.py                  # CLI entry point (NEW)
```

---

## Implementation Phases

### Phase 1: Core Orchestrator (P0)

| Task | Hours | Dependencies |
|------|-------|--------------|
| `orchestrator.py` main class | 4 | state_manager, cli_adapters |
| Phase handlers (init, planning) | 4 | orchestrator |
| Convergence detection | 2 | iteration_controller |
| Error recovery layer | 3 | circuit_breaker |
| `__main__.py` CLI entry | 2 | orchestrator |
| **Total** | **15** | |

### Phase 2: Review Loop (P1)

| Task | Hours | Dependencies |
|------|-------|--------------|
| Phase handlers (reviewing, fixing) | 4 | feedback_classifier |
| Specialist routing integration | 2 | reviewer_router |
| Stale feedback detection | 2 | convergence |
| **Total** | **8** | |

### Phase 3: Research Integration (P1)

| Task | Hours | Dependencies |
|------|-------|--------------|
| `deep_research_runner.py` | 3 | MCP tools |
| `measure_twice_checker.py` | 3 | code_catalog |
| Phase handlers (research, measure) | 2 | research module |
| **Total** | **8** | |

### Phase 4: Implementation & Checks (P2)

| Task | Hours | Dependencies |
|------|-------|--------------|
| Phase handlers (implementing, post_checks) | 4 | post_checks module |
| Human gate integration | 2 | decision_gates |
| Final review handler | 2 | |
| **Total** | **8** | |

### Phase 5: Testing & Polish (P2)

| Task | Hours | Dependencies |
|------|-------|--------------|
| Unit tests for orchestrator | 4 | |
| Integration test (Claude-only) | 3 | |
| E2E test script | 3 | |
| **Total** | **10** | |

**Grand Total: ~49 hours**

---

## Anti-Patterns to Avoid

Based on research findings:

| Anti-Pattern | Risk | Mitigation |
|--------------|------|------------|
| **Role drift** | Agents do wrong tasks | Explicit role in every prompt |
| **Context bloat** | Token overflow | Summarize between phases |
| **Endless loops** | Burn tokens forever | Max iterations + stale detection |
| **Shared mutable state** | Race conditions | Atomic state updates |
| **Non-specializing agents** | Wasted resources | Single responsibility per CLI |
| **Latency blindness** | Slow workflows | Concurrent where possible |
| **Wrong pattern choice** | Poor results | Use Sequential+Concurrent hybrid |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Workflow completion rate | > 90% | Completed / Started |
| Avg iterations to converge | < 2.5 | Mean iterations across workflows |
| Human intervention rate | < 20% | Workflows requiring human gates |
| Error recovery success | > 80% | Recovered / Total errors |
| Time to completion | < 30 min | Avg workflow duration |
| Token efficiency | 10x vs naive | Cache hits ratio |

---

## Risk Register

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| All CLIs fail simultaneously | HIGH | LOW | Graceful degradation + human handoff |
| Infinite review loop | MEDIUM | MEDIUM | Max iterations + stale detection |
| State corruption on crash | HIGH | LOW | Atomic writes + backups |
| Context window overflow | MEDIUM | MEDIUM | Summarization between phases |
| Rate limiting across CLIs | MEDIUM | HIGH | Circuit breakers + backoff |
| Human approval timeout | LOW | MEDIUM | Async approval + escalation |
