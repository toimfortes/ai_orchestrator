# Forensic Critique: Orchestrator Implementation Plan v2.0

> Applied using forensic audit methodology from research findings.

## Executive Summary

**Risk Level**: MEDIUM
**Recommendation**: APPROVE WITH CHANGES

The plan addresses most production concerns from research but has gaps in specific areas that need resolution before implementation.

---

## 1. Pre-Implementation Gate Checklist

| Gate | Status | Notes |
|------|--------|-------|
| Research completed | ✅ PASS | Extensive web research on orchestration patterns |
| Existing patterns checked | ✅ PASS | Reviewed LangGraph, AgentCoder, Microsoft patterns |
| Failure modes identified | ✅ PASS | 7 failure modes from Galileo research |
| Anti-patterns documented | ✅ PASS | 7 anti-patterns with mitigations |
| State schema defined | ⚠️ PARTIAL | Pydantic model shown but needs validation rules |
| Error recovery designed | ✅ PASS | 5-layer recovery strategy |
| Convergence criteria defined | ✅ PASS | 4 exit conditions |
| Human gates specified | ✅ PASS | 3 strategic approval points |

---

## 2. Architecture Gate Review

### Pattern Selection: ✅ APPROVED

| Criterion | Assessment |
|-----------|------------|
| Pattern fit | Coordinator + Sequential hybrid is appropriate |
| Concurrent sub-phases | Well-designed for planning/reviewing |
| Fallback strategy | Defined for each CLI |
| State persistence | Atomic writes specified |

### Identified Gaps

#### Gap 1: State Validation Rules Missing
**Problem**: `OrchestratorState` Pydantic model lacks validation.
**Risk**: Invalid state transitions could corrupt workflow.

**Resolution Required**:
```python
class OrchestratorState(BaseModel):
    current_iteration: int = Field(ge=0, le=10)  # Bounded

    @model_validator(mode='after')
    def validate_phase_transitions(self) -> 'OrchestratorState':
        # Ensure valid phase progression
        valid_transitions = {
            OrchestratorPhase.INIT: [OrchestratorPhase.DEEP_RESEARCH],
            OrchestratorPhase.DEEP_RESEARCH: [OrchestratorPhase.MEASURE_TWICE],
            # ... etc
        }
        # Validate last transition
        return self
```

#### Gap 2: Timeout Handling Incomplete
**Problem**: Timeouts defined but handling not specified.
**Risk**: Hung workflows from CLI timeouts.

**Resolution Required**:
```python
async def _invoke_with_timeout(
    self,
    adapter: CLIAdapter,
    prompt: str,
    timeout: float,
) -> CLIResult:
    try:
        return await asyncio.wait_for(
            adapter.invoke(prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        # Record timeout for circuit breaker
        self._record_timeout(adapter.name)
        # Return timeout result (don't raise)
        return CLIResult(
            cli_name=adapter.name,
            status=CLIStatus.TIMEOUT,
            exit_code=-1,
            stderr=f"Timeout after {timeout}s",
        )
```

#### Gap 3: Human Approval Timeout Not Specified
**Problem**: What happens if human doesn't respond to approval request?
**Risk**: Workflow hangs indefinitely.

**Resolution Required**:
```yaml
human_gates:
  approval_timeout: 3600  # 1 hour
  timeout_action: "escalate"  # or "auto_reject", "auto_approve_low_risk"
  escalation_channel: "slack"
```

```python
async def _await_human_approval(
    self,
    state: OrchestratorState,
    gate: str,
) -> ApprovalResult:
    timeout = self.config.human_gates.approval_timeout

    try:
        return await asyncio.wait_for(
            self.decision_gates.request_approval(gate, state),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        action = self.config.human_gates.timeout_action
        if action == "escalate":
            await self._escalate_to_secondary(gate, state)
            # Wait again with extended timeout
        elif action == "auto_reject":
            return ApprovalResult(approved=False, reason="timeout")
        elif action == "auto_approve_low_risk":
            if self._is_low_risk(state):
                return ApprovalResult(approved=True, reason="auto_low_risk")
            return ApprovalResult(approved=False, reason="timeout_high_risk")
```

#### Gap 4: Context Window Management
**Problem**: Plan mentions "summarize between phases" but no implementation.
**Risk**: Context overflow on long workflows.

**Resolution Required**:
```python
class ContextManager:
    """Manage context window across phases."""

    MAX_CONTEXT_TOKENS = 100_000  # Leave room for response

    async def prepare_context(
        self,
        state: OrchestratorState,
        phase: OrchestratorPhase,
    ) -> str:
        """Build context for next phase, summarizing if needed."""

        context_parts = []

        # Always include task
        context_parts.append(f"## Task\n{state.task}")

        # Include research (may need summarization)
        if state.research_context:
            if self._estimate_tokens(state.research_context) > 20_000:
                context_parts.append(
                    f"## Research Summary\n{await self._summarize(state.research_context)}"
                )
            else:
                context_parts.append(f"## Research\n{state.research_context}")

        # Include only relevant history based on phase
        if phase in [OrchestratorPhase.FIXING, OrchestratorPhase.REVIEWING]:
            # Include recent feedback only
            recent_feedback = state.review_rounds[-2:] if state.review_rounds else []
            context_parts.append(f"## Recent Feedback\n{self._format_feedback(recent_feedback)}")

        return "\n\n".join(context_parts)
```

#### Gap 5: Correlation ID / Tracing
**Problem**: Research emphasizes correlation IDs for debugging but plan doesn't include.
**Risk**: Difficult to trace failures across phases.

**Resolution Required**:
```python
import structlog
from uuid import uuid4

class OrchestratorState(BaseModel):
    workflow_id: str
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))

# Usage in phase handlers
async def _handle_planning(self, state: OrchestratorState) -> OrchestratorState:
    logger = structlog.get_logger().bind(
        workflow_id=state.workflow_id,
        correlation_id=state.correlation_id,
        phase="planning",
    )
    logger.info("Starting planning phase")
    # ...
```

#### Gap 6: Fallback CLI Selection Logic
**Problem**: Plan mentions fallbacks but doesn't specify selection criteria.
**Risk**: Poor fallback choices waste time.

**Resolution Required**:
```python
FALLBACK_CHAINS = {
    "planning": ["claude", "codex", "gemini"],  # Order by capability
    "reviewing": ["claude", "gemini", "codex"],
    "implementing": ["claude"],  # No fallback - Claude only
}

def _get_fallback_cli(
    self,
    phase: OrchestratorPhase,
    failed_clis: set[str],
) -> Optional[str]:
    """Get next available CLI in fallback chain."""
    chain = FALLBACK_CHAINS.get(phase.value, [])

    for cli in chain:
        if cli not in failed_clis:
            if not self.circuit_breakers.get(cli, CircuitBreaker()).is_open:
                return cli

    return None  # All fallbacks exhausted
```

---

## 3. Error Handling Gate Review

### 5-Layer Strategy: ✅ APPROVED with modifications

| Layer | Design | Assessment |
|-------|--------|------------|
| 1. Retry | Exponential backoff | ✅ Good |
| 2. Fallback | CLI chain | ⚠️ Needs selection logic |
| 3. Circuit breaker | Per-CLI | ✅ Good |
| 4. Graceful degradation | Phase-specific | ⚠️ Needs definition |
| 5. Human escalation | Approval gate | ✅ Good |

### Graceful Degradation Definition Required

```python
DEGRADATION_STRATEGIES = {
    OrchestratorPhase.MULTI_PLANNING: {
        "min_candidates": 1,  # Can proceed with 1 plan
        "action": "continue_with_partial",
    },
    OrchestratorPhase.MULTI_REVIEWING: {
        "min_reviews": 1,  # Can proceed with 1 review
        "action": "continue_with_partial",
    },
    OrchestratorPhase.IMPLEMENTING: {
        "min_candidates": 0,  # Cannot degrade
        "action": "fail_and_escalate",
    },
    OrchestratorPhase.POST_CHECKS: {
        "required_gates": ["static_analysis", "unit_tests"],
        "optional_gates": ["security_scan", "manual_smoke"],
        "action": "skip_optional_on_failure",
    },
}
```

---

## 4. Security Gate Review

| Check | Status | Notes |
|-------|--------|-------|
| No shell=True | ✅ | Using create_subprocess_exec |
| Prompt sanitization | ✅ | PromptSanitizer exists |
| No secrets in state | ⚠️ | Verify no API keys in state |
| Audit logging | ⚠️ | Need explicit audit events |
| Permission boundaries | ✅ | CLI adapters have isolation |

### Audit Logging Required

```python
class AuditLogger:
    """Log security-relevant events for compliance."""

    async def log_event(
        self,
        event_type: str,
        workflow_id: str,
        details: dict,
    ) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "workflow_id": workflow_id,
            **details,
        }
        # Write to audit log file (separate from app logs)
        await self._write_audit(event)

# Usage
await audit.log_event(
    "human_approval",
    state.workflow_id,
    {"gate": "plan_synthesis", "approved": True, "reviewer": "user@example.com"},
)
```

---

## 5. Convergence Logic Gate Review

### Exit Conditions: ✅ APPROVED

| Condition | Implementation | Assessment |
|-----------|----------------|------------|
| Zero critical for N rounds | `consecutive_clean_rounds >= 2` | ✅ Good |
| Max iterations | `current_iteration >= max_iterations` | ✅ Good |
| Stale feedback | 80% overlap detection | ✅ Good |

### Missing: Early Exit Optimization

```python
async def _check_early_exit(self, state: OrchestratorState) -> bool:
    """
    Exit early if:
    1. First round has zero issues (no iteration needed)
    2. All issues are LOW severity (not worth iterating)
    """
    if state.current_iteration == 1 and state.review_rounds:
        latest = state.review_rounds[-1]
        issues = latest.get("feedback", [])

        # Early exit: no issues at all
        if len(issues) == 0:
            return True

        # Early exit: all issues are low severity
        severities = [f.get("severity") for f in issues]
        if all(s in ["LOW", "INFO"] for s in severities):
            return True

    return False
```

---

## 6. Performance Gate Review

| Concern | Status | Notes |
|---------|--------|-------|
| Concurrent CLI calls | ✅ | Semaphore-limited |
| Token caching | ⚠️ | Relies on CLI-level caching |
| Context window efficiency | ⚠️ | Needs summarization impl |
| Timeout management | ⚠️ | Needs explicit handling |

### Token Optimization Required

```python
class TokenOptimizer:
    """Optimize token usage across workflow."""

    def __init__(self):
        self.cache = {}  # In-memory cache for prompt prefixes

    def get_cached_prefix(self, prefix_type: str) -> Optional[str]:
        """Get cached prompt prefix (static content)."""
        return self.cache.get(prefix_type)

    def cache_prefix(self, prefix_type: str, content: str) -> None:
        """Cache static prefix for reuse."""
        self.cache[prefix_type] = content

    def optimize_prompt(
        self,
        dynamic_content: str,
        prefix_type: str = "standard",
    ) -> str:
        """
        Structure prompt for maximum cache hits:
        [Cached static prefix] + [Dynamic content]
        """
        prefix = self.get_cached_prefix(prefix_type)
        if prefix:
            return f"{prefix}\n\n---\n\n{dynamic_content}"
        return dynamic_content
```

---

## 7. Testing Gate Review

| Test Type | Coverage | Notes |
|-----------|----------|-------|
| Unit tests | Planned (4h) | Need specific test list |
| Integration tests | Planned (3h) | Need CLI mock strategy |
| E2E tests | Planned (3h) | Need test scenarios |
| Chaos tests | ❌ NOT PLANNED | Should add for resilience |

### Required Test Scenarios

```python
# tests/test_orchestrator.py

class TestOrchestratorPhases:
    """Unit tests for phase handlers."""

    async def test_planning_success_multiple_candidates(self):
        """All planners succeed."""

    async def test_planning_partial_failure(self):
        """Some planners fail, continues with partial."""

    async def test_planning_total_failure(self):
        """All planners fail, graceful degradation."""

    async def test_convergence_clean_rounds(self):
        """Exit on consecutive clean rounds."""

    async def test_convergence_max_iterations(self):
        """Exit on max iterations."""

    async def test_convergence_stale_feedback(self):
        """Exit on stale feedback detection."""

class TestOrchestratorRecovery:
    """Tests for error recovery."""

    async def test_retry_transient_error(self):
        """Retry on network error."""

    async def test_fallback_on_cli_failure(self):
        """Fall back to secondary CLI."""

    async def test_circuit_breaker_trip(self):
        """Circuit trips after threshold."""

    async def test_graceful_degradation(self):
        """Continue with partial results."""

class TestOrchestratorChaos:
    """Chaos tests for resilience."""

    async def test_crash_recovery_from_checkpoint(self):
        """Resume from saved state after crash."""

    async def test_timeout_during_planning(self):
        """Handle CLI timeout gracefully."""

    async def test_all_clis_rate_limited(self):
        """Handle simultaneous rate limiting."""
```

---

## 8. Blocking Questions Resolved

| Question | Resolution |
|----------|------------|
| What if all CLIs fail? | 5-layer recovery + human escalation |
| How to detect convergence? | 4 exit conditions defined |
| How to handle human timeout? | Configurable timeout + escalation |
| How to manage context window? | Summarization between phases |
| How to trace failures? | Correlation IDs + structured logging |

---

## 9. Final Recommendation

### APPROVE with these required changes:

1. **Add state validation rules** to OrchestratorState
2. **Define human approval timeout handling**
3. **Implement context summarization** for long workflows
4. **Add correlation ID tracing**
5. **Define graceful degradation strategies** per phase
6. **Add chaos tests** for resilience validation

### Estimated additional effort: +6 hours

### Revised total: 55 hours

---

## 10. Implementation Priority

| Priority | Item | Hours |
|----------|------|-------|
| P0 | Core orchestrator + phase handlers | 15 |
| P0 | State validation + timeout handling | 3 |
| P1 | Review loop + convergence | 8 |
| P1 | Research integration | 8 |
| P1 | Context management | 2 |
| P2 | Implementation + post-checks | 8 |
| P2 | Correlation ID tracing | 1 |
| P2 | Testing (including chaos) | 10 |
| **Total** | | **55** |
