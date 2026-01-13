"""CONSENSAGENT multi-planner consensus implementation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any

from ai_orchestrator.cli_adapters.base import CLIAdapter, CLIResult, CLIStatus
from ai_orchestrator.core.workflow_phases import Plan

logger = logging.getLogger(__name__)


class PlannerRole(str, Enum):
    """Role assignment for planners based on their strengths."""

    SYNTHESIS = "synthesis"  # Claude Opus - completeness, correctness
    SECURITY = "security"  # GPT-5 - security, testability
    ARCHITECTURE = "architecture"  # Gemini Ultra - blast radius, performance


@dataclass
class PlannerConfig:
    """Configuration for a single planner."""

    name: str
    cli_adapter: CLIAdapter
    role: PlannerRole
    scoring_focus: list[str] = field(default_factory=list)
    weight: float = 1.0  # Weight in consensus voting
    timeout_seconds: float = 900.0  # 15 minutes default


@dataclass
class PlanCandidate:
    """A plan candidate from a single planner."""

    planner_name: str
    role: PlannerRole
    plan: Plan
    cli_result: CLIResult
    generation_time: float
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        """Calculate overall score from individual scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class DivergenceArea:
    """An area where planners disagree."""

    aspect: str  # e.g., "architecture", "security_approach", "testing_strategy"
    positions: dict[str, str]  # planner_name -> their position
    severity: str  # "high", "medium", "low"
    resolution_hint: str = ""


@dataclass
class ConsensusResult:
    """Result of the consensus planning process."""

    synthesized_plan: Plan
    candidates: list[PlanCandidate]
    divergence_areas: list[DivergenceArea]
    consensus_score: float  # 0-1, how much agreement
    iterations: int
    total_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_strong_consensus(self) -> bool:
        """Whether there is strong consensus (80%+)."""
        return self.consensus_score >= 0.8


@dataclass
class ConsensAgentConfig:
    """Configuration for CONSENSAGENT planning."""

    max_divergence_rounds: int = 2
    agreement_threshold: float = 0.8  # 80% agreement = converged
    parallel_planning: bool = True
    require_all_planners: bool = False  # Continue if some fail
    synthesis_prompt_template: str = ""


class ConsensAgentPlanner:
    """
    CONSENSAGENT implementation for multi-planner consensus.

    Instead of majority voting (expensive, often wrong), uses:
    1. Parallel planning with all configured planners
    2. Divergence detection in specific areas
    3. Structured prompt optimization for re-planning
    4. Synthesis of best elements from all plans
    """

    def __init__(
        self,
        planners: list[PlannerConfig],
        config: ConsensAgentConfig | None = None,
    ) -> None:
        """
        Initialize CONSENSAGENT planner.

        Args:
            planners: List of planner configurations.
            config: CONSENSAGENT configuration.
        """
        self.planners = planners
        self.config = config or ConsensAgentConfig()
        self._synthesis_adapter: CLIAdapter | None = None

        # Find synthesis adapter (Claude by default)
        for planner in planners:
            if planner.role == PlannerRole.SYNTHESIS:
                self._synthesis_adapter = planner.cli_adapter
                break

    async def generate_consensus_plan(
        self,
        task: str,
        research_context: str = "",
        project_context: str = "",
    ) -> ConsensusResult:
        """
        Generate a consensus plan from multiple planners.

        Args:
            task: The task to plan for.
            research_context: Research findings to include.
            project_context: Project-specific context.

        Returns:
            ConsensusResult with synthesized plan.
        """
        import time

        start_time = time.monotonic()
        iterations = 0

        # Phase 1: Parallel planning
        logger.info("Phase 1: Generating plans from %d planners", len(self.planners))
        candidates = await self._parallel_plan(task, research_context, project_context)

        if not candidates:
            raise PlanningError("All planners failed to generate plans")

        iterations += 1

        # Phase 2: Detect divergence
        logger.info("Phase 2: Detecting divergence areas")
        divergence = self._detect_divergence(candidates)

        # Phase 3: Re-plan focused areas if significant divergence
        if divergence and iterations < self.config.max_divergence_rounds:
            logger.info(
                "Phase 3: Re-planning %d divergent areas",
                len(divergence),
            )
            focused_candidates = await self._replan_focused(
                task,
                research_context,
                project_context,
                divergence,
                candidates,
            )
            candidates.extend(focused_candidates)
            iterations += 1

        # Phase 4: Synthesize best elements
        logger.info("Phase 4: Synthesizing final plan")
        synthesized = await self._synthesize_plans(
            task,
            candidates,
            research_context,
            project_context,
        )

        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(candidates, divergence)

        total_time = time.monotonic() - start_time

        return ConsensusResult(
            synthesized_plan=synthesized,
            candidates=candidates,
            divergence_areas=divergence,
            consensus_score=consensus_score,
            iterations=iterations,
            total_time_seconds=total_time,
        )

    async def _parallel_plan(
        self,
        task: str,
        research_context: str,
        project_context: str,
    ) -> list[PlanCandidate]:
        """Generate plans from all planners in parallel."""
        import time

        prompt = self._build_planning_prompt(task, research_context, project_context)

        async def invoke_planner(planner: PlannerConfig) -> PlanCandidate | None:
            start = time.monotonic()
            try:
                result = await planner.cli_adapter.invoke(
                    prompt,
                    planning_mode=True,
                    timeout_seconds=planner.timeout_seconds,
                )

                if result.status != CLIStatus.SUCCESS:
                    logger.warning(
                        "Planner %s failed: %s",
                        planner.name,
                        result.stderr[:200] if result.stderr else "unknown error",
                    )
                    return None

                # Parse plan from output
                plan = self._parse_plan_output(result.stdout, planner.name)
                duration = time.monotonic() - start

                return PlanCandidate(
                    planner_name=planner.name,
                    role=planner.role,
                    plan=plan,
                    cli_result=result,
                    generation_time=duration,
                )

            except Exception as e:
                logger.error("Planner %s exception: %s", planner.name, e)
                return None

        # Run all planners in parallel
        if self.config.parallel_planning:
            results = await asyncio.gather(
                *[invoke_planner(p) for p in self.planners],
                return_exceptions=True,
            )
        else:
            results = []
            for planner in self.planners:
                result = await invoke_planner(planner)
                results.append(result)

        # Filter successful results
        candidates = [r for r in results if isinstance(r, PlanCandidate)]

        logger.info(
            "Generated %d/%d plans successfully",
            len(candidates),
            len(self.planners),
        )

        return candidates

    def _detect_divergence(
        self,
        candidates: list[PlanCandidate],
    ) -> list[DivergenceArea]:
        """Detect areas where planners disagree."""
        if len(candidates) < 2:
            return []

        divergence_areas = []

        # Compare key aspects across all plans
        aspects_to_compare = [
            "architecture",
            "security_approach",
            "testing_strategy",
            "implementation_order",
            "error_handling",
        ]

        for aspect in aspects_to_compare:
            positions: dict[str, str] = {}

            for candidate in candidates:
                # Extract aspect from plan content
                position = self._extract_aspect(candidate.plan, aspect)
                if position:
                    positions[candidate.planner_name] = position

            # Check if there's significant divergence
            if len(set(positions.values())) > 1:
                # Calculate divergence severity
                unique_positions = len(set(positions.values()))
                total_positions = len(positions)
                severity = "high" if unique_positions == total_positions else "medium"

                divergence_areas.append(
                    DivergenceArea(
                        aspect=aspect,
                        positions=positions,
                        severity=severity,
                    )
                )

        return divergence_areas

    async def _replan_focused(
        self,
        task: str,
        research_context: str,
        project_context: str,
        divergence: list[DivergenceArea],
        existing_candidates: list[PlanCandidate],
    ) -> list[PlanCandidate]:
        """Re-plan with focused prompts on divergent areas."""
        import time

        if not self._synthesis_adapter:
            return []

        # Build focused prompt
        focused_prompt = self._build_focused_prompt(
            task,
            research_context,
            project_context,
            divergence,
            existing_candidates,
        )

        start = time.monotonic()
        try:
            result = await self._synthesis_adapter.invoke(
                focused_prompt,
                planning_mode=True,
                timeout_seconds=900.0,
            )

            if result.status != CLIStatus.SUCCESS:
                return []

            plan = self._parse_plan_output(result.stdout, "focused_replan")
            duration = time.monotonic() - start

            return [
                PlanCandidate(
                    planner_name="focused_replan",
                    role=PlannerRole.SYNTHESIS,
                    plan=plan,
                    cli_result=result,
                    generation_time=duration,
                    metadata={"focused_areas": [d.aspect for d in divergence]},
                )
            ]

        except Exception as e:
            logger.error("Focused replan failed: %s", e)
            return []

    async def _synthesize_plans(
        self,
        task: str,
        candidates: list[PlanCandidate],
        research_context: str,
        project_context: str,
    ) -> Plan:
        """Synthesize the best elements from all plan candidates."""
        if not self._synthesis_adapter:
            # Fall back to first successful plan
            if candidates:
                return candidates[0].plan
            raise PlanningError("No plans to synthesize")

        # Build synthesis prompt
        synthesis_prompt = self._build_synthesis_prompt(
            task,
            candidates,
            research_context,
            project_context,
        )

        try:
            result = await self._synthesis_adapter.invoke(
                synthesis_prompt,
                planning_mode=True,
                timeout_seconds=900.0,
            )

            if result.status != CLIStatus.SUCCESS:
                # Fall back to highest-scoring plan
                if candidates:
                    sorted_candidates = sorted(
                        candidates,
                        key=lambda c: c.overall_score,
                        reverse=True,
                    )
                    return sorted_candidates[0].plan
                raise PlanningError("Synthesis failed and no fallback available")

            return self._parse_plan_output(result.stdout, "synthesized")

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            if candidates:
                return candidates[0].plan
            raise PlanningError(f"Synthesis failed: {e}") from e

    def _calculate_consensus_score(
        self,
        candidates: list[PlanCandidate],
        divergence: list[DivergenceArea],
    ) -> float:
        """Calculate overall consensus score."""
        if not candidates:
            return 0.0

        if len(candidates) == 1:
            return 1.0  # Single plan = full consensus

        # Base score from number of divergence areas
        divergence_penalty = len(divergence) * 0.1
        base_score = max(0.0, 1.0 - divergence_penalty)

        # Adjust for severity of divergence
        high_severity_count = sum(1 for d in divergence if d.severity == "high")
        severity_penalty = high_severity_count * 0.15

        score = max(0.0, base_score - severity_penalty)

        return min(1.0, score)

    def _build_planning_prompt(
        self,
        task: str,
        research_context: str,
        project_context: str,
    ) -> str:
        """Build the initial planning prompt."""
        parts = [
            "You are an expert software architect. Create a detailed implementation plan.",
            "",
            "## Task",
            task,
            "",
        ]

        if research_context:
            parts.extend([
                "## Research Context",
                research_context,
                "",
            ])

        if project_context:
            parts.extend([
                "## Project Context",
                project_context,
                "",
            ])

        parts.extend([
            "## Required Sections",
            "1. **Overview**: Brief summary of the approach",
            "2. **Architecture**: Key components and their relationships",
            "3. **Implementation Steps**: Ordered list of changes",
            "4. **Files to Modify**: List of files and what changes",
            "5. **Testing Strategy**: How to verify the implementation",
            "6. **Security Considerations**: Any security implications",
            "7. **Risks**: Potential issues and mitigations",
            "",
            "Provide your plan in a structured format.",
        ])

        return "\n".join(parts)

    def _build_focused_prompt(
        self,
        task: str,
        research_context: str,
        project_context: str,
        divergence: list[DivergenceArea],
        candidates: list[PlanCandidate],
    ) -> str:
        """Build a focused prompt for divergent areas."""
        parts = [
            "Multiple planning approaches were generated for this task.",
            "There is divergence in the following areas that needs resolution:",
            "",
        ]

        for area in divergence:
            parts.append(f"## {area.aspect.replace('_', ' ').title()}")
            for planner, position in area.positions.items():
                parts.append(f"- **{planner}**: {position[:200]}...")
            parts.append("")

        parts.extend([
            "## Task",
            task,
            "",
            "Please analyze these divergent approaches and provide:",
            "1. Which approach is best for each divergent area and why",
            "2. A unified plan that resolves the divergence",
            "",
        ])

        return "\n".join(parts)

    def _build_synthesis_prompt(
        self,
        task: str,
        candidates: list[PlanCandidate],
        research_context: str,
        project_context: str,
    ) -> str:
        """Build synthesis prompt combining all plans."""
        parts = [
            "You are synthesizing multiple implementation plans into one optimal plan.",
            "",
            "## Original Task",
            task,
            "",
            "## Plans to Synthesize",
            "",
        ]

        for i, candidate in enumerate(candidates, 1):
            parts.append(f"### Plan {i} (from {candidate.planner_name})")
            parts.append(candidate.plan.content[:2000])  # Truncate for context
            parts.append("")

        parts.extend([
            "## Instructions",
            "1. Extract the best elements from each plan",
            "2. Resolve any conflicts between approaches",
            "3. Create a unified, coherent implementation plan",
            "4. Ensure all security and testing considerations are included",
            "",
            "Provide the synthesized plan in structured format.",
        ])

        return "\n".join(parts)

    def _parse_plan_output(self, output: str, source: str) -> Plan:
        """Parse CLI output into a Plan object."""
        # Try to extract structured content
        content = output.strip()

        # Remove common wrapper patterns
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1])

        return Plan(
            content=content,
            source_cli=source,
        )

    def _extract_aspect(self, plan: Plan, aspect: str) -> str | None:
        """Extract a specific aspect from a plan."""
        content_lower = plan.content.lower()

        # Simple keyword-based extraction
        aspect_keywords = {
            "architecture": ["architecture", "component", "structure", "design"],
            "security_approach": ["security", "auth", "permission", "access"],
            "testing_strategy": ["test", "testing", "verify", "validation"],
            "implementation_order": ["step", "order", "first", "then", "after"],
            "error_handling": ["error", "exception", "fail", "fallback"],
        }

        keywords = aspect_keywords.get(aspect, [aspect])

        for keyword in keywords:
            idx = content_lower.find(keyword)
            if idx != -1:
                # Extract surrounding context
                start = max(0, idx - 50)
                end = min(len(plan.content), idx + 200)
                return plan.content[start:end]

        return None


class PlanningError(Exception):
    """Error during planning process."""

    pass
