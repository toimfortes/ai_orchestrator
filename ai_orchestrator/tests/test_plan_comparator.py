"""Tests for PlanComparator."""

import pytest

from ai_orchestrator.core.workflow_phases import Plan
from ai_orchestrator.planning.plan_comparator import (
    ComparisonResult,
    CriterionScore,
    PlanComparator,
    PlanScore,
    ScoringCriterion,
)


class TestPlanComparator:
    """Tests for PlanComparator."""

    def test_score_complete_plan(self):
        """Test scoring a complete plan."""
        comparator = PlanComparator()

        plan = Plan(
            content="""
            ## Overview
            This plan adds user authentication to the application.

            ## Architecture
            We will use JWT tokens for stateless authentication.

            ## Implementation Steps
            1. Add auth middleware
            2. Create login endpoint
            3. Add token validation

            ## Testing Strategy
            Unit tests for middleware and integration tests for endpoints.

            ## Security Considerations
            Tokens will be encrypted and have short expiration times.
            We will sanitize all inputs and use parameterized queries.

            ## Risks
            - Token expiration handling
            - Rate limiting needed
            """,
            source_cli="claude",
        )

        score = comparator.score_plan(plan)

        assert score.plan_source == "claude"
        assert score.overall_score > 0.4  # Should score reasonably well
        assert len(score.criterion_scores) == len(ScoringCriterion)

        # Should score high on completeness
        completeness = score.get_score(ScoringCriterion.COMPLETENESS)
        assert completeness >= 0.8

    def test_score_incomplete_plan(self):
        """Test scoring an incomplete plan."""
        comparator = PlanComparator()

        plan = Plan(
            content="Just add a button to the page.",
            source_cli="minimal",
        )

        score = comparator.score_plan(plan)

        assert score.plan_source == "minimal"
        assert score.overall_score < 0.5  # Should score poorly

        # Should have weaknesses
        assert len(score.weaknesses) > 0

    def test_compare_multiple_plans(self):
        """Test comparing multiple plans."""
        comparator = PlanComparator()

        good_plan = Plan(
            content="""
            ## Overview
            Complete implementation plan.

            ## Architecture
            Well-designed component structure.

            ## Implementation
            Clear step-by-step guide.

            ## Testing
            Comprehensive test coverage.

            ## Security
            All security considerations addressed.
            """,
            source_cli="good_planner",
        )

        poor_plan = Plan(
            content="Just do it.",
            source_cli="poor_planner",
        )

        result = comparator.compare_plans([good_plan, poor_plan])

        assert result.best_plan_source == "good_planner"
        assert result.ranking[0] == "good_planner"
        assert result.ranking[1] == "poor_planner"
        assert len(result.scores) == 2

    def test_compare_empty_list(self):
        """Test comparing empty list of plans."""
        comparator = PlanComparator()
        result = comparator.compare_plans([])

        assert result.best_plan_source == ""
        assert result.ranking == []

    def test_explain_comparison(self):
        """Test generating comparison explanation."""
        comparator = PlanComparator()

        plan1 = Plan(content="## Overview\nGood plan", source_cli="plan1")
        plan2 = Plan(content="Bad plan", source_cli="plan2")

        result = comparator.compare_plans([plan1, plan2])
        explanation = comparator.explain_comparison(result)

        assert "PLAN COMPARISON" in explanation
        assert "1st" in explanation or "[1st]" in explanation

    def test_custom_weights(self):
        """Test custom scoring weights."""
        weights = {
            ScoringCriterion.SECURITY: 0.5,  # High weight on security
            ScoringCriterion.COMPLETENESS: 0.1,
        }
        comparator = PlanComparator(weights=weights)

        security_plan = Plan(
            content="""
            Security is paramount. We will use encryption, authentication,
            access control, and sanitize all inputs.
            """,
            source_cli="security_focused",
        )

        score = comparator.score_plan(security_plan)

        # Security criterion should have higher weight
        security_criterion = next(
            cs for cs in score.criterion_scores
            if cs.criterion == ScoringCriterion.SECURITY
        )
        assert security_criterion.weight == 0.5


class TestPlanScore:
    """Tests for PlanScore dataclass."""

    def test_overall_score_calculation(self):
        """Test overall score calculation."""
        score = PlanScore(
            plan_source="test",
            criterion_scores=[
                CriterionScore(
                    criterion=ScoringCriterion.COMPLETENESS,
                    score=0.8,
                    weight=0.5,
                ),
                CriterionScore(
                    criterion=ScoringCriterion.SECURITY,
                    score=0.6,
                    weight=0.5,
                ),
            ],
        )

        # (0.8 * 0.5 + 0.6 * 0.5) / 1.0 = 0.7
        assert score.overall_score == pytest.approx(0.7)

    def test_strengths_and_weaknesses(self):
        """Test identifying strengths and weaknesses."""
        score = PlanScore(
            plan_source="test",
            criterion_scores=[
                CriterionScore(
                    criterion=ScoringCriterion.COMPLETENESS,
                    score=0.9,  # Strength
                    weight=0.2,
                ),
                CriterionScore(
                    criterion=ScoringCriterion.SECURITY,
                    score=0.3,  # Weakness
                    weight=0.2,
                ),
                CriterionScore(
                    criterion=ScoringCriterion.PERFORMANCE,
                    score=0.5,  # Neither
                    weight=0.2,
                ),
            ],
        )

        assert "completeness" in score.strengths
        assert "security" in score.weaknesses
        assert "performance" not in score.strengths
        assert "performance" not in score.weaknesses

    def test_to_summary(self):
        """Test summary generation."""
        score = PlanScore(
            plan_source="test_plan",
            criterion_scores=[
                CriterionScore(
                    criterion=ScoringCriterion.COMPLETENESS,
                    score=0.8,
                    weight=0.5,
                ),
            ],
        )

        summary = score.to_summary()

        assert "test_plan" in summary
        assert "Overall Score" in summary
        assert "completeness" in summary
