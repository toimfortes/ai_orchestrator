"""Tests for ReviewerRouter and specialist routing."""

import pytest

from ai_orchestrator.reviewing.feedback_classifier import (
    Actionability,
    ClassificationResult,
    ClassifiedFeedback,
    IssueCategory,
    IssueSeverity,
)
from ai_orchestrator.reviewing.reviewer_router import (
    CATEGORY_TO_STRENGTH,
    DEFAULT_REVIEWER_PROFILES,
    ReviewerProfile,
    ReviewerRouter,
    ReviewerStrength,
    ReviewRoutingPlan,
    RoutingDecision,
    create_default_router,
)


class TestReviewerProfiles:
    """Tests for reviewer profile definitions."""

    def test_default_profiles_exist(self):
        """Default profiles are defined for all expected CLIs."""
        assert "claude" in DEFAULT_REVIEWER_PROFILES
        assert "codex" in DEFAULT_REVIEWER_PROFILES
        assert "gemini" in DEFAULT_REVIEWER_PROFILES
        assert "kilocode" in DEFAULT_REVIEWER_PROFILES

    def test_claude_profile_strengths(self):
        """Claude has expected security and architecture strengths."""
        profile = DEFAULT_REVIEWER_PROFILES["claude"]
        assert ReviewerStrength.SECURITY in profile.strengths
        assert ReviewerStrength.ARCHITECTURE in profile.strengths
        assert ReviewerStrength.CORRECTNESS in profile.strengths

    def test_gemini_profile_strengths(self):
        """Gemini has performance strength."""
        profile = DEFAULT_REVIEWER_PROFILES["gemini"]
        assert ReviewerStrength.PERFORMANCE in profile.strengths
        assert ReviewerStrength.ARCHITECTURE in profile.strengths

    def test_tier_ordering(self):
        """Reviewers are in expected tier order."""
        claude_tier = DEFAULT_REVIEWER_PROFILES["claude"].tier
        gemini_tier = DEFAULT_REVIEWER_PROFILES["gemini"].tier
        kilocode_tier = DEFAULT_REVIEWER_PROFILES["kilocode"].tier

        assert claude_tier < gemini_tier < kilocode_tier

    def test_category_to_strength_mapping(self):
        """All issue categories map to reviewer strengths."""
        for category in IssueCategory:
            assert category in CATEGORY_TO_STRENGTH


class TestReviewerRouter:
    """Tests for ReviewerRouter routing logic."""

    def test_create_default_router(self):
        """Can create router with default configuration."""
        router = create_default_router(["claude", "codex", "gemini"])
        assert len(router.available_reviewers) == 3
        assert router.profiles is DEFAULT_REVIEWER_PROFILES

    def test_round_one_all_reviewers(self):
        """Round 1 (no prior feedback) uses all reviewers."""
        router = ReviewerRouter(["claude", "codex", "gemini"])
        plan = router.create_routing_plan(feedback=None)

        assert set(plan.general_reviewers) == {"claude", "codex", "gemini"}
        assert len(plan.specialist_assignments) == 0
        assert len(plan.specialist_issues) == 0

    def test_force_general_only(self):
        """force_general_only skips specialist routing."""
        router = ReviewerRouter(["claude", "gemini"])
        feedback = self._create_feedback_with_security_issue()

        plan = router.create_routing_plan(feedback, force_general_only=True)

        assert set(plan.general_reviewers) == {"claude", "gemini"}
        assert len(plan.specialist_assignments) == 0

    def test_security_issues_route_to_specialists(self):
        """Critical security issues route to security specialists."""
        router = ReviewerRouter(["claude", "codex", "gemini"])
        feedback = self._create_feedback_with_security_issue()

        plan = router.create_routing_plan(feedback)

        # Security should be assigned to claude (has SECURITY strength)
        assert IssueCategory.SECURITY in plan.specialist_assignments
        assert "claude" in plan.specialist_assignments[IssueCategory.SECURITY]

    def test_performance_issues_route_to_gemini(self):
        """Performance issues route to gemini."""
        router = ReviewerRouter(["claude", "gemini", "codex"])
        feedback = self._create_feedback_with_performance_issue()

        plan = router.create_routing_plan(feedback)

        assert IssueCategory.PERFORMANCE in plan.specialist_assignments
        assert "gemini" in plan.specialist_assignments[IssueCategory.PERFORMANCE]

    def test_specialist_issues_tracked(self):
        """Critical/high issues in routed categories are tracked."""
        router = ReviewerRouter(["claude", "gemini"])
        feedback = self._create_feedback_with_security_issue()

        plan = router.create_routing_plan(feedback)

        assert len(plan.specialist_issues) > 0
        assert all(
            issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
            for issue in plan.specialist_issues
        )

    def test_max_specialists_per_category(self):
        """Respects max_specialists_per_category limit."""
        router = ReviewerRouter(
            ["claude", "codex", "gemini"],
            max_specialists_per_category=1,
        )
        feedback = self._create_feedback_with_security_issue()

        plan = router.create_routing_plan(feedback)

        for reviewers in plan.specialist_assignments.values():
            assert len(reviewers) <= 1

    def test_at_least_one_general_reviewer(self):
        """Always keeps at least one general reviewer."""
        router = ReviewerRouter(["claude"])  # Only one reviewer
        feedback = self._create_feedback_with_security_issue()

        plan = router.create_routing_plan(feedback)

        # Should have at least one general reviewer even with specialists
        assert len(plan.general_reviewers) >= 1

    def test_get_reviewers_for_category(self):
        """get_reviewers_for_category combines general + specialist."""
        plan = ReviewRoutingPlan(
            general_reviewers=["codex"],
            specialist_assignments={IssueCategory.SECURITY: ["claude"]},
            specialist_issues=[],
        )

        reviewers = plan.get_reviewers_for_category(IssueCategory.SECURITY)
        assert "claude" in reviewers
        assert "codex" in reviewers

    def test_routing_decision_priority(self):
        """Security issues get higher priority than others."""
        router = ReviewerRouter(["claude", "gemini"])
        feedback = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="N+1 query issue",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.PERFORMANCE,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="test",
            raw_text="test",
        )

        plan = router.create_routing_plan(feedback)

        security_decision = next(
            (d for d in plan.decisions if d.category == IssueCategory.SECURITY),
            None,
        )
        perf_decision = next(
            (d for d in plan.decisions if d.category == IssueCategory.PERFORMANCE),
            None,
        )

        if security_decision and perf_decision:
            assert security_decision.priority <= perf_decision.priority

    def test_get_prompt_for_specialist(self):
        """Generate focused prompts for specialists."""
        router = ReviewerRouter(["claude"])
        issues = [
            ClassifiedFeedback(
                original_text="SQL injection in user input",
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SECURITY,
                actionability=Actionability.IMMEDIATE,
                is_blocker=True,
            ),
        ]

        prompt = router.get_prompt_for_specialist(
            "claude",
            IssueCategory.SECURITY,
            issues,
        )

        assert "security" in prompt.lower()
        assert "SQL injection" in prompt
        assert "specialist" in prompt.lower()

    def test_routing_plan_summary(self):
        """Routing plan generates human-readable summary."""
        plan = ReviewRoutingPlan(
            general_reviewers=["codex", "gemini"],
            specialist_assignments={
                IssueCategory.SECURITY: ["claude"],
            },
            specialist_issues=[],
            decisions=[
                RoutingDecision(
                    category=IssueCategory.SECURITY,
                    assigned_reviewers=["claude"],
                    reason="Critical security issues",
                ),
            ],
        )

        summary = plan.summary()

        assert "General reviewers" in summary
        assert "Specialist" in summary
        assert "security" in summary.lower()

    def _create_feedback_with_security_issue(self) -> ClassificationResult:
        """Create test feedback with a security issue."""
        return ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability in user input",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="test_reviewer",
            raw_text="SQL injection vulnerability found",
        )

    def _create_feedback_with_performance_issue(self) -> ClassificationResult:
        """Create test feedback with a performance issue."""
        return ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="N+1 query causing slow response",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.PERFORMANCE,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="test_reviewer",
            raw_text="Performance issue found",
        )


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Can create routing decision with all fields."""
        decision = RoutingDecision(
            category=IssueCategory.SECURITY,
            assigned_reviewers=["claude", "codex"],
            reason="Security specialist needed",
            priority=1,
        )

        assert decision.category == IssueCategory.SECURITY
        assert len(decision.assigned_reviewers) == 2
        assert decision.priority == 1


class TestLowSeverityFiltering:
    """Tests for filtering low severity issues from specialist routing."""

    def test_low_severity_not_routed(self):
        """Low severity issues don't trigger specialist routing."""
        router = ReviewerRouter(["claude", "gemini"])
        feedback = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Consider adding a comment here",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.DOCUMENTATION,
                    actionability=Actionability.NICE_TO_HAVE,
                    is_blocker=False,
                ),
            ],
            reviewer_name="test",
            raw_text="Minor suggestion",
        )

        plan = router.create_routing_plan(feedback)

        # Low severity shouldn't trigger specialist routing
        assert len(plan.specialist_assignments) == 0

    def test_medium_severity_threshold(self):
        """Multiple medium issues accumulate to trigger routing."""
        router = ReviewerRouter(["claude", "gemini"])
        feedback = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Input validation issue 1",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=False,
                ),
                ClassifiedFeedback(
                    original_text="Input validation issue 2",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=False,
                ),
                ClassifiedFeedback(
                    original_text="Input validation issue 3",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=False,
                ),
            ],
            reviewer_name="test",
            raw_text="Multiple medium issues",
        )

        plan = router.create_routing_plan(feedback)

        # Accumulated medium issues should trigger routing
        # 3 medium issues = 6 points (threshold is 5)
        assert IssueCategory.SECURITY in plan.specialist_assignments
