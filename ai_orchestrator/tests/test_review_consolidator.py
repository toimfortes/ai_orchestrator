"""Tests for ReviewConsolidator."""

import pytest

from ai_orchestrator.reviewing.feedback_classifier import (
    Actionability,
    ClassificationResult,
    ClassifiedFeedback,
    IssueCategory,
    IssueSeverity,
)
from ai_orchestrator.reviewing.review_consolidator import (
    ConsolidatedIssue,
    ConsolidationResult,
    ReviewConsolidator,
    consolidate_reviews,
)


class TestConsolidatedIssue:
    """Tests for ConsolidatedIssue dataclass."""

    def test_is_blocker_majority_vote(self):
        """is_blocker returns True if majority voted blocker."""
        issue = ConsolidatedIssue(
            source_items=[
                ClassifiedFeedback(
                    original_text="Security issue",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="Security issue",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="Security issue",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=False,
                ),
            ],
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            message="Security vulnerability",
            reviewer_count=3,
        )

        assert issue.is_blocker is True  # 2/3 voted blocker

    def test_is_blocker_minority_not_blocker(self):
        """is_blocker returns False if minority voted blocker."""
        issue = ConsolidatedIssue(
            source_items=[
                ClassifiedFeedback(
                    original_text="Issue",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.OTHER,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="Issue",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.OTHER,
                    actionability=Actionability.NICE_TO_HAVE,
                    is_blocker=False,
                ),
                ClassifiedFeedback(
                    original_text="Issue",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.OTHER,
                    actionability=Actionability.NICE_TO_HAVE,
                    is_blocker=False,
                ),
            ],
            severity=IssueSeverity.LOW,
            category=IssueCategory.OTHER,
            message="Minor issue",
            reviewer_count=3,
        )

        assert issue.is_blocker is False  # 1/3 voted blocker

    def test_has_consensus_majority(self):
        """has_consensus returns True if majority agree on severity."""
        issue = ConsolidatedIssue(
            source_items=[],
            severity=IssueSeverity.HIGH,
            category=IssueCategory.SECURITY,
            message="Test",
            reviewer_count=3,
            severity_votes={
                IssueSeverity.HIGH: 2,
                IssueSeverity.MEDIUM: 1,
            },
        )

        assert issue.has_consensus is True

    def test_has_no_consensus(self):
        """has_consensus returns False if no majority."""
        issue = ConsolidatedIssue(
            source_items=[],
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.SECURITY,
            message="Test",
            reviewer_count=4,
            severity_votes={
                IssueSeverity.HIGH: 2,
                IssueSeverity.MEDIUM: 2,
            },
        )

        assert issue.has_consensus is False  # 50-50 split

    def test_to_classified_feedback(self):
        """Convert consolidated issue back to ClassifiedFeedback."""
        issue = ConsolidatedIssue(
            source_items=[
                ClassifiedFeedback(
                    original_text="SQL injection",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=45,
                ),
            ],
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            message="SQL injection vulnerability",
            location="auth.py:45",
            reviewer_count=2,
            consensus_score=0.9,
            fix_suggestions=["Use parameterized queries"],
            reviewers=["claude", "codex"],
        )

        feedback = issue.to_classified_feedback()

        assert feedback.severity == IssueSeverity.CRITICAL
        assert feedback.category == IssueCategory.SECURITY
        assert feedback.actionability == Actionability.IMMEDIATE
        assert feedback.file_path == "auth.py"
        assert feedback.line_number == 45
        assert "parameterized" in feedback.fix_suggestion.lower()
        assert feedback.metadata["consolidated"] is True
        assert feedback.metadata["reviewer_count"] == 2

    def test_extract_file_path_from_location(self):
        """Extract file path from location string."""
        issue = ConsolidatedIssue(
            source_items=[],
            severity=IssueSeverity.HIGH,
            category=IssueCategory.SECURITY,
            message="Test",
            location="src/auth/login.py:123",
        )

        assert issue._extract_file_path() == "src/auth/login.py"
        assert issue._extract_line_number() == 123


class TestReviewConsolidator:
    """Tests for ReviewConsolidator consolidation logic."""

    def test_consolidate_empty_results(self):
        """Consolidating empty list returns empty result."""
        consolidator = ReviewConsolidator()
        result = consolidator.consolidate([])

        assert len(result.issues) == 0
        assert len(result.source_results) == 0
        assert result.total_source_items == 0

    def test_consolidate_single_result(self):
        """Consolidating single result returns all items."""
        consolidator = ReviewConsolidator()
        source = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue 1",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="claude",
            raw_text="Claude review",
        )

        result = consolidator.consolidate([source])

        assert len(result.issues) == 1
        assert result.total_source_items == 1
        assert result.duplicate_count == 0

    def test_duplicate_detection(self):
        """Similar issues from different reviewers are merged."""
        consolidator = ReviewConsolidator(similarity_threshold=0.6)

        source1 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability in auth.py:45",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=45,
                ),
            ],
            reviewer_name="claude",
            raw_text="Claude review",
        )

        source2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection issue in auth.py:45 - user input",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=45,
                ),
            ],
            reviewer_name="codex",
            raw_text="Codex review",
        )

        result = consolidator.consolidate([source1, source2])

        # Should be merged into 1 issue
        assert len(result.issues) == 1
        assert result.total_source_items == 2
        assert result.duplicate_count == 1
        assert result.issues[0].reviewer_count == 2

    def test_consensus_severity_voting(self):
        """Consensus severity is determined by majority vote."""
        consolidator = ReviewConsolidator(similarity_threshold=0.5)

        source1 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue X in file.py",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        source2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue X in file.py line 10",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="codex",
            raw_text="",
        )

        source3 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue X in file.py at line 10",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="gemini",
            raw_text="",
        )

        result = consolidator.consolidate([source1, source2, source3])

        # 2/3 voted HIGH, so consensus should be HIGH
        assert len(result.issues) == 1
        assert result.issues[0].severity == IssueSeverity.HIGH
        assert result.issues[0].severity_votes[IssueSeverity.HIGH] == 2
        assert result.issues[0].severity_votes[IssueSeverity.CRITICAL] == 1

    def test_fix_suggestions_aggregated(self):
        """Fix suggestions from all reviewers are collected."""
        consolidator = ReviewConsolidator(similarity_threshold=0.5)

        source1 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue X",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    fix_suggestion="Use parameterized queries",
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        source2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue X problem",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    fix_suggestion="Add input sanitization",
                ),
            ],
            reviewer_name="codex",
            raw_text="",
        )

        result = consolidator.consolidate([source1, source2])

        assert len(result.issues) == 1
        assert "parameterized" in result.issues[0].fix_suggestions[0].lower()
        assert "sanitization" in result.issues[0].fix_suggestions[1].lower()

    def test_different_issues_not_merged(self):
        """Distinct issues remain separate."""
        consolidator = ReviewConsolidator(similarity_threshold=0.8)

        source = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection in auth.py",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="N+1 query performance issue in api.py",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.PERFORMANCE,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        result = consolidator.consolidate([source])

        # Different issues should remain separate
        assert len(result.issues) == 2

    def test_issues_sorted_by_severity(self):
        """Consolidated issues are sorted by severity (critical first)."""
        # Use high threshold to prevent unintended merging
        consolidator = ReviewConsolidator(similarity_threshold=0.95)

        source = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Rename variable x to more descriptive name in utils.py",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.STYLE,
                    actionability=Actionability.NICE_TO_HAVE,
                    is_blocker=False,
                    file_path="utils.py",
                ),
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability in auth module line 45",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                ),
                ClassifiedFeedback(
                    original_text="N+1 query pattern in database access causing slowdown",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.PERFORMANCE,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=False,
                    file_path="db.py",
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        result = consolidator.consolidate([source])

        assert len(result.issues) == 3  # All 3 should be distinct
        assert result.issues[0].severity == IssueSeverity.CRITICAL
        assert result.issues[-1].severity == IssueSeverity.LOW

    def test_by_severity_grouping(self):
        """Issues are grouped by severity in by_severity dict."""
        # Use high threshold to prevent merging of distinct issues
        consolidator = ReviewConsolidator(similarity_threshold=0.95)

        source = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="SQL injection vulnerability found in authentication module",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                ),
                ClassifiedFeedback(
                    original_text="XSS vulnerability in user profile rendering template",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="profile.py",
                ),
                ClassifiedFeedback(
                    original_text="Null pointer exception when user data is missing",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.CORRECTNESS,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="user.py",
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        result = consolidator.consolidate([source])

        assert len(result.issues) == 3  # All 3 should be distinct
        assert result.critical_count == 2
        assert result.high_count == 1


class TestConsolidationResult:
    """Tests for ConsolidationResult properties."""

    def test_blocker_count(self):
        """blocker_count returns number of blocking issues."""
        result = ConsolidationResult(
            issues=[
                ConsolidatedIssue(
                    source_items=[
                        ClassifiedFeedback(
                            original_text="Blocker",
                            severity=IssueSeverity.CRITICAL,
                            category=IssueCategory.SECURITY,
                            actionability=Actionability.IMMEDIATE,
                            is_blocker=True,
                        ),
                        ClassifiedFeedback(
                            original_text="Blocker",
                            severity=IssueSeverity.CRITICAL,
                            category=IssueCategory.SECURITY,
                            actionability=Actionability.IMMEDIATE,
                            is_blocker=True,
                        ),
                    ],
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    message="Blocker issue",
                    reviewer_count=2,
                ),
                ConsolidatedIssue(
                    source_items=[
                        ClassifiedFeedback(
                            original_text="Non-blocker",
                            severity=IssueSeverity.LOW,
                            category=IssueCategory.STYLE,
                            actionability=Actionability.NICE_TO_HAVE,
                            is_blocker=False,
                        ),
                    ],
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.STYLE,
                    message="Non-blocker",
                    reviewer_count=1,
                ),
            ],
            source_results=[],
        )

        assert result.blocker_count == 1
        assert result.has_blockers is True

    def test_get_blocking_issues(self):
        """get_blocking_issues returns only blocking issues."""
        result = ConsolidationResult(
            issues=[
                ConsolidatedIssue(
                    source_items=[
                        ClassifiedFeedback(
                            original_text="Blocker",
                            severity=IssueSeverity.CRITICAL,
                            category=IssueCategory.SECURITY,
                            actionability=Actionability.IMMEDIATE,
                            is_blocker=True,
                        ),
                    ],
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    message="Blocker",
                    reviewer_count=1,
                ),
                ConsolidatedIssue(
                    source_items=[
                        ClassifiedFeedback(
                            original_text="Non-blocker",
                            severity=IssueSeverity.LOW,
                            category=IssueCategory.STYLE,
                            actionability=Actionability.NICE_TO_HAVE,
                            is_blocker=False,
                        ),
                    ],
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.STYLE,
                    message="Non-blocker",
                    reviewer_count=1,
                ),
            ],
            source_results=[],
        )

        blockers = result.get_blocking_issues()
        assert len(blockers) == 1
        assert blockers[0].severity == IssueSeverity.CRITICAL

    def test_to_classification_result(self):
        """Convert ConsolidationResult to ClassificationResult."""
        result = ConsolidationResult(
            issues=[
                ConsolidatedIssue(
                    source_items=[
                        ClassifiedFeedback(
                            original_text="Issue",
                            severity=IssueSeverity.HIGH,
                            category=IssueCategory.SECURITY,
                            actionability=Actionability.IMMEDIATE,
                            is_blocker=True,
                        ),
                    ],
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    message="Issue",
                    reviewer_count=1,
                ),
            ],
            source_results=[
                ClassificationResult(
                    feedback_items=[],
                    reviewer_name="claude",
                    raw_text="Claude's review",
                ),
            ],
            total_source_items=1,
            duplicate_count=0,
            consensus_rate=1.0,
        )

        classification = result.to_classification_result()

        assert len(classification.feedback_items) == 1
        assert classification.reviewer_name == "consolidated"
        assert "claude" in classification.raw_text.lower()
        assert classification.metadata["consolidated"] is True

    def test_summary_generation(self):
        """summary() generates human-readable summary."""
        result = ConsolidationResult(
            issues=[],
            source_results=[],
            total_source_items=10,
            duplicate_count=3,
            consensus_rate=0.85,
        )

        summary = result.summary()

        assert "10" in summary  # total source items
        assert "3" in summary  # duplicates
        assert "85" in summary or "0.85" in summary  # consensus rate


class TestConvenienceFunction:
    """Tests for consolidate_reviews convenience function."""

    def test_consolidate_reviews_function(self):
        """consolidate_reviews creates consolidator and runs."""
        source = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        result = consolidate_reviews([source])

        assert isinstance(result, ConsolidationResult)
        assert len(result.issues) == 1


class TestLocationGrouping:
    """Tests for location-based grouping."""

    def test_issues_grouped_by_file(self):
        """Issues in same file are more likely to match."""
        consolidator = ReviewConsolidator(
            similarity_threshold=0.5,
            location_weight=0.3,
            content_weight=0.7,
        )

        source1 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Input validation issue",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=10,
                ),
            ],
            reviewer_name="claude",
            raw_text="",
        )

        source2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Input validation problem",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                    file_path="auth.py",
                    line_number=12,
                ),
            ],
            reviewer_name="codex",
            raw_text="",
        )

        result = consolidator.consolidate([source1, source2])

        # Should be merged due to location proximity + content similarity
        assert len(result.issues) == 1
