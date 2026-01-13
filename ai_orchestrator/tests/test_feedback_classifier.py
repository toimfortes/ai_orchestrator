"""Tests for FeedbackClassifier."""

import pytest

from ai_orchestrator.reviewing.feedback_classifier import (
    Actionability,
    ClassifiedFeedback,
    ClassificationResult,
    FeedbackClassifier,
    IssueCategory,
    IssueSeverity,
    merge_classifications,
)


class TestFeedbackClassifier:
    """Tests for FeedbackClassifier."""

    def test_classify_critical_security_issue(self):
        """Test classification of critical security issue."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "SQL injection vulnerability in auth.py:45 - user input not sanitized",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity == IssueSeverity.CRITICAL
        assert item.category == IssueCategory.SECURITY
        assert item.is_blocker is True
        assert item.actionability == Actionability.IMMEDIATE

    def test_classify_high_severity_bug(self):
        """Test classification of high severity bug."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Bug: function returns incorrect value when input is negative",
            reviewer_name="codex",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity == IssueSeverity.HIGH
        assert item.category == IssueCategory.CORRECTNESS
        assert item.is_blocker is True

    def test_classify_low_severity_suggestion(self):
        """Test classification of low severity suggestion."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Nitpick: Consider renaming 'x' to a more descriptive name",
            reviewer_name="gemini",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity == IssueSeverity.LOW
        assert item.is_blocker is False
        assert item.actionability == Actionability.NICE_TO_HAVE

    def test_classify_performance_issue(self):
        """Test classification of performance issue."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Performance: N+1 query issue in get_users() causing slow response",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.category == IssueCategory.PERFORMANCE

    def test_classify_multiple_items(self):
        """Test classification of multiple feedback items."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            """
            1. Critical: XSS vulnerability in user input handling
            2. Bug: Edge case not handled for empty arrays
            3. Suggestion: Add more unit tests for error paths
            """,
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 3
        assert result.critical_count == 1
        assert result.has_blockers is True

    def test_extract_file_location(self):
        """Test extraction of file location from feedback."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Error in auth/login.py:123 - missing null check",
            reviewer_name="codex",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.file_path == "auth/login.py"
        assert item.line_number == 123
        assert item.location == "auth/login.py:123"

    def test_extract_file_location_alternative_format(self):
        """Test extraction of file location with 'at file line N' format."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Issue at `utils/helpers.py` line 45",
            reviewer_name="gemini",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.file_path == "utils/helpers.py"
        assert item.line_number == 45

    def test_classification_result_properties(self):
        """Test ClassificationResult computed properties."""
        result = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Critical issue",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="High issue",
                    severity=IssueSeverity.HIGH,
                    category=IssueCategory.CORRECTNESS,
                    actionability=Actionability.IMMEDIATE,
                    is_blocker=True,
                ),
                ClassifiedFeedback(
                    original_text="Low issue",
                    severity=IssueSeverity.LOW,
                    category=IssueCategory.STYLE,
                    actionability=Actionability.NICE_TO_HAVE,
                    is_blocker=False,
                ),
            ],
            reviewer_name="test",
            raw_text="test feedback",
        )

        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.blocker_count == 2
        assert result.has_blockers is True
        assert len(result.get_by_severity(IssueSeverity.LOW)) == 1
        assert len(result.get_by_category(IssueCategory.SECURITY)) == 1

    def test_merge_classifications(self):
        """Test merging multiple classification results."""
        result1 = ClassificationResult(
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
            raw_text="Claude's feedback",
        )

        result2 = ClassificationResult(
            feedback_items=[
                ClassifiedFeedback(
                    original_text="Issue 2",
                    severity=IssueSeverity.MEDIUM,
                    category=IssueCategory.PERFORMANCE,
                    actionability=Actionability.SHOULD_FIX,
                    is_blocker=False,
                ),
            ],
            reviewer_name="codex",
            raw_text="Codex's feedback",
        )

        merged = merge_classifications([result1, result2])

        assert len(merged.feedback_items) == 2
        assert merged.reviewer_name == "merged"
        assert "claude" in merged.metadata.get("source_reviewers", [])
        assert "codex" in merged.metadata.get("source_reviewers", [])

    def test_empty_feedback(self):
        """Test classification of empty feedback."""
        classifier = FeedbackClassifier()
        result = classifier.classify("", reviewer_name="test")

        assert len(result.feedback_items) == 0
        assert result.critical_count == 0
        assert result.has_blockers is False

    def test_extract_suggestion(self):
        """Test extraction of fix suggestions."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Performance issue. Consider using a cache to improve response time.",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.fix_suggestion is not None
        assert "cache" in item.fix_suggestion.lower()


class TestClassifiedFeedback:
    """Tests for ClassifiedFeedback dataclass."""

    def test_location_with_file_and_line(self):
        """Test location property with both file and line."""
        feedback = ClassifiedFeedback(
            original_text="test",
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.OTHER,
            actionability=Actionability.SHOULD_FIX,
            is_blocker=False,
            file_path="test.py",
            line_number=42,
        )

        assert feedback.location == "test.py:42"

    def test_location_with_file_only(self):
        """Test location property with file only."""
        feedback = ClassifiedFeedback(
            original_text="test",
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.OTHER,
            actionability=Actionability.SHOULD_FIX,
            is_blocker=False,
            file_path="test.py",
            line_number=None,
        )

        assert feedback.location == "test.py"

    def test_location_without_file(self):
        """Test location property without file."""
        feedback = ClassifiedFeedback(
            original_text="test",
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.OTHER,
            actionability=Actionability.SHOULD_FIX,
            is_blocker=False,
            file_path=None,
            line_number=None,
        )

        assert feedback.location is None


class TestNegationAwareClassification:
    """Tests for negation-aware keyword classification.

    The classifier should recognize when keywords are negated and
    avoid false positives like classifying "no security issues" as CRITICAL.
    """

    def test_negated_security_not_classified_critical(self):
        """Test that 'no security issues' is not classified as CRITICAL."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "LGTM - no security issues found in this code",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        # Should NOT be critical because "security" is negated
        assert item.severity != IssueSeverity.CRITICAL

    def test_negated_vulnerability_not_critical(self):
        """Test that 'not a vulnerability' is not classified as CRITICAL."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "This pattern is not a vulnerability in this context",
            reviewer_name="codex",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity != IssueSeverity.CRITICAL

    def test_negated_bug_not_high_severity(self):
        """Test that 'no bugs' is not classified as HIGH severity."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Code review complete - no bugs detected",
            reviewer_name="gemini",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity not in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)

    def test_without_issues_not_critical(self):
        """Test that 'without issues' pattern is recognized."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "The authentication module is implemented without security issues",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        # "without security issues" should not match security keywords
        assert item.severity != IssueSeverity.CRITICAL

    def test_affirmed_security_issue_is_critical(self):
        """Test that actual security issues are still classified correctly."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Found security vulnerability: SQL injection in user input",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        # This SHOULD be critical - "found" indicates actual issue
        assert item.severity == IssueSeverity.CRITICAL
        assert item.category == IssueCategory.SECURITY

    def test_category_negation_awareness(self):
        """Test that category detection respects negation."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "No performance problems in this implementation",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        # Should NOT be categorized as PERFORMANCE since it's negated
        # Will fall back to OTHER
        assert item.category != IssueCategory.PERFORMANCE

    def test_complex_negation_free_of(self):
        """Test 'free of' negation pattern."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            "Code is free of critical bugs and vulnerabilities",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity != IssueSeverity.CRITICAL

    def test_mixed_positive_and_negated(self):
        """Test mixed feedback with both positive and negated issues."""
        classifier = FeedbackClassifier()
        result = classifier.classify(
            """
            1. No SQL injection issues found
            2. Critical: XSS vulnerability in user display
            """,
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 2

        # First item (negated) should not be critical
        # Second item (actual issue) should be critical
        critical_items = [i for i in result.feedback_items if i.severity == IssueSeverity.CRITICAL]
        assert len(critical_items) == 1
        assert "XSS" in critical_items[0].original_text

    def test_contraction_negation(self):
        """Test negation with contractions like 'isn't', 'doesn't'."""
        classifier = FeedbackClassifier()

        result = classifier.classify(
            "This code doesn't have any security vulnerabilities",
            reviewer_name="claude",
        )

        assert len(result.feedback_items) == 1
        item = result.feedback_items[0]

        assert item.severity != IssueSeverity.CRITICAL
