"""Tests for contributor badge logic in action_entrypoint.py."""

from action_entrypoint import (
    _badge_section,
    _format_comment,
    _format_issue_comment,
    _qualifies_for_badge,
)
from oss_maintainer_toolkit.gatekeeper.models import (
    AssessmentScorecard,
    DimensionScore,
    FlagSeverity,
    IssueScorecard,
    SuspicionFlag,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)


def _make_pr_scorecard(
    verdict=Verdict.FAST_TRACK,
    flags=None,
    vision_result=None,
    summary="Test summary",
):
    return AssessmentScorecard(
        owner="owner",
        repo="repo",
        pr_number=42,
        verdict=verdict,
        confidence=0.9,
        dimensions=[DimensionScore(dimension="test", score=0.9, summary="ok")],
        flags=flags or [],
        vision_result=vision_result,
        summary=summary,
    )


def _make_issue_scorecard(
    verdict=Verdict.FAST_TRACK,
    flags=None,
    vision_result=None,
    summary="Test summary",
):
    return IssueScorecard(
        owner="owner",
        repo="repo",
        issue_number=101,
        verdict=verdict,
        confidence=0.9,
        dimensions=[DimensionScore(dimension="test", score=0.9, summary="ok")],
        flags=flags or [],
        vision_result=vision_result,
        summary=summary,
    )


def _make_flag():
    return SuspicionFlag(
        rule_id="test_flag",
        severity=FlagSeverity.MEDIUM,
        title="Test Flag",
        explanation="A test flag",
    )


def _make_vision_result(score=0.9):
    return VisionAlignmentResult(
        outcome=TierOutcome.PASS,
        alignment_score=score,
        strengths=["Good alignment"],
    )


class TestQualifiesForBadge:
    def test_fast_track_no_flags_qualifies(self):
        sc = _make_pr_scorecard()
        assert _qualifies_for_badge(sc) is True

    def test_review_required_does_not_qualify(self):
        sc = _make_pr_scorecard(verdict=Verdict.REVIEW_REQUIRED)
        assert _qualifies_for_badge(sc) is False

    def test_recommend_close_does_not_qualify(self):
        sc = _make_pr_scorecard(verdict=Verdict.RECOMMEND_CLOSE)
        assert _qualifies_for_badge(sc) is False

    def test_fast_track_with_flags_does_not_qualify(self):
        sc = _make_pr_scorecard(flags=[_make_flag()])
        assert _qualifies_for_badge(sc) is False

    def test_fast_track_high_vision_qualifies(self):
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.85))
        assert _qualifies_for_badge(sc) is True

    def test_fast_track_low_vision_does_not_qualify(self):
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.5))
        assert _qualifies_for_badge(sc) is False

    def test_fast_track_no_vision_qualifies(self):
        """When Tier 3 didn't run, badge is still awarded."""
        sc = _make_pr_scorecard(vision_result=None)
        assert _qualifies_for_badge(sc) is True

    def test_fast_track_vision_at_threshold(self):
        """Alignment exactly at 0.8 qualifies."""
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.8))
        assert _qualifies_for_badge(sc) is True

    def test_fast_track_vision_just_below_threshold(self):
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.79))
        assert _qualifies_for_badge(sc) is False


class TestBadgeSection:
    def test_includes_trophy_emoji(self):
        sc = _make_pr_scorecard()
        lines = _badge_section(sc)
        assert any("1F3C6" in line for line in lines)

    def test_includes_vision_score_when_high(self):
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.9))
        lines = _badge_section(sc)
        badge_text = "\n".join(lines)
        assert "90%" in badge_text

    def test_no_vision_score_without_result(self):
        sc = _make_pr_scorecard(vision_result=None)
        lines = _badge_section(sc)
        badge_text = "\n".join(lines)
        assert "Vision alignment" not in badge_text


class TestFormatCommentWithBadge:
    def test_badge_appears_for_exemplary_pr(self):
        sc = _make_pr_scorecard(vision_result=_make_vision_result(0.9))
        comment = _format_comment(sc)
        assert "Exemplary Contribution" in comment
        assert "FAST TRACK" in comment

    def test_no_badge_for_review_required(self):
        sc = _make_pr_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=[_make_flag()],
        )
        comment = _format_comment(sc)
        assert "Exemplary" not in comment
        assert "REVIEW REQUIRED" in comment

    def test_no_badge_for_flagged_fast_track(self):
        sc = _make_pr_scorecard(flags=[_make_flag()])
        comment = _format_comment(sc)
        assert "Exemplary" not in comment

    def test_badge_before_verdict(self):
        """Badge section should appear before the verdict section."""
        sc = _make_pr_scorecard()
        comment = _format_comment(sc)
        badge_pos = comment.find("Exemplary")
        verdict_pos = comment.find("FAST TRACK")
        assert badge_pos < verdict_pos


class TestFormatIssueCommentWithBadge:
    def test_badge_appears_for_exemplary_issue(self):
        sc = _make_issue_scorecard()
        comment = _format_issue_comment(sc)
        assert "Well-Formed Issue" in comment
        assert "FAST TRACK" in comment

    def test_no_badge_for_flagged_issue(self):
        sc = _make_issue_scorecard(flags=[_make_flag()])
        comment = _format_issue_comment(sc)
        assert "Well-Formed" not in comment
