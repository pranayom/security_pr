"""Tests for the PR coaching bot — flag-to-coaching mapping and comment integration."""

from action_entrypoint import _format_comment, _format_issue_comment
from oss_maintainer_toolkit.gatekeeper.coaching import (
    COACHING_ADVICE,
    build_flag_coaching,
    build_vision_coaching,
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


# --- Helpers ---

def _make_flag(rule_id: str, evidence: str = "", severity=FlagSeverity.MEDIUM, title: str = ""):
    return SuspicionFlag(
        rule_id=rule_id,
        severity=severity,
        title=title or rule_id.replace("_", " ").title(),
        explanation="Test explanation",
        evidence=evidence,
    )


def _make_pr_scorecard(
    verdict=Verdict.REVIEW_REQUIRED,
    flags=None,
    vision_result=None,
    summary="Test summary",
):
    return AssessmentScorecard(
        owner="owner",
        repo="repo",
        pr_number=42,
        verdict=verdict,
        confidence=0.8,
        dimensions=[DimensionScore(dimension="test", score=0.5, summary="ok")],
        flags=flags or [],
        vision_result=vision_result,
        summary=summary,
    )


def _make_issue_scorecard(
    verdict=Verdict.REVIEW_REQUIRED,
    flags=None,
    vision_result=None,
    summary="Test summary",
):
    return IssueScorecard(
        owner="owner",
        repo="repo",
        issue_number=101,
        verdict=verdict,
        confidence=0.8,
        dimensions=[DimensionScore(dimension="test", score=0.5, summary="ok")],
        flags=flags or [],
        vision_result=vision_result,
        summary=summary,
    )


def _make_vision_result(
    score=0.4,
    violated_principles=None,
    concerns=None,
    strengths=None,
):
    return VisionAlignmentResult(
        outcome=TierOutcome.PASS,
        alignment_score=score,
        violated_principles=violated_principles or [],
        concerns=concerns or [],
        strengths=strengths or [],
    )


# --- build_flag_coaching ---

class TestBuildFlagCoaching:
    def test_known_rule_produces_coaching(self):
        flags = [_make_flag("low_test_ratio", evidence="code=50, test=2")]
        result = build_flag_coaching(flags)
        assert len(result) == 1
        assert "Add tests" in result[0]

    def test_evidence_substitution(self):
        flags = [_make_flag("sensitive_paths", evidence="auth.py, login.py")]
        result = build_flag_coaching(flags)
        assert "auth.py, login.py" in result[0]

    def test_empty_evidence_removes_placeholder(self):
        flags = [_make_flag("sensitive_paths", evidence="")]
        result = build_flag_coaching(flags)
        assert "{evidence}" not in result[0]

    def test_unknown_rule_skipped(self):
        flags = [_make_flag("some_unknown_rule_xyz")]
        result = build_flag_coaching(flags)
        assert result == []

    def test_multiple_flags_produce_multiple_bullets(self):
        flags = [
            _make_flag("new_account"),
            _make_flag("first_contribution"),
            _make_flag("low_test_ratio", evidence="code=100, test=0"),
        ]
        result = build_flag_coaching(flags)
        assert len(result) == 3

    def test_each_rule_id_has_advice(self):
        """Verify all known rule_ids produce coaching text."""
        for rule_id in COACHING_ADVICE:
            flags = [_make_flag(rule_id, evidence="test_evidence")]
            result = build_flag_coaching(flags)
            assert len(result) == 1, f"No coaching for {rule_id}"
            assert result[0].startswith("- **"), f"Bad format for {rule_id}"

    def test_new_account_advice(self):
        result = build_flag_coaching([_make_flag("new_account")])
        assert "smaller PRs" in result[0]

    def test_first_contribution_advice(self):
        result = build_flag_coaching([_make_flag("first_contribution")])
        assert "CONTRIBUTING.md" in result[0]

    def test_unjustified_deps_advice(self):
        result = build_flag_coaching([_make_flag("unjustified_deps", evidence="package.json")])
        assert "package.json" in result[0]
        assert "justification" in result[0].lower()

    def test_large_diff_hiding_advice(self):
        result = build_flag_coaching([_make_flag("large_diff_hiding", evidence="1200 changes")])
        assert "splitting" in result[0].lower()

    def test_temporal_clustering_advice(self):
        result = build_flag_coaching([_make_flag("temporal_clustering")])
        assert "community channels" in result[0]

    def test_vague_description_advice(self):
        result = build_flag_coaching([_make_flag("vague_description")])
        assert "detail" in result[0].lower()

    def test_missing_reproduction_advice(self):
        result = build_flag_coaching([_make_flag("missing_reproduction")])
        assert "reproduction" in result[0].lower() or "step-by-step" in result[0].lower()


# --- build_vision_coaching ---

class TestBuildVisionCoaching:
    def test_violated_principles_produce_coaching(self):
        vr = _make_vision_result(violated_principles=["Architecture-First Contribution"])
        result = build_vision_coaching(vr)
        assert len(result) == 1
        assert "Architecture-First Contribution" in result[0]

    def test_concerns_produce_coaching(self):
        vr = _make_vision_result(concerns=["This PR modifies credential handling without security tests"])
        result = build_vision_coaching(vr)
        assert len(result) == 1
        assert "credential handling" in result[0]

    def test_both_principles_and_concerns(self):
        vr = _make_vision_result(
            violated_principles=["Modularity"],
            concerns=["Tightly coupled to core"],
        )
        result = build_vision_coaching(vr)
        assert len(result) == 2

    def test_no_violations_no_concerns_empty(self):
        vr = _make_vision_result(strengths=["Good work"])
        result = build_vision_coaching(vr)
        assert result == []

    def test_strengths_not_included(self):
        vr = _make_vision_result(strengths=["Well tested"])
        result = build_vision_coaching(vr)
        assert result == []


# --- Comment integration ---

class TestPRCommentCoaching:
    def test_coaching_appears_for_review_required(self):
        flags = [_make_flag("low_test_ratio", evidence="code=50, test=0")]
        sc = _make_pr_scorecard(verdict=Verdict.REVIEW_REQUIRED, flags=flags)
        comment = _format_comment(sc)
        assert "### How to Improve" in comment
        assert "Add tests" in comment

    def test_no_coaching_for_fast_track(self):
        flags = [_make_flag("first_contribution")]
        sc = _make_pr_scorecard(verdict=Verdict.FAST_TRACK, flags=flags)
        comment = _format_comment(sc)
        assert "How to Improve" not in comment

    def test_coaching_appears_for_recommend_close(self):
        flags = [_make_flag("new_account")]
        sc = _make_pr_scorecard(verdict=Verdict.RECOMMEND_CLOSE, flags=flags)
        comment = _format_comment(sc)
        assert "### How to Improve" in comment

    def test_vision_coaching_in_pr_comment(self):
        vr = _make_vision_result(
            violated_principles=["Security-First"],
            concerns=["No security tests"],
        )
        flags = [_make_flag("sensitive_paths", evidence="auth.py")]
        sc = _make_pr_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=flags,
            vision_result=vr,
        )
        comment = _format_comment(sc)
        assert "governance principles" in comment
        assert "Security-First" in comment

    def test_coaching_section_before_vision_details(self):
        """Coaching section should appear before the Vision Alignment details."""
        vr = _make_vision_result(
            score=0.3,
            violated_principles=["Modularity"],
        )
        flags = [_make_flag("low_test_ratio", evidence="code=50, test=0")]
        sc = _make_pr_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=flags,
            vision_result=vr,
        )
        comment = _format_comment(sc)
        improve_pos = comment.find("How to Improve")
        vision_pos = comment.find("### Vision Alignment")
        assert improve_pos > 0
        assert vision_pos > 0
        assert improve_pos < vision_pos


class TestIssueCommentCoaching:
    def test_coaching_appears_for_review_required_issue(self):
        flags = [_make_flag("vague_description")]
        sc = _make_issue_scorecard(verdict=Verdict.REVIEW_REQUIRED, flags=flags)
        comment = _format_issue_comment(sc)
        assert "### How to Improve" in comment
        assert "detail" in comment.lower()

    def test_no_coaching_for_fast_track_issue(self):
        sc = _make_issue_scorecard(verdict=Verdict.FAST_TRACK)
        comment = _format_issue_comment(sc)
        assert "How to Improve" not in comment

    def test_vision_coaching_in_issue_comment(self):
        vr = _make_vision_result(
            concerns=["Feature request outside project scope"],
        )
        flags = [_make_flag("vague_description")]
        sc = _make_issue_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=flags,
            vision_result=vr,
        )
        comment = _format_issue_comment(sc)
        assert "governance principles" in comment
        assert "outside project scope" in comment
