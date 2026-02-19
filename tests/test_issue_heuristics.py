"""Tests for the Gatekeeper issue heuristics (Tier 2)."""

from datetime import datetime, timedelta, timezone

import pytest

from oss_maintainer_toolkit.gatekeeper.models import (
    FlagSeverity,
    IssueAuthor,
    IssueMetadata,
    TierOutcome,
)
from oss_maintainer_toolkit.gatekeeper.issue_heuristics import (
    check_all_caps_title,
    check_first_contribution,
    check_missing_reproduction,
    check_new_account,
    check_short_title,
    check_temporal_clustering,
    check_vague_description,
    run_issue_heuristics,
)


def _make_issue(
    number: int = 1,
    title: str = "A reasonable issue title",
    body: str = "This is a detailed description of the issue with enough chars.",
    login: str = "testuser",
    account_age_days: int | None = None,
    contributions: int = 5,
    labels: list[str] | None = None,
    created_at: datetime | None = None,
) -> IssueMetadata:
    account_created = None
    if account_age_days is not None:
        account_created = datetime.now(timezone.utc) - timedelta(days=account_age_days)

    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(
            login=login,
            account_created_at=account_created,
            contributions_to_repo=contributions,
        ),
        labels=labels or [],
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestCheckVagueDescription:
    def test_short_body_flagged(self):
        issue = _make_issue(body="help")
        flag = check_vague_description(issue)
        assert flag is not None
        assert flag.rule_id == "vague_description"
        assert flag.severity == FlagSeverity.MEDIUM

    def test_empty_body_flagged(self):
        issue = _make_issue(body="")
        flag = check_vague_description(issue)
        assert flag is not None

    def test_adequate_body_passes(self):
        issue = _make_issue(body="This is a sufficiently detailed description of the problem.")
        flag = check_vague_description(issue)
        assert flag is None


class TestCheckNewAccount:
    def test_new_account_flagged(self):
        issue = _make_issue(account_age_days=10)
        flag = check_new_account(issue)
        assert flag is not None
        assert flag.rule_id == "new_account"

    def test_old_account_passes(self):
        issue = _make_issue(account_age_days=200)
        flag = check_new_account(issue)
        assert flag is None

    def test_no_account_date_passes(self):
        issue = _make_issue(account_age_days=None)
        flag = check_new_account(issue)
        assert flag is None


class TestCheckFirstContribution:
    def test_first_contribution_flagged(self):
        issue = _make_issue(contributions=0)
        flag = check_first_contribution(issue)
        assert flag is not None
        assert flag.rule_id == "first_contribution"
        assert flag.severity == FlagSeverity.LOW

    def test_existing_contributor_passes(self):
        issue = _make_issue(contributions=5)
        flag = check_first_contribution(issue)
        assert flag is None


class TestCheckMissingReproduction:
    def test_bug_without_repro_flagged(self):
        issue = _make_issue(
            title="Bug: app crashes",
            body="The app crashes randomly.",
            labels=["bug"],
        )
        flag = check_missing_reproduction(issue)
        assert flag is not None
        assert flag.rule_id == "missing_reproduction"

    def test_bug_with_repro_passes(self):
        issue = _make_issue(
            title="Bug: app crashes",
            body="Steps to reproduce:\n1. Open app\n2. Click button\n\nExpected: no crash",
        )
        flag = check_missing_reproduction(issue)
        assert flag is None

    def test_bug_with_code_snippet_passes(self):
        issue = _make_issue(
            title="Bug in parser",
            body="Error when parsing:\n```\nparser.parse(data)\n```",
        )
        flag = check_missing_reproduction(issue)
        assert flag is None

    def test_feature_request_not_flagged(self):
        issue = _make_issue(
            title="Feature request: dark mode",
            body="It would be nice to have dark mode.",
            labels=["enhancement"],
        )
        flag = check_missing_reproduction(issue)
        assert flag is None

    def test_bug_keyword_in_body_flagged(self):
        issue = _make_issue(
            title="Something weird",
            body="This is a bug that crashes the app",
        )
        flag = check_missing_reproduction(issue)
        assert flag is not None


class TestCheckShortTitle:
    def test_short_title_flagged(self):
        issue = _make_issue(title="help")
        flag = check_short_title(issue)
        assert flag is not None
        assert flag.rule_id == "short_title"

    def test_adequate_title_passes(self):
        issue = _make_issue(title="Login page crashes on submit")
        flag = check_short_title(issue)
        assert flag is None


class TestCheckAllCapsTitle:
    def test_all_caps_flagged(self):
        issue = _make_issue(title="THIS IS ALL CAPS TITLE")
        flag = check_all_caps_title(issue)
        assert flag is not None
        assert flag.rule_id == "all_caps_title"

    def test_normal_title_passes(self):
        issue = _make_issue(title="Normal Title Here")
        flag = check_all_caps_title(issue)
        assert flag is None

    def test_short_caps_not_flagged(self):
        """Titles with <5 letters aren't flagged for caps."""
        issue = _make_issue(title="BUG")
        flag = check_all_caps_title(issue)
        assert flag is None


class TestCheckTemporalClustering:
    def test_no_recent_issues(self):
        issue = _make_issue()
        flag = check_temporal_clustering(issue, None)
        assert flag is None

    def test_clustering_flagged(self):
        now = datetime.now(timezone.utc)
        target = _make_issue(number=1, created_at=now, account_age_days=10)

        recent = [
            _make_issue(number=i, created_at=now - timedelta(hours=1), account_age_days=5, login=f"user{i}")
            for i in range(2, 6)
        ]

        flag = check_temporal_clustering(target, recent)
        assert flag is not None
        assert flag.rule_id == "temporal_clustering"
        assert flag.severity == FlagSeverity.HIGH

    def test_no_clustering_with_old_accounts(self):
        now = datetime.now(timezone.utc)
        target = _make_issue(number=1, created_at=now, account_age_days=200)

        recent = [
            _make_issue(number=i, created_at=now - timedelta(hours=1), account_age_days=200, login=f"user{i}")
            for i in range(2, 6)
        ]

        flag = check_temporal_clustering(target, recent)
        assert flag is None


class TestRunIssueHeuristics:
    def test_clean_issue_passes(self):
        issue = _make_issue(
            title="Clear and detailed bug report",
            body="This is a very detailed description of the issue that should pass all heuristic checks.",
            account_age_days=200,
            contributions=5,
        )
        result = run_issue_heuristics(issue)
        assert result.outcome == TierOutcome.PASS
        assert result.suspicion_score < 0.6

    def test_suspicious_issue_gated(self):
        """Issue with multiple flags should be gated."""
        issue = _make_issue(
            title="help",
            body="",
            account_age_days=5,
            contributions=0,
            labels=["bug"],
        )
        result = run_issue_heuristics(issue, threshold=0.3)
        assert result.outcome == TierOutcome.GATED
        assert len(result.flags) >= 3

    def test_score_capped_at_one(self):
        """Score should never exceed 1.0."""
        issue = _make_issue(
            title="BUG",
            body="",
            account_age_days=1,
            contributions=0,
            labels=["bug"],
        )
        result = run_issue_heuristics(issue, threshold=0.01)
        assert result.suspicion_score <= 1.0

    def test_uses_config_default_threshold(self):
        """Threshold 0 falls back to config default (0.6)."""
        issue = _make_issue()
        result = run_issue_heuristics(issue, threshold=0)
        # Clean issue should pass with default threshold
        assert result.outcome == TierOutcome.PASS
