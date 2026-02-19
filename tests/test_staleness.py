"""Tests for the Gatekeeper smart stale detection."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    IssueAuthor,
    IssueMetadata,
    PRAuthor,
    PRMetadata,
    StaleItem,
    StalenessReport,
)
from oss_maintainer_toolkit.gatekeeper.staleness import (
    _find_addressed_issues,
    _find_blocked_prs,
    _find_inactive_items,
    _find_superseded_prs,
    detect_stale_items,
)
from oss_maintainer_toolkit.gatekeeper.staleness_scorecard import (
    render_staleness_report,
    staleness_report_to_json,
)


_NOW = datetime.now(timezone.utc)


def _make_pr(
    number: int = 1,
    title: str = "Fix bug",
    body: str = "",
    linked_issues: list[int] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    merged_at: datetime | None = None,
    state: str = "open",
) -> PRMetadata:
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=PRAuthor(login="dev"),
        linked_issues=linked_issues or [],
        created_at=created_at,
        updated_at=updated_at,
        merged_at=merged_at,
        state=state,
    )


def _make_issue(
    number: int = 1,
    title: str = "Bug report",
    body: str = "",
    updated_at: datetime | None = None,
) -> IssueMetadata:
    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(login="user"),
        updated_at=updated_at,
    )


# ---- TestFindSupersededPrs ----


class TestFindSupersededPrs:
    def test_empty_open_prs(self):
        result = _find_superseded_prs([], [], [_make_pr()], [[1.0, 0.0]], 0.75)
        assert result == []

    def test_empty_merged_prs(self):
        result = _find_superseded_prs([_make_pr()], [[1.0, 0.0]], [], [], 0.75)
        assert result == []

    def test_above_threshold_flagged(self):
        open_pr = _make_pr(number=1, title="Add feature", created_at=_NOW - timedelta(days=30))
        merged_pr = _make_pr(
            number=2, title="Add same feature", state="closed",
            merged_at=_NOW - timedelta(days=5),
        )
        # Identical embeddings -> similarity = 1.0
        result = _find_superseded_prs(
            [open_pr], [[1.0, 0.0]], [merged_pr], [[1.0, 0.0]], 0.75,
        )
        assert len(result) == 1
        assert result[0].number == 1
        assert result[0].related_number == 2
        assert result[0].signal == "superseded"
        assert result[0].similarity >= 0.75

    def test_below_threshold_not_flagged(self):
        open_pr = _make_pr(number=1, created_at=_NOW - timedelta(days=30))
        merged_pr = _make_pr(number=2, merged_at=_NOW - timedelta(days=5))
        # Orthogonal embeddings -> similarity = 0.0
        result = _find_superseded_prs(
            [open_pr], [[1.0, 0.0]], [merged_pr], [[0.0, 1.0]], 0.75,
        )
        assert result == []

    def test_temporal_guard_merged_before_created(self):
        """Merged PR older than the open PR should not flag superseded."""
        open_pr = _make_pr(number=1, created_at=_NOW - timedelta(days=5))
        merged_pr = _make_pr(number=2, merged_at=_NOW - timedelta(days=30))
        # Identical embeddings but temporal guard should prevent flagging
        result = _find_superseded_prs(
            [open_pr], [[1.0, 0.0]], [merged_pr], [[1.0, 0.0]], 0.75,
        )
        assert result == []


# ---- TestFindAddressedIssues ----


class TestFindAddressedIssues:
    def test_empty_issues(self):
        result = _find_addressed_issues([], [], [_make_pr()], [[1.0, 0.0]], 0.75)
        assert result == []

    def test_empty_merged_prs(self):
        result = _find_addressed_issues([_make_issue()], [[1.0, 0.0]], [], [], 0.75)
        assert result == []

    def test_above_threshold_flagged(self):
        issue = _make_issue(number=10, title="Auth broken")
        merged_pr = _make_pr(number=2, title="Fix auth", merged_at=_NOW - timedelta(days=5))
        # Identical embeddings -> similarity = 1.0
        result = _find_addressed_issues(
            [issue], [[1.0, 0.0]], [merged_pr], [[1.0, 0.0]], 0.75,
        )
        assert len(result) == 1
        assert result[0].number == 10
        assert result[0].related_number == 2
        assert result[0].signal == "addressed"
        assert result[0].similarity >= 0.75

    def test_below_threshold_not_flagged(self):
        issue = _make_issue(number=10)
        merged_pr = _make_pr(number=2, merged_at=_NOW - timedelta(days=5))
        # Orthogonal embeddings -> similarity = 0.0
        result = _find_addressed_issues(
            [issue], [[1.0, 0.0]], [merged_pr], [[0.0, 1.0]], 0.75,
        )
        assert result == []


# ---- TestFindBlockedPrs ----


class TestFindBlockedPrs:
    def test_no_blocking_issues(self):
        pr = _make_pr(number=1, linked_issues=[])
        issue = _make_issue(number=10)
        result = _find_blocked_prs([pr], [issue])
        assert result == []

    def test_blocked_by_open_issue(self):
        pr = _make_pr(number=1, linked_issues=[10])
        issue = _make_issue(number=10)
        result = _find_blocked_prs([pr], [issue])
        assert len(result) == 1
        assert result[0].number == 1
        assert result[0].signal == "blocked"
        assert result[0].related_number == 10
        assert "#10" in result[0].explanation

    def test_linked_issue_not_in_open_set(self):
        """PR links to issue 99, but that issue isn't in the open set — not blocked."""
        pr = _make_pr(number=1, linked_issues=[99])
        issue = _make_issue(number=10)
        result = _find_blocked_prs([pr], [issue])
        assert result == []

    def test_multiple_blockers(self):
        pr = _make_pr(number=1, linked_issues=[10, 20])
        issues = [_make_issue(number=10), _make_issue(number=20)]
        result = _find_blocked_prs([pr], issues)
        assert len(result) == 1
        assert "#10" in result[0].explanation
        assert "#20" in result[0].explanation


# ---- TestFindInactiveItems ----


class TestFindInactiveItems:
    def test_recent_activity_not_flagged(self):
        pr = _make_pr(number=1, updated_at=_NOW - timedelta(days=5))
        issue = _make_issue(number=10, updated_at=_NOW - timedelta(days=5))
        inactive_prs, inactive_issues = _find_inactive_items([pr], [issue], 90)
        assert inactive_prs == []
        assert inactive_issues == []

    def test_inactive_pr_flagged(self):
        pr = _make_pr(number=1, updated_at=_NOW - timedelta(days=120))
        inactive_prs, _ = _find_inactive_items([pr], [], 90)
        assert len(inactive_prs) == 1
        assert inactive_prs[0].number == 1
        assert inactive_prs[0].signal == "inactive"

    def test_inactive_issue_flagged(self):
        issue = _make_issue(number=10, updated_at=_NOW - timedelta(days=120))
        _, inactive_issues = _find_inactive_items([], [issue], 90)
        assert len(inactive_issues) == 1
        assert inactive_issues[0].number == 10
        assert inactive_issues[0].signal == "inactive"

    def test_no_updated_at_not_flagged(self):
        """Items without updated_at should not be flagged."""
        pr = _make_pr(number=1, updated_at=None)
        issue = _make_issue(number=10, updated_at=None)
        inactive_prs, inactive_issues = _find_inactive_items([pr], [issue], 90)
        assert inactive_prs == []
        assert inactive_issues == []


# ---- TestDetectStaleItems ----


class TestDetectStaleItems:
    def test_report_metadata(self):
        open_pr = _make_pr(number=1, created_at=_NOW - timedelta(days=30))
        open_issue = _make_issue(number=10)
        merged_pr = _make_pr(number=2, merged_at=_NOW - timedelta(days=5))

        report = detect_stale_items(
            [open_pr], [[1.0, 0.0]],
            [open_issue], [[0.0, 1.0]],
            [merged_pr], [[0.0, 1.0]],
            threshold=0.75,
            inactive_days=90,
        )
        assert report.owner == "owner"
        assert report.repo == "repo"
        assert report.total_open_prs == 1
        assert report.total_open_issues == 1
        assert report.total_merged_prs_checked == 1
        assert report.threshold == 0.75
        assert report.inactive_days == 90

    def test_config_default_threshold(self):
        """Threshold 0 falls back to config default (0.75)."""
        report = detect_stale_items([], [], [], [], [], [], threshold=0, inactive_days=0)
        assert report.threshold == 0.75
        assert report.inactive_days == 90


# ---- TestStalenessScorecard ----


class TestStalenessScorecard:
    def test_json_serialization(self):
        report = StalenessReport(
            owner="owner",
            repo="repo",
            superseded_prs=[
                StaleItem(
                    item_type="pr", number=1, title="Old PR",
                    signal="superseded", related_number=2,
                    similarity=0.85, explanation="superseded",
                ),
            ],
            total_open_prs=5,
            total_open_issues=3,
            total_merged_prs_checked=10,
            threshold=0.75,
            inactive_days=90,
        )
        json_str = staleness_report_to_json(report)
        data = json.loads(json_str)
        assert data["owner"] == "owner"
        assert data["total_open_prs"] == 5
        assert len(data["superseded_prs"]) == 1
        assert data["superseded_prs"][0]["similarity"] == 0.85
        assert data["threshold"] == 0.75

    def test_rich_rendering_with_data(self):
        console = Console(record=True, width=120)
        report = StalenessReport(
            owner="owner",
            repo="repo",
            superseded_prs=[
                StaleItem(
                    item_type="pr", number=1, title="Old PR",
                    signal="superseded", related_number=2,
                    related_title="New PR", similarity=0.85,
                    explanation="PR #1 is 85% similar to merged PR #2",
                ),
            ],
            addressed_issues=[
                StaleItem(
                    item_type="issue", number=10, title="Bug",
                    signal="addressed", related_number=3,
                    similarity=0.80,
                    explanation="Issue #10 is 80% similar to merged PR #3",
                ),
            ],
            blocked_prs=[
                StaleItem(
                    item_type="pr", number=4, title="Blocked PR",
                    signal="blocked", related_number=11,
                    explanation="PR #4 is blocked by open issue(s): #11.",
                ),
            ],
            inactive_prs=[
                StaleItem(
                    item_type="pr", number=5, title="Stale PR",
                    signal="inactive",
                    last_activity=datetime(2025, 6, 1, tzinfo=timezone.utc),
                    explanation="No activity since 2025-06-01",
                ),
            ],
            inactive_issues=[
                StaleItem(
                    item_type="issue", number=12, title="Old Issue",
                    signal="inactive",
                    last_activity=datetime(2025, 5, 15, tzinfo=timezone.utc),
                    explanation="No activity since 2025-05-15",
                ),
            ],
            total_open_prs=10,
            total_open_issues=8,
            total_merged_prs_checked=20,
            threshold=0.75,
            inactive_days=90,
        )
        render_staleness_report(report, console)
        output = console.export_text()
        assert "Smart Stale Detection" in output
        assert "owner/repo" in output
        assert "Superseded PRs" in output
        assert "#1" in output
        assert "#2" in output
        assert "Already-Addressed Issues" in output
        assert "#10" in output
        assert "Blocked PRs" in output
        assert "#4" in output
        assert "Inactive PRs" in output
        assert "#5" in output
        assert "Inactive Issues" in output
        assert "#12" in output

    def test_empty_report_rendering(self):
        console = Console(record=True, width=120)
        report = StalenessReport(
            owner="owner",
            repo="repo",
            threshold=0.75,
            inactive_days=90,
        )
        render_staleness_report(report, console)
        output = console.export_text()
        assert "Smart Stale Detection" in output
        assert "No stale items detected" in output
