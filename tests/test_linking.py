"""Tests for the Gatekeeper issue-to-PR linking (Tier 1)."""

import json
import math

import numpy as np
import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    IssueAuthor,
    IssueMetadata,
    LinkingReport,
    LinkSuggestion,
    PRAuthor,
    PRMetadata,
)
from oss_maintainer_toolkit.gatekeeper.linking import (
    _compute_similarity_matrix,
    find_issue_pr_links,
)
from oss_maintainer_toolkit.gatekeeper.linking_scorecard import (
    linking_report_to_json,
    render_linking_report,
)


def _make_pr(number: int = 1, title: str = "Fix bug", body: str = "", linked_issues: list[int] | None = None) -> PRMetadata:
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=PRAuthor(login="dev"),
        linked_issues=linked_issues or [],
    )


def _make_issue(number: int = 1, title: str = "Bug report", body: str = "", labels: list[str] | None = None) -> IssueMetadata:
    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(login="user"),
        labels=labels or [],
    )


class TestComputeSimilarityMatrix:
    def test_empty_pr_embeddings(self):
        result = _compute_similarity_matrix([], [[1.0, 0.0]])
        assert result.shape == (0, 0)

    def test_empty_issue_embeddings(self):
        result = _compute_similarity_matrix([[1.0, 0.0]], [])
        assert result.shape == (0, 0)

    def test_identical_vectors(self):
        pr_embs = [[1.0, 0.0]]
        issue_embs = [[1.0, 0.0]]
        result = _compute_similarity_matrix(pr_embs, issue_embs)
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        pr_embs = [[1.0, 0.0]]
        issue_embs = [[0.0, 1.0]]
        result = _compute_similarity_matrix(pr_embs, issue_embs)
        assert result.shape == (1, 1)
        assert abs(result[0, 0]) < 1e-6

    def test_dimensions_match(self):
        pr_embs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        issue_embs = [[1.0, 0.0], [0.0, 1.0]]
        result = _compute_similarity_matrix(pr_embs, issue_embs)
        assert result.shape == (3, 2)


class TestFindIssuePrLinks:
    def test_no_prs(self):
        issues = [_make_issue(number=1)]
        report = find_issue_pr_links([], [], issues, [[1.0, 0.0]], threshold=0.45)
        assert report.total_prs == 0
        assert report.total_issues == 1
        assert report.orphan_issues == [1]
        assert report.suggestions == []

    def test_no_issues(self):
        prs = [_make_pr(number=1)]
        report = find_issue_pr_links(prs, [[1.0, 0.0]], [], [], threshold=0.45)
        assert report.total_issues == 0
        assert report.suggestions == []
        assert report.orphan_issues == []

    def test_above_threshold_creates_suggestion(self):
        prs = [_make_pr(number=1, title="Fix auth")]
        issues = [_make_issue(number=10, title="Auth broken")]
        # Identical embeddings -> similarity = 1.0
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[1.0, 0.0]], threshold=0.45
        )
        assert len(report.suggestions) == 1
        assert report.suggestions[0].pr_number == 1
        assert report.suggestions[0].issue_number == 10
        assert report.suggestions[0].similarity >= 0.45
        assert report.orphan_issues == []

    def test_below_threshold_no_suggestion(self):
        prs = [_make_pr(number=1)]
        issues = [_make_issue(number=10)]
        # Orthogonal embeddings -> similarity = 0.0
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[0.0, 1.0]], threshold=0.45
        )
        assert len(report.suggestions) == 0
        assert report.orphan_issues == [10]

    def test_explicit_links_excluded_from_suggestions(self):
        prs = [_make_pr(number=1, body="Fixes #10", linked_issues=[10])]
        issues = [_make_issue(number=10)]
        # Identical embeddings, but PR explicitly links to issue 10
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[1.0, 0.0]], threshold=0.45
        )
        assert len(report.suggestions) == 0
        assert len(report.explicit_links) == 1
        assert report.explicit_links[0].pr_number == 1
        assert report.explicit_links[0].issue_number == 10
        assert report.explicit_links[0].is_explicit is True
        assert report.orphan_issues == []

    def test_orphan_issues_detected(self):
        prs = [_make_pr(number=1)]
        issues = [_make_issue(number=10), _make_issue(number=20)]
        # PR similar to issue 10 only
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]],
            issues, [[1.0, 0.0], [0.0, 1.0]],
            threshold=0.45,
        )
        assert 10 not in report.orphan_issues
        assert 20 in report.orphan_issues

    def test_multi_match_sorted_by_similarity(self):
        prs = [_make_pr(number=1), _make_pr(number=2)]
        issues = [_make_issue(number=10)]
        # PR 1 embedding [0.8, 0.6], PR 2 embedding [1.0, 0.0]
        # Issue embedding [1.0, 0.0]
        # PR 2 should have higher similarity
        report = find_issue_pr_links(
            prs, [[0.8, 0.6], [1.0, 0.0]],
            issues, [[1.0, 0.0]],
            threshold=0.45,
        )
        assert len(report.suggestions) == 2
        # Sorted descending by similarity
        assert report.suggestions[0].similarity >= report.suggestions[1].similarity
        assert report.suggestions[0].pr_number == 2

    def test_config_default_threshold(self):
        """Threshold 0 falls back to config default (0.45)."""
        prs = [_make_pr(number=1)]
        issues = [_make_issue(number=10)]
        # Use embeddings that give similarity ~0.5 (above 0.45 default)
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[1.0, 0.0]], threshold=0
        )
        # Similarity = 1.0, well above 0.45 default
        assert len(report.suggestions) == 1

    def test_report_metadata(self):
        prs = [_make_pr(number=1)]
        issues = [_make_issue(number=10)]
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[1.0, 0.0]], threshold=0.5
        )
        assert report.owner == "owner"
        assert report.repo == "repo"
        assert report.total_prs == 1
        assert report.total_issues == 1
        assert report.threshold == 0.5

    def test_explicit_link_outside_issue_set_ignored(self):
        """Explicit link to an issue not in the issue list is not recorded."""
        prs = [_make_pr(number=1, linked_issues=[999])]
        issues = [_make_issue(number=10)]
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]], issues, [[0.0, 1.0]], threshold=0.45
        )
        # Issue 999 not in our issue set, so no explicit link recorded
        assert len(report.explicit_links) == 0
        assert report.orphan_issues == [10]

    def test_both_explicit_and_suggestion(self):
        """PR links to one issue explicitly and is similar to another."""
        prs = [_make_pr(number=1, linked_issues=[10])]
        issues = [_make_issue(number=10), _make_issue(number=20)]
        # PR embedding similar to both issues
        report = find_issue_pr_links(
            prs, [[1.0, 0.0]],
            issues, [[1.0, 0.0], [1.0, 0.0]],
            threshold=0.45,
        )
        assert len(report.explicit_links) == 1
        assert report.explicit_links[0].issue_number == 10
        # Issue 20 should be a suggestion (not explicit)
        assert len(report.suggestions) == 1
        assert report.suggestions[0].issue_number == 20
        assert report.orphan_issues == []


class TestLinkingScorecard:
    def test_json_serialization(self):
        report = LinkingReport(
            owner="owner",
            repo="repo",
            total_prs=2,
            total_issues=3,
            suggestions=[
                LinkSuggestion(
                    pr_number=1,
                    issue_number=10,
                    similarity=0.75,
                    pr_title="Fix auth",
                    issue_title="Auth broken",
                ),
            ],
            orphan_issues=[20],
            threshold=0.45,
        )
        json_str = linking_report_to_json(report)
        data = json.loads(json_str)
        assert data["owner"] == "owner"
        assert data["total_prs"] == 2
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["similarity"] == 0.75
        assert data["orphan_issues"] == [20]

    def test_rich_rendering(self):
        console = Console(record=True, width=120)
        report = LinkingReport(
            owner="owner",
            repo="repo",
            total_prs=1,
            total_issues=2,
            suggestions=[
                LinkSuggestion(
                    pr_number=1,
                    issue_number=10,
                    similarity=0.72,
                    pr_title="Fix auth",
                    issue_title="Auth broken",
                ),
            ],
            orphan_issues=[20],
            threshold=0.45,
        )
        render_linking_report(report, console)
        output = console.export_text()
        assert "Issue-to-PR Linking" in output
        assert "owner/repo" in output
        assert "#1" in output
        assert "#10" in output
        assert "#20" in output
        assert "Suggested PR-Issue Links" in output
        assert "Orphan" in output
        assert "linked PRs" in output
