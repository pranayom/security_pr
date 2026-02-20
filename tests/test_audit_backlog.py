"""Tests for audit backlog — batch triage, clustering, and report generation."""

import pytest

from oss_maintainer_toolkit.gatekeeper.audit_backlog import (
    _run_all_heuristics,
    find_duplicate_clusters,
)
from oss_maintainer_toolkit.gatekeeper.audit_scorecard import (
    _pct,
    audit_report_to_json,
    audit_report_to_markdown,
    render_audit_report,
)
from oss_maintainer_toolkit.gatekeeper.models import (
    AuditReport,
    AuditRiskEntry,
    DuplicateCluster,
    PRAuthor,
    PRFileChange,
    PRMetadata,
)


def _make_pr(number: int, title: str = "", author: str = "user1", body: str = "") -> PRMetadata:
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title or f"PR #{number}",
        body=body,
        author=PRAuthor(login=author, contributions_to_repo=5),
        files=[PRFileChange(filename="src/main.py", additions=10, deletions=2)],
        diff_text=f"diff for PR {number}",
    )


def _make_report(**kwargs) -> AuditReport:
    defaults = dict(
        owner="owner",
        repo="repo",
        prs_analyzed=100,
        total_open_prs=500,
        elapsed_seconds=45.2,
        fast_track_count=70,
        review_required_count=25,
        recommend_close_count=5,
        unique_authors=60,
        first_time_contributors=40,
        new_accounts=10,
        sensitive_path_prs=20,
        low_test_prs=15,
        flag_frequency={"first_contribution": 40, "sensitive_paths": 20, "low_test_ratio": 15},
        highest_risk=[
            AuditRiskEntry(
                pr_number=42, title="Risky change", author="newuser",
                score=0.85, flag_count=3, high_severity_count=1,
                flags=["sensitive_paths", "new_account", "first_contribution"],
            ),
        ],
        clusters_090=[
            DuplicateCluster(
                members=[
                    {"pr": 10, "title": "Fix bug", "author": "a", "similarity": 0.0},
                    {"pr": 20, "title": "Fix same bug", "author": "b", "similarity": 0.95},
                ],
                threshold=0.90,
            ),
        ],
        clusters_085=[],
        clusters_080=[],
    )
    defaults.update(kwargs)
    return AuditReport(**defaults)


class TestFindDuplicateClusters:
    def test_finds_identical_embeddings(self):
        prs = [_make_pr(1, "Fix A"), _make_pr(2, "Fix A copy")]
        embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # identical

        clusters = find_duplicate_clusters(prs, embeddings, threshold=0.90)

        assert len(clusters) == 1
        assert len(clusters[0].members) == 2
        assert clusters[0].members[0]["pr"] == 1
        assert clusters[0].members[1]["pr"] == 2

    def test_no_clusters_below_threshold(self):
        prs = [_make_pr(1, "Fix A"), _make_pr(2, "Unrelated")]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # orthogonal

        clusters = find_duplicate_clusters(prs, embeddings, threshold=0.90)

        assert clusters == []

    def test_multiple_clusters(self):
        prs = [_make_pr(i) for i in range(4)]
        embeddings = [
            [1.0, 0.0, 0.0],  # cluster 1
            [1.0, 0.0, 0.0],  # cluster 1
            [0.0, 1.0, 0.0],  # cluster 2
            [0.0, 1.0, 0.0],  # cluster 2
        ]

        clusters = find_duplicate_clusters(prs, embeddings, threshold=0.90)

        assert len(clusters) == 2

    def test_empty_input(self):
        clusters = find_duplicate_clusters([], [], threshold=0.90)
        assert clusters == []

    def test_single_pr(self):
        prs = [_make_pr(1)]
        embeddings = [[1.0, 0.0]]
        clusters = find_duplicate_clusters(prs, embeddings, threshold=0.90)
        assert clusters == []

    def test_cluster_threshold_respected(self):
        """PRs with 0.85 similarity should cluster at 0.80 but not at 0.90."""
        prs = [_make_pr(1), _make_pr(2)]
        # cosine_similarity([1, 0.5], [1, 0.6]) ~= 0.997 but let's use real values
        import math
        # Construct vectors with known cosine similarity ~0.85
        embeddings = [
            [1.0, 0.0],
            [0.85, math.sqrt(1 - 0.85**2)],
        ]

        clusters_090 = find_duplicate_clusters(prs, embeddings, threshold=0.90)
        clusters_080 = find_duplicate_clusters(prs, embeddings, threshold=0.80)

        assert len(clusters_090) == 0
        assert len(clusters_080) == 1


class TestRunAllHeuristics:
    def test_returns_results_for_all_prs(self):
        prs = [_make_pr(1), _make_pr(2), _make_pr(3)]
        results = _run_all_heuristics(prs)

        assert len(results) == 3
        for pr, hr in results:
            assert hasattr(hr, "outcome")
            assert hasattr(hr, "flags")


class TestAuditReportModel:
    def test_serializes_to_json(self):
        report = _make_report()
        json_str = audit_report_to_json(report)

        import json
        data = json.loads(json_str)
        assert data["owner"] == "owner"
        assert data["prs_analyzed"] == 100
        assert data["fast_track_count"] == 70

    def test_empty_report(self):
        report = AuditReport(owner="o", repo="r")
        json_str = audit_report_to_json(report)
        assert '"prs_analyzed": 0' in json_str


class TestAuditReportToMarkdown:
    def test_includes_header(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "owner/repo" in md
        assert "100 most recent" in md

    def test_includes_verdict_table(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "FAST_TRACK" in md
        assert "REVIEW_REQUIRED" in md
        assert "RECOMMEND_CLOSE" in md
        assert "| 70 |" in md

    def test_includes_clusters(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "Duplicate Clusters" in md
        assert "Fix bug" in md

    def test_includes_highest_risk(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "Highest-Risk" in md
        assert "#42" in md
        assert "sensitive_paths" in md

    def test_includes_flag_frequency(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "Flag Frequency" in md
        assert "first_contribution" in md

    def test_includes_contributor_stats(self):
        report = _make_report()
        md = audit_report_to_markdown(report)

        assert "Contributor Profile" in md
        assert "60" in md  # unique authors
        assert "40" in md  # first-time

    def test_empty_report(self):
        report = AuditReport(owner="o", repo="r")
        md = audit_report_to_markdown(report)

        assert "No PRs analyzed" in md

    def test_includes_vision_document(self):
        report = _make_report(vision_document="openclaw.yaml")
        md = audit_report_to_markdown(report)

        assert "openclaw.yaml" in md


class TestRenderAuditReport:
    def test_renders_without_error(self):
        """Smoke test — just verify it doesn't crash."""
        import os
        from rich.console import Console

        report = _make_report()
        console = Console(file=open(os.devnull, "w"))
        render_audit_report(report, console)

    def test_renders_empty_report(self):
        import os
        from rich.console import Console

        report = AuditReport(owner="o", repo="r")
        console = Console(file=open(os.devnull, "w"))
        render_audit_report(report, console)


class TestPctHelper:
    def test_normal_pct(self):
        assert _pct(25, 100) == "25%"

    def test_zero_total(self):
        assert _pct(5, 0) == "0%"

    def test_full_pct(self):
        assert _pct(100, 100) == "100%"
