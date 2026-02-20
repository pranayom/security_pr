"""Tests for the Gatekeeper cross-PR conflict detection feature."""

import json

import numpy as np
import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    ConflictPair,
    ConflictReport,
    PRAuthor,
    PRFileChange,
    PRMetadata,
)
from oss_maintainer_toolkit.gatekeeper.conflict_detection import (
    _compute_file_overlap,
    _file_overlap_score,
    detect_conflicts,
)
from oss_maintainer_toolkit.gatekeeper.conflict_scorecard import (
    conflict_report_to_json,
    render_conflict_report,
)


def _make_pr(
    number: int = 1,
    title: str = "Fix bug",
    files: list[str] | None = None,
) -> PRMetadata:
    file_changes = [PRFileChange(filename=f) for f in (files or [])]
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        author=PRAuthor(login="dev"),
        files=file_changes,
    )


# --- File overlap helpers ---

class TestFileOverlap:
    def test_overlapping_files(self):
        pr_a = _make_pr(files=["src/auth.py", "src/utils.py"])
        pr_b = _make_pr(files=["src/auth.py", "src/models.py"])
        overlap = _compute_file_overlap(pr_a, pr_b)
        assert overlap == ["src/auth.py"]

    def test_no_overlap(self):
        pr_a = _make_pr(files=["src/auth.py"])
        pr_b = _make_pr(files=["src/models.py"])
        overlap = _compute_file_overlap(pr_a, pr_b)
        assert overlap == []

    def test_complete_overlap(self):
        pr_a = _make_pr(files=["a.py", "b.py"])
        pr_b = _make_pr(files=["a.py", "b.py"])
        overlap = _compute_file_overlap(pr_a, pr_b)
        assert len(overlap) == 2

    def test_empty_files(self):
        pr_a = _make_pr(files=[])
        pr_b = _make_pr(files=["a.py"])
        overlap = _compute_file_overlap(pr_a, pr_b)
        assert overlap == []


class TestFileOverlapScore:
    def test_identical_sets(self):
        pr_a = _make_pr(files=["a.py", "b.py"])
        pr_b = _make_pr(files=["a.py", "b.py"])
        assert _file_overlap_score(pr_a, pr_b) == 1.0

    def test_no_overlap_score(self):
        pr_a = _make_pr(files=["a.py"])
        pr_b = _make_pr(files=["b.py"])
        assert _file_overlap_score(pr_a, pr_b) == 0.0

    def test_partial_overlap(self):
        pr_a = _make_pr(files=["a.py", "b.py"])
        pr_b = _make_pr(files=["b.py", "c.py"])
        # Jaccard: 1 / 3 = 0.333
        score = _file_overlap_score(pr_a, pr_b)
        assert abs(score - 1.0 / 3.0) < 1e-6

    def test_empty_both(self):
        pr_a = _make_pr(files=[])
        pr_b = _make_pr(files=[])
        assert _file_overlap_score(pr_a, pr_b) == 0.0


# --- Conflict detection ---

class TestDetectConflicts:
    def test_single_pr(self):
        prs = [_make_pr(number=1)]
        report = detect_conflicts(prs, [[1.0, 0.0]], threshold=0.1)
        assert report.conflict_pairs == []
        assert report.total_open_prs == 1

    def test_empty_prs(self):
        report = detect_conflicts([], [], threshold=0.1)
        assert report.conflict_pairs == []

    def test_overlapping_files_detected(self):
        prs = [
            _make_pr(number=1, files=["src/auth.py", "src/utils.py"]),
            _make_pr(number=2, files=["src/auth.py", "src/models.py"]),
        ]
        # Identical embeddings -> high similarity
        emb = [1.0, 0.0]
        report = detect_conflicts(prs, [emb, emb], threshold=0.1)
        assert len(report.conflict_pairs) >= 1
        pair = report.conflict_pairs[0]
        assert pair.pr_a == 1
        assert pair.pr_b == 2
        assert "src/auth.py" in pair.overlapping_files

    def test_no_overlap_low_similarity(self):
        prs = [
            _make_pr(number=1, files=["src/auth.py"]),
            _make_pr(number=2, files=["docs/README.md"]),
        ]
        # Orthogonal embeddings -> 0 similarity
        report = detect_conflicts(
            prs, [[1.0, 0.0], [0.0, 1.0]],
            threshold=0.5, file_overlap_weight=0.5,
        )
        assert len(report.conflict_pairs) == 0

    def test_sorted_by_confidence(self):
        prs = [
            _make_pr(number=1, files=["a.py"]),
            _make_pr(number=2, files=["a.py"]),
            _make_pr(number=3, files=["b.py"]),
        ]
        # PR 1 and 2 share a file, PR 3 doesn't
        report = detect_conflicts(
            prs, [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            threshold=0.01, file_overlap_weight=0.01,
        )
        if len(report.conflict_pairs) >= 2:
            assert report.conflict_pairs[0].confidence >= report.conflict_pairs[1].confidence

    def test_threshold_filters(self):
        prs = [
            _make_pr(number=1, files=["a.py"]),
            _make_pr(number=2, files=["b.py"]),
        ]
        # Very different -> low confidence
        report = detect_conflicts(
            prs, [[1.0, 0.0], [0.0, 1.0]],
            threshold=0.9, file_overlap_weight=0.5,
        )
        assert len(report.conflict_pairs) == 0

    def test_multiple_pairs(self):
        prs = [
            _make_pr(number=1, files=["a.py"]),
            _make_pr(number=2, files=["a.py"]),
            _make_pr(number=3, files=["a.py"]),
        ]
        emb = [1.0, 0.0]
        report = detect_conflicts(prs, [emb, emb, emb], threshold=0.1)
        # 3 PRs -> 3 pairs (1-2, 1-3, 2-3)
        assert len(report.conflict_pairs) == 3

    def test_report_metadata(self):
        prs = [_make_pr(number=1), _make_pr(number=2)]
        report = detect_conflicts(prs, [[1.0, 0.0], [1.0, 0.0]], threshold=0.1)
        assert report.owner == "owner"
        assert report.repo == "repo"
        assert report.total_open_prs == 2


# --- Model tests ---

class TestModels:
    def test_conflict_pair_defaults(self):
        p = ConflictPair(pr_a=1, pr_b=2)
        assert p.overlapping_files == []
        assert p.confidence == 0.0

    def test_conflict_report_defaults(self):
        r = ConflictReport(owner="o", repo="r")
        assert r.conflict_pairs == []
        assert r.total_open_prs == 0


# --- Scorecard ---

class TestConflictScorecard:
    def test_json_serialization(self):
        report = ConflictReport(
            owner="owner", repo="repo", total_open_prs=5,
            conflict_pairs=[
                ConflictPair(
                    pr_a=1, pr_b=2, pr_a_title="Fix auth", pr_b_title="Update auth",
                    overlapping_files=["src/auth.py"],
                    semantic_similarity=0.85, confidence=0.7,
                ),
            ],
            threshold=0.3,
        )
        json_str = conflict_report_to_json(report)
        data = json.loads(json_str)
        assert data["total_open_prs"] == 5
        assert len(data["conflict_pairs"]) == 1
        assert data["conflict_pairs"][0]["pr_a"] == 1

    def test_rich_rendering(self):
        console = Console(record=True, width=120)
        report = ConflictReport(
            owner="owner", repo="repo", total_open_prs=3,
            conflict_pairs=[
                ConflictPair(
                    pr_a=1, pr_b=2, pr_a_title="Fix auth", pr_b_title="Update auth",
                    overlapping_files=["src/auth.py"],
                    semantic_similarity=0.85, confidence=0.7,
                ),
            ],
            threshold=0.3,
        )
        render_conflict_report(report, console)
        output = console.export_text()
        assert "Conflict Detection" in output
        assert "#1" in output
        assert "#2" in output
        assert "src/auth.py" in output

    def test_rich_rendering_no_conflicts(self):
        console = Console(record=True, width=120)
        report = ConflictReport(owner="o", repo="r", total_open_prs=2)
        render_conflict_report(report, console)
        output = console.export_text()
        assert "No conflicting PR pairs" in output
