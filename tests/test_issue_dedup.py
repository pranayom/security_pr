"""Tests for the Gatekeeper issue dedup (Tier 1)."""

from unittest.mock import MagicMock, patch

import pytest

from oss_maintainer_toolkit.gatekeeper.models import (
    DedupResult,
    IssueAuthor,
    IssueMetadata,
    TierOutcome,
)
from oss_maintainer_toolkit.gatekeeper.issue_dedup import (
    _build_issue_embedding_text,
    check_issue_duplicates,
    compute_issue_embedding,
)


def _make_issue(number: int = 1, title: str = "Test", body: str = "", labels: list[str] | None = None) -> IssueMetadata:
    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(login="user"),
        labels=labels or [],
    )


class TestBuildEmbeddingText:
    def test_title_only(self):
        issue = _make_issue(title="Bug in login")
        text = _build_issue_embedding_text(issue)
        assert "Bug in login" in text

    def test_includes_body(self):
        issue = _make_issue(title="Bug", body="The login page crashes when...")
        text = _build_issue_embedding_text(issue)
        assert "login page crashes" in text

    def test_includes_labels(self):
        issue = _make_issue(title="Bug", labels=["bug", "critical"])
        text = _build_issue_embedding_text(issue)
        assert "bug" in text
        assert "critical" in text

    def test_truncates_body(self):
        issue = _make_issue(title="T", body="x" * 2000)
        text = _build_issue_embedding_text(issue)
        # Body should be truncated to 1000 chars
        assert len(text) < 1500


class TestCheckIssueDuplicates:
    def test_no_existing_issues(self):
        issue = _make_issue()
        result = check_issue_duplicates(issue, [0.1, 0.2], [], [], threshold=0.85)
        assert result.outcome == TierOutcome.PASS
        assert not result.is_duplicate

    def test_duplicate_found(self):
        issue = _make_issue(number=1)
        existing = _make_issue(number=2)
        # Use identical embeddings to trigger duplicate
        result = check_issue_duplicates(
            issue, [1.0, 0.0], [existing], [[1.0, 0.0]], threshold=0.85
        )
        assert result.outcome == TierOutcome.GATED
        assert result.is_duplicate
        assert result.duplicate_of == 2
        assert result.max_similarity >= 0.85

    def test_no_duplicate_below_threshold(self):
        issue = _make_issue(number=1)
        existing = _make_issue(number=2)
        result = check_issue_duplicates(
            issue, [1.0, 0.0], [existing], [[0.0, 1.0]], threshold=0.85
        )
        assert result.outcome == TierOutcome.PASS
        assert not result.is_duplicate

    def test_skips_self_comparison(self):
        issue = _make_issue(number=1)
        # Same number = same issue
        existing = _make_issue(number=1)
        result = check_issue_duplicates(
            issue, [1.0, 0.0], [existing], [[1.0, 0.0]], threshold=0.85
        )
        assert result.outcome == TierOutcome.PASS

    def test_uses_config_default_threshold(self):
        """Threshold 0 falls back to config."""
        issue = _make_issue(number=1)
        existing = _make_issue(number=2)
        result = check_issue_duplicates(
            issue, [1.0, 0.0], [existing], [[1.0, 0.0]], threshold=0
        )
        # Config default is 0.85, similarity is 1.0, so should be flagged
        assert result.is_duplicate

    def test_multiple_existing_finds_highest(self):
        issue = _make_issue(number=1)
        e1 = _make_issue(number=2)
        e2 = _make_issue(number=3)
        result = check_issue_duplicates(
            issue,
            [1.0, 0.0],
            [e1, e2],
            [[0.5, 0.5], [1.0, 0.0]],
            threshold=0.85,
        )
        assert result.is_duplicate
        assert result.duplicate_of == 3

    @patch("oss_maintainer_toolkit.gatekeeper.issue_dedup._get_model")
    def test_compute_issue_embedding(self, mock_get_model):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5, 0.5, 0.5])
        mock_get_model.return_value = mock_model

        issue = _make_issue(title="Test issue")
        embedding = compute_issue_embedding(issue)

        assert len(embedding) == 3
        mock_model.encode.assert_called_once()
