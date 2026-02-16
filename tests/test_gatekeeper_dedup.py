"""Tests for the Gatekeeper dedup engine (Tier 1)."""

import pytest

from src.gatekeeper.dedup import (
    _build_embedding_text,
    check_duplicates,
    cosine_similarity,
)
from src.gatekeeper.models import PRAuthor, PRFileChange, PRMetadata, TierOutcome


def _make_pr(number: int, title: str = "Test PR", body: str = "", files: list | None = None) -> PRMetadata:
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=PRAuthor(login="user"),
        files=files or [],
        diff_text="",
    )


class TestCosinesimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.9, 0.1]
        sim = cosine_similarity(a, b)
        assert 0.9 < sim < 1.0

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0


class TestBuildEmbeddingText:
    def test_title_only(self):
        pr = _make_pr(1, title="Fix login bug")
        text = _build_embedding_text(pr)
        assert "Fix login bug" in text

    def test_title_and_body(self):
        pr = _make_pr(1, title="Fix login", body="This fixes the auth flow")
        text = _build_embedding_text(pr)
        assert "Fix login" in text
        assert "This fixes the auth flow" in text

    def test_includes_filenames(self):
        files = [PRFileChange(filename="src/auth.py"), PRFileChange(filename="tests/test_auth.py")]
        pr = _make_pr(1, files=files)
        text = _build_embedding_text(pr)
        assert "src/auth.py" in text
        assert "tests/test_auth.py" in text

    def test_body_truncated(self):
        pr = _make_pr(1, body="x" * 5000)
        text = _build_embedding_text(pr)
        # Body should be truncated to 1000 chars
        assert len(text) < 5100


class TestCheckDuplicates:
    def test_no_existing_prs(self):
        pr = _make_pr(1)
        result = check_duplicates(pr, [0.1, 0.2], [], [], threshold=0.9)
        assert result.outcome == TierOutcome.PASS
        assert result.is_duplicate is False

    def test_duplicate_found(self):
        pr = _make_pr(1)
        existing = _make_pr(2)
        # Same embedding = cosine similarity 1.0
        emb = [1.0, 0.0, 0.0]
        result = check_duplicates(pr, emb, [existing], [emb], threshold=0.9)
        assert result.outcome == TierOutcome.GATED
        assert result.is_duplicate is True
        assert result.duplicate_of == 2
        assert result.max_similarity == pytest.approx(1.0)

    def test_no_duplicate_below_threshold(self):
        pr = _make_pr(1)
        existing = _make_pr(2)
        emb_pr = [1.0, 0.0, 0.0]
        emb_existing = [0.0, 1.0, 0.0]  # orthogonal
        result = check_duplicates(pr, emb_pr, [existing], [emb_existing], threshold=0.9)
        assert result.outcome == TierOutcome.PASS
        assert result.is_duplicate is False

    def test_skips_self_comparison(self):
        pr = _make_pr(1)
        emb = [1.0, 0.0, 0.0]
        # Same PR number should be skipped
        result = check_duplicates(pr, emb, [pr], [emb], threshold=0.9)
        assert result.outcome == TierOutcome.PASS

    def test_multiple_existing_highest_similarity(self):
        pr = _make_pr(1)
        existing1 = _make_pr(2)
        existing2 = _make_pr(3)

        emb_pr = [1.0, 0.0, 0.0]
        emb1 = [0.5, 0.5, 0.0]  # moderate similarity
        emb2 = [0.99, 0.01, 0.0]  # high similarity

        result = check_duplicates(
            pr, emb_pr,
            [existing1, existing2], [emb1, emb2],
            threshold=0.9,
        )
        assert result.duplicate_of == 3  # highest similarity
