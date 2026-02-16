"""Tier 1: Embedding-based PR deduplication using sentence-transformers."""

from __future__ import annotations

import numpy as np

from src.gatekeeper.config import gatekeeper_settings
from src.gatekeeper.models import DedupResult, PRMetadata, TierOutcome

_model = None


def _get_model():
    """Lazy-load SentenceTransformer singleton. Raises ImportError if not installed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(gatekeeper_settings.embedding_model)
    return _model


def _build_embedding_text(pr: PRMetadata) -> str:
    """Combine PR fields into a single text for embedding."""
    parts = [pr.title]

    if pr.body:
        parts.append(pr.body[:1000])

    # Include changed filenames and first 100 lines of diff
    if pr.files:
        parts.append(" ".join(f.filename for f in pr.files))

    if pr.diff_text:
        diff_lines = pr.diff_text.split("\n")[:100]
        parts.append("\n".join(diff_lines))

    return "\n".join(parts)


def compute_embedding(pr: PRMetadata) -> list[float]:
    """Compute embedding vector for a PR."""
    model = _get_model()
    text = _build_embedding_text(pr)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def check_duplicates(
    pr: PRMetadata,
    pr_embedding: list[float],
    existing_prs: list[PRMetadata],
    existing_embeddings: list[list[float]],
    threshold: float = 0.0,
) -> DedupResult:
    """Check if a PR is a duplicate of any existing PRs.

    Args:
        pr: The PR to check.
        pr_embedding: Pre-computed embedding for the PR.
        existing_prs: List of existing PRs to compare against.
        existing_embeddings: Corresponding embeddings for existing PRs.
        threshold: Similarity threshold (0 = use config default).

    Returns:
        DedupResult with outcome GATED if duplicate found, PASS otherwise.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.duplicate_threshold

    if not existing_prs or not existing_embeddings:
        return DedupResult(outcome=TierOutcome.PASS, is_duplicate=False)

    max_sim = 0.0
    dup_of: int | None = None

    for existing_pr, existing_emb in zip(existing_prs, existing_embeddings):
        # Skip self-comparison
        if existing_pr.number == pr.number:
            continue

        sim = cosine_similarity(pr_embedding, existing_emb)
        if sim > max_sim:
            max_sim = sim
            dup_of = existing_pr.number

    if max_sim >= threshold:
        return DedupResult(
            outcome=TierOutcome.GATED,
            is_duplicate=True,
            duplicate_of=dup_of,
            max_similarity=max_sim,
        )

    return DedupResult(
        outcome=TierOutcome.PASS,
        is_duplicate=False,
        duplicate_of=None,
        max_similarity=max_sim,
    )
