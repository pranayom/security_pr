"""Tier 1: Embedding-based issue deduplication using sentence-transformers."""

from __future__ import annotations

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.dedup import _get_model, cosine_similarity
from oss_maintainer_toolkit.gatekeeper.models import DedupResult, IssueMetadata, TierOutcome


def _build_issue_embedding_text(issue: IssueMetadata) -> str:
    """Combine issue fields into a single text for embedding."""
    parts = [issue.title]

    if issue.body:
        parts.append(issue.body[:1000])

    if issue.labels:
        parts.append(" ".join(issue.labels))

    return "\n".join(parts)


def compute_issue_embedding(issue: IssueMetadata) -> list[float]:
    """Compute embedding vector for an issue."""
    model = _get_model()
    text = _build_issue_embedding_text(issue)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def check_issue_duplicates(
    issue: IssueMetadata,
    issue_embedding: list[float],
    existing_issues: list[IssueMetadata],
    existing_embeddings: list[list[float]],
    threshold: float = 0.0,
) -> DedupResult:
    """Check if an issue is a duplicate of any existing issues.

    Args:
        issue: The issue to check.
        issue_embedding: Pre-computed embedding for the issue.
        existing_issues: List of existing issues to compare against.
        existing_embeddings: Corresponding embeddings for existing issues.
        threshold: Similarity threshold (0 = use config default).

    Returns:
        DedupResult with outcome GATED if duplicate found, PASS otherwise.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.issue_duplicate_threshold

    if not existing_issues or not existing_embeddings:
        return DedupResult(outcome=TierOutcome.PASS, is_duplicate=False)

    max_sim = 0.0
    dup_of: int | None = None

    for existing_issue, existing_emb in zip(existing_issues, existing_embeddings):
        # Skip self-comparison
        if existing_issue.number == issue.number:
            continue

        sim = cosine_similarity(issue_embedding, existing_emb)
        if sim > max_sim:
            max_sim = sim
            dup_of = existing_issue.number

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
