"""Cross-PR Conflict Detection: identify PRs with overlapping file changes.

Combines file path overlap (Tier 2) and embedding similarity (Tier 1)
to detect PRs that should be reviewed together or sequenced.
No LLM, $0 cost.
"""

from __future__ import annotations

import numpy as np

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.linking import _compute_similarity_matrix
from oss_maintainer_toolkit.gatekeeper.models import (
    ConflictPair,
    ConflictReport,
    PRMetadata,
)


def _compute_file_overlap(pr_a: PRMetadata, pr_b: PRMetadata) -> list[str]:
    """Compute overlapping filenames between two PRs."""
    files_a = {f.filename for f in pr_a.files}
    files_b = {f.filename for f in pr_b.files}
    return sorted(files_a & files_b)


def _file_overlap_score(pr_a: PRMetadata, pr_b: PRMetadata) -> float:
    """Compute Jaccard similarity of file sets between two PRs."""
    files_a = {f.filename for f in pr_a.files}
    files_b = {f.filename for f in pr_b.files}
    if not files_a and not files_b:
        return 0.0
    intersection = files_a & files_b
    union = files_a | files_b
    return len(intersection) / len(union) if union else 0.0


def detect_conflicts(
    prs: list[PRMetadata],
    embeddings: list[list[float]],
    file_overlap_weight: float = 0.0,
    threshold: float = 0.0,
) -> ConflictReport:
    """Detect conflicting PR pairs via file overlap + embedding similarity.

    Blended score: file_overlap_weight * file_jaccard + (1 - file_overlap_weight) * embedding_similarity.

    Args:
        prs: List of open PRs.
        embeddings: Corresponding embedding vectors for each PR.
        file_overlap_weight: Weight for file overlap vs embedding (0 = config default).
        threshold: Minimum confidence to report (0 = config default).

    Returns:
        ConflictReport with conflict pairs sorted by confidence descending.
    """
    if file_overlap_weight <= 0:
        file_overlap_weight = gatekeeper_settings.conflict_file_overlap_weight
    if threshold <= 0:
        threshold = gatekeeper_settings.conflict_threshold

    owner = prs[0].owner if prs else ""
    repo = prs[0].repo if prs else ""

    report = ConflictReport(
        owner=owner,
        repo=repo,
        total_open_prs=len(prs),
        file_overlap_weight=file_overlap_weight,
        threshold=threshold,
    )

    if len(prs) < 2:
        return report

    # Compute all-pairs embedding similarity
    sim_matrix = _compute_similarity_matrix(embeddings, embeddings)
    embed_weight = 1.0 - file_overlap_weight

    pairs: list[ConflictPair] = []

    for i in range(len(prs)):
        for j in range(i + 1, len(prs)):
            overlap_files = _compute_file_overlap(prs[i], prs[j])
            jaccard = _file_overlap_score(prs[i], prs[j])
            emb_sim = float(sim_matrix[i, j]) if sim_matrix.size > 0 else 0.0

            confidence = file_overlap_weight * jaccard + embed_weight * emb_sim

            if confidence >= threshold:
                pairs.append(ConflictPair(
                    pr_a=prs[i].number,
                    pr_b=prs[j].number,
                    pr_a_title=prs[i].title,
                    pr_b_title=prs[j].title,
                    overlapping_files=overlap_files,
                    semantic_similarity=round(emb_sim, 4),
                    confidence=round(confidence, 4),
                ))

    pairs.sort(key=lambda p: p.confidence, reverse=True)
    report.conflict_pairs = pairs
    return report
