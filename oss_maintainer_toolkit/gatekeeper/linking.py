"""Tier 1: Embedding-based issue-to-PR linking via cosine similarity."""

from __future__ import annotations

import numpy as np

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import (
    IssueMetadata,
    LinkingReport,
    LinkSuggestion,
    PRMetadata,
)


def _compute_similarity_matrix(
    pr_embeddings: list[list[float]],
    issue_embeddings: list[list[float]],
) -> np.ndarray:
    """Compute all-pairs cosine similarity between PR and issue embeddings.

    Args:
        pr_embeddings: List of PR embedding vectors (N items).
        issue_embeddings: List of issue embedding vectors (M items).

    Returns:
        2D numpy array of shape (N, M) where [i][j] is the cosine similarity
        between PR i and issue j. Returns empty (0, 0) array if either input is empty.
    """
    if not pr_embeddings or not issue_embeddings:
        return np.empty((0, 0))

    pr_matrix = np.array(pr_embeddings)
    issue_matrix = np.array(issue_embeddings)

    # Normalize rows to unit vectors
    pr_norms = np.linalg.norm(pr_matrix, axis=1, keepdims=True)
    issue_norms = np.linalg.norm(issue_matrix, axis=1, keepdims=True)

    # Avoid division by zero
    pr_norms = np.where(pr_norms == 0, 1, pr_norms)
    issue_norms = np.where(issue_norms == 0, 1, issue_norms)

    pr_normalized = pr_matrix / pr_norms
    issue_normalized = issue_matrix / issue_norms

    return pr_normalized @ issue_normalized.T


def find_issue_pr_links(
    prs: list[PRMetadata],
    pr_embeddings: list[list[float]],
    issues: list[IssueMetadata],
    issue_embeddings: list[list[float]],
    threshold: float = 0.0,
) -> LinkingReport:
    """Find potential issue-to-PR links via embedding similarity.

    Args:
        prs: List of PR metadata objects.
        pr_embeddings: Corresponding embedding vectors for each PR.
        issues: List of issue metadata objects.
        issue_embeddings: Corresponding embedding vectors for each issue.
        threshold: Similarity threshold (0 = use config default).

    Returns:
        LinkingReport with suggested links, explicit links, and orphan issues.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.linking_similarity_threshold

    owner = prs[0].owner if prs else (issues[0].owner if issues else "")
    repo = prs[0].repo if prs else (issues[0].repo if issues else "")

    report = LinkingReport(
        owner=owner,
        repo=repo,
        total_prs=len(prs),
        total_issues=len(issues),
        threshold=threshold,
    )

    if not prs or not issues:
        report.orphan_issues = [issue.number for issue in issues]
        return report

    # Build lookup of explicitly linked issue numbers per PR
    explicit_pairs: set[tuple[int, int]] = set()
    for pr in prs:
        for issue_num in pr.linked_issues:
            explicit_pairs.add((pr.number, issue_num))

    # Record explicit links
    issue_map = {issue.number: issue for issue in issues}
    for pr in prs:
        for issue_num in pr.linked_issues:
            if issue_num in issue_map:
                report.explicit_links.append(LinkSuggestion(
                    pr_number=pr.number,
                    issue_number=issue_num,
                    similarity=1.0,
                    pr_title=pr.title,
                    issue_title=issue_map[issue_num].title,
                    is_explicit=True,
                ))

    # Compute similarity matrix
    sim_matrix = _compute_similarity_matrix(pr_embeddings, issue_embeddings)

    # Collect suggestions above threshold (excluding explicit links)
    linked_issue_numbers: set[int] = set()
    suggestions: list[LinkSuggestion] = []

    for i, pr in enumerate(prs):
        for j, issue in enumerate(issues):
            sim = float(sim_matrix[i, j])
            if sim >= threshold and (pr.number, issue.number) not in explicit_pairs:
                suggestions.append(LinkSuggestion(
                    pr_number=pr.number,
                    issue_number=issue.number,
                    similarity=sim,
                    pr_title=pr.title,
                    issue_title=issue.title,
                    is_explicit=False,
                ))
                linked_issue_numbers.add(issue.number)

    # Also mark issues that have explicit links as linked
    for link in report.explicit_links:
        linked_issue_numbers.add(link.issue_number)

    # Sort suggestions by similarity descending
    suggestions.sort(key=lambda s: s.similarity, reverse=True)
    report.suggestions = suggestions

    # Orphan issues: not linked by any suggestion or explicit link
    all_issue_numbers = {issue.number for issue in issues}
    report.orphan_issues = sorted(all_issue_numbers - linked_issue_numbers)

    return report
