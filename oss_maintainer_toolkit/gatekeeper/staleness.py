"""Smart Stale Detection — semantic staleness via embeddings + metadata."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.linking import _compute_similarity_matrix
from oss_maintainer_toolkit.gatekeeper.models import (
    IssueMetadata,
    PRMetadata,
    StaleItem,
    StalenessReport,
)


def _find_superseded_prs(
    open_prs: list[PRMetadata],
    open_pr_embeddings: list[list[float]],
    merged_prs: list[PRMetadata],
    merged_pr_embeddings: list[list[float]],
    threshold: float,
) -> list[StaleItem]:
    """Find open PRs whose diffs are very similar to recently merged PRs.

    Only flags when the merged PR was merged after the open PR was created
    (temporal guard — the open PR is likely obsolete, not the other way around).
    Returns at most one match per open PR (the best match).
    """
    if not open_prs or not merged_prs:
        return []

    # Rows = open PRs, Cols = merged PRs
    sim_matrix = _compute_similarity_matrix(open_pr_embeddings, merged_pr_embeddings)
    if sim_matrix.size == 0:
        return []

    results: list[StaleItem] = []
    for i, open_pr in enumerate(open_prs):
        best_sim = 0.0
        best_merged: PRMetadata | None = None
        for j, merged_pr in enumerate(merged_prs):
            sim = float(sim_matrix[i, j])
            if sim < threshold:
                continue
            # Temporal guard: only flag if merged AFTER the open PR was created
            if open_pr.created_at and merged_pr.merged_at:
                if merged_pr.merged_at <= open_pr.created_at:
                    continue
            elif not merged_pr.merged_at:
                continue
            if sim > best_sim:
                best_sim = sim
                best_merged = merged_pr

        if best_merged is not None:
            results.append(StaleItem(
                item_type="pr",
                number=open_pr.number,
                title=open_pr.title,
                signal="superseded",
                related_number=best_merged.number,
                related_title=best_merged.title,
                similarity=round(best_sim, 4),
                explanation=(
                    f"PR #{open_pr.number} is {best_sim:.0%} similar to "
                    f"merged PR #{best_merged.number} — likely superseded."
                ),
            ))

    return results


def _find_addressed_issues(
    open_issues: list[IssueMetadata],
    open_issue_embeddings: list[list[float]],
    merged_prs: list[PRMetadata],
    merged_pr_embeddings: list[list[float]],
    threshold: float,
) -> list[StaleItem]:
    """Find open issues semantically similar to merged PRs (likely already fixed).

    Returns at most one match per open issue (the best match).
    """
    if not open_issues or not merged_prs:
        return []

    # Rows = merged PRs, Cols = open issues
    sim_matrix = _compute_similarity_matrix(merged_pr_embeddings, open_issue_embeddings)
    if sim_matrix.size == 0:
        return []

    results: list[StaleItem] = []
    for j, issue in enumerate(open_issues):
        best_sim = 0.0
        best_pr: PRMetadata | None = None
        for i, merged_pr in enumerate(merged_prs):
            sim = float(sim_matrix[i, j])
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_pr = merged_pr

        if best_pr is not None:
            results.append(StaleItem(
                item_type="issue",
                number=issue.number,
                title=issue.title,
                signal="addressed",
                related_number=best_pr.number,
                related_title=best_pr.title,
                similarity=round(best_sim, 4),
                explanation=(
                    f"Issue #{issue.number} is {best_sim:.0%} similar to "
                    f"merged PR #{best_pr.number} — may already be addressed."
                ),
            ))

    return results


def _find_blocked_prs(
    open_prs: list[PRMetadata],
    open_issues: list[IssueMetadata],
) -> list[StaleItem]:
    """Find open PRs that reference still-open issues (blocked).

    Pure metadata check — no embeddings needed.
    """
    open_issue_numbers = {issue.number for issue in open_issues}
    results: list[StaleItem] = []

    for pr in open_prs:
        blocking = [n for n in pr.linked_issues if n in open_issue_numbers]
        if blocking:
            issue_refs = ", ".join(f"#{n}" for n in blocking)
            results.append(StaleItem(
                item_type="pr",
                number=pr.number,
                title=pr.title,
                signal="blocked",
                related_number=blocking[0],
                explanation=(
                    f"PR #{pr.number} is blocked by open issue(s): {issue_refs}."
                ),
            ))

    return results


def _find_inactive_items(
    open_prs: list[PRMetadata],
    open_issues: list[IssueMetadata],
    inactive_days: int,
) -> tuple[list[StaleItem], list[StaleItem]]:
    """Find open PRs and issues with no activity beyond the threshold.

    Items without an updated_at timestamp are not flagged (no data to judge).
    Returns (inactive_prs, inactive_issues), each sorted oldest-first.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=inactive_days)

    inactive_prs: list[StaleItem] = []
    for pr in open_prs:
        if pr.updated_at and pr.updated_at < cutoff:
            inactive_prs.append(StaleItem(
                item_type="pr",
                number=pr.number,
                title=pr.title,
                signal="inactive",
                last_activity=pr.updated_at,
                explanation=(
                    f"PR #{pr.number} has had no activity since "
                    f"{pr.updated_at.strftime('%Y-%m-%d')}."
                ),
            ))

    inactive_issues: list[StaleItem] = []
    for issue in open_issues:
        if issue.updated_at and issue.updated_at < cutoff:
            inactive_issues.append(StaleItem(
                item_type="issue",
                number=issue.number,
                title=issue.title,
                signal="inactive",
                last_activity=issue.updated_at,
                explanation=(
                    f"Issue #{issue.number} has had no activity since "
                    f"{issue.updated_at.strftime('%Y-%m-%d')}."
                ),
            ))

    # Sort oldest-first
    inactive_prs.sort(key=lambda x: x.last_activity or datetime.max.replace(tzinfo=timezone.utc))
    inactive_issues.sort(key=lambda x: x.last_activity or datetime.max.replace(tzinfo=timezone.utc))

    return inactive_prs, inactive_issues


def detect_stale_items(
    open_prs: list[PRMetadata],
    open_pr_embeddings: list[list[float]],
    open_issues: list[IssueMetadata],
    open_issue_embeddings: list[list[float]],
    merged_prs: list[PRMetadata],
    merged_pr_embeddings: list[list[float]],
    threshold: float = 0.0,
    inactive_days: int = 0,
) -> StalenessReport:
    """Orchestrate all four staleness detection signals.

    Args:
        open_prs: Open pull requests with embeddings.
        open_pr_embeddings: Embedding vectors for open PRs.
        open_issues: Open issues with embeddings.
        open_issue_embeddings: Embedding vectors for open issues.
        merged_prs: Recently merged PRs with embeddings.
        merged_pr_embeddings: Embedding vectors for merged PRs.
        threshold: Similarity threshold (0 = use config default of 0.75).
        inactive_days: Inactivity threshold in days (0 = use config default of 90).

    Returns:
        StalenessReport with all detected stale items.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.stale_similarity_threshold
    if inactive_days <= 0:
        inactive_days = gatekeeper_settings.stale_inactive_days

    owner = (
        open_prs[0].owner if open_prs
        else (open_issues[0].owner if open_issues else "")
    )
    repo = (
        open_prs[0].repo if open_prs
        else (open_issues[0].repo if open_issues else "")
    )

    superseded = _find_superseded_prs(
        open_prs, open_pr_embeddings, merged_prs, merged_pr_embeddings, threshold,
    )
    addressed = _find_addressed_issues(
        open_issues, open_issue_embeddings, merged_prs, merged_pr_embeddings, threshold,
    )
    blocked = _find_blocked_prs(open_prs, open_issues)
    inactive_pr_list, inactive_issue_list = _find_inactive_items(
        open_prs, open_issues, inactive_days,
    )

    return StalenessReport(
        owner=owner,
        repo=repo,
        superseded_prs=superseded,
        addressed_issues=addressed,
        blocked_prs=blocked,
        inactive_prs=inactive_pr_list,
        inactive_issues=inactive_issue_list,
        total_open_prs=len(open_prs),
        total_open_issues=len(open_issues),
        total_merged_prs_checked=len(merged_prs),
        threshold=threshold,
        inactive_days=inactive_days,
    )
