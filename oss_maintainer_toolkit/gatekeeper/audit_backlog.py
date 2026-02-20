"""Batch audit of a repository's PR backlog.

Fetches N open PRs, runs Tiers 1+2 (embedding dedup + heuristics),
and produces a structured AuditReport with verdict distribution,
duplicate clusters, highest-risk PRs, and contributor stats.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding, cosine_similarity
from oss_maintainer_toolkit.gatekeeper.heuristics import run_heuristics
from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr
from oss_maintainer_toolkit.gatekeeper.models import (
    AuditReport,
    AuditRiskEntry,
    DuplicateCluster,
    FlagSeverity,
    PRMetadata,
    TierOutcome,
)


def find_duplicate_clusters(
    prs: list[PRMetadata],
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list[DuplicateCluster]:
    """Find clusters of duplicate PRs above threshold using BFS.

    Returns list of DuplicateCluster, each with 2+ members.
    """
    n = len(prs)
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                adj[i].append((j, sim))
                adj[j].append((i, sim))

    visited: set[int] = set()
    clusters: list[DuplicateCluster] = []

    for start in range(n):
        if start in visited or start not in adj:
            continue
        members = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            max_sim = 0.0
            for other, sim in adj.get(node, []):
                max_sim = max(max_sim, sim)
                if other not in visited:
                    visited.add(other)
                    queue.append(other)
            members.append({
                "pr": prs[node].number,
                "title": prs[node].title,
                "author": prs[node].author.login,
                "similarity": round(max_sim if node != start else 0.0, 4),
            })
        if len(members) >= 2:
            clusters.append(DuplicateCluster(members=members, threshold=threshold))

    return clusters


def _run_all_heuristics(
    prs: list[PRMetadata],
    vision_focus_areas: list[str] | None = None,
) -> list[tuple[PRMetadata, object]]:
    """Run Tier 2 heuristics on all PRs."""
    results = []
    for pr in prs:
        hr = run_heuristics(pr, recent_prs=prs, extra_sensitive_paths=vision_focus_areas)
        results.append((pr, hr))
    return results


async def run_audit(
    owner: str,
    repo: str,
    count: int = 100,
    concurrency: int = 3,
    vision_document_path: str = "",
) -> AuditReport:
    """Run a full backlog audit on a repository.

    Fetches `count` most recent open PRs, computes embeddings,
    finds duplicate clusters, runs heuristics, and returns an AuditReport.
    """
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient

    start_time = time.time()

    # Load vision document if provided
    vision_focus_areas: list[str] | None = None
    vision_name = ""
    if vision_document_path:
        import os
        from oss_maintainer_toolkit.gatekeeper.vision import load_vision_document
        if os.path.exists(vision_document_path):
            vision = load_vision_document(vision_document_path)
            vision_focus_areas = vision.focus_areas
            vision_name = os.path.basename(vision_document_path)

    # Fetch PR numbers
    async with GitHubClient(rate_limit_buffer=0) as client:
        raw_prs = await client.list_open_prs(owner, repo)
        total_open = len(raw_prs)
        pr_numbers = [p["number"] for p in raw_prs[:count]]

        # Batch ingest with concurrency control
        prs: list[PRMetadata] = []
        sem = asyncio.Semaphore(concurrency)

        async def _ingest(number: int) -> PRMetadata | None:
            async with sem:
                try:
                    return await ingest_pr(owner, repo, number, client)
                except Exception:
                    return None

        tasks = [_ingest(n) for n in pr_numbers]
        results = await asyncio.gather(*tasks)
        prs = [r for r in results if r is not None]

    if not prs:
        return AuditReport(
            owner=owner, repo=repo,
            total_open_prs=total_open,
            elapsed_seconds=time.time() - start_time,
            vision_document=vision_name,
        )

    # Tier 1: Compute embeddings and find clusters
    embeddings = [compute_embedding(pr) for pr in prs]
    clusters_090 = find_duplicate_clusters(prs, embeddings, 0.90)
    clusters_085 = find_duplicate_clusters(prs, embeddings, 0.85)
    clusters_080 = find_duplicate_clusters(prs, embeddings, 0.80)

    # Mark duplicate PRs (from 0.90 clusters)
    dup_prs: set[int] = set()
    for cluster in clusters_090:
        for member in cluster.members[1:]:  # skip anchor
            dup_prs.add(member["pr"])

    # Tier 2: Run heuristics
    heuristic_results = _run_all_heuristics(prs, vision_focus_areas)

    # Classify verdicts
    fast_track = 0
    review_required = 0
    recommend_close = len(dup_prs)

    for pr, hr in heuristic_results:
        if pr.number in dup_prs:
            continue
        if hr.outcome == TierOutcome.GATED:
            review_required += 1
        else:
            fast_track += 1

    # Flag frequency
    flag_counter: Counter = Counter()
    for _, hr in heuristic_results:
        for f in hr.flags:
            flag_counter[f.rule_id] += 1

    # Highest-risk PRs
    risk_ranked = sorted(
        heuristic_results,
        key=lambda x: (
            sum(1 for f in x[1].flags if f.severity == FlagSeverity.HIGH),
            len(x[1].flags),
            x[1].suspicion_score,
        ),
        reverse=True,
    )
    highest_risk = []
    for pr, hr in risk_ranked[:15]:
        if not hr.flags:
            break
        highest_risk.append(AuditRiskEntry(
            pr_number=pr.number,
            title=pr.title,
            author=pr.author.login,
            score=round(hr.suspicion_score, 3),
            flag_count=len(hr.flags),
            high_severity_count=sum(1 for f in hr.flags if f.severity == FlagSeverity.HIGH),
            flags=[f.rule_id for f in hr.flags],
        ))

    # Contributor stats
    n = len(prs)
    unique_authors = len({pr.author.login for pr in prs})
    first_timers = sum(1 for pr in prs if pr.author.contributions_to_repo == 0)
    new_accounts = sum(
        1 for pr in prs
        if pr.author.account_created_at
        and (datetime.now(timezone.utc) - pr.author.account_created_at).days < 90
    )
    sensitive_prs = sum(
        1 for _, hr in heuristic_results
        if any(f.rule_id == "sensitive_paths" for f in hr.flags)
    )
    low_test_prs = sum(
        1 for _, hr in heuristic_results
        if any(f.rule_id == "low_test_ratio" for f in hr.flags)
    )

    elapsed = time.time() - start_time

    return AuditReport(
        owner=owner,
        repo=repo,
        prs_analyzed=n,
        total_open_prs=total_open,
        elapsed_seconds=round(elapsed, 1),
        fast_track_count=fast_track,
        review_required_count=review_required,
        recommend_close_count=recommend_close,
        clusters_090=clusters_090,
        clusters_085=clusters_085,
        clusters_080=clusters_080,
        highest_risk=highest_risk,
        flag_frequency=dict(flag_counter.most_common()),
        unique_authors=unique_authors,
        first_time_contributors=first_timers,
        new_accounts=new_accounts,
        sensitive_path_prs=sensitive_prs,
        low_test_prs=low_test_prs,
        vision_document=vision_name,
    )
