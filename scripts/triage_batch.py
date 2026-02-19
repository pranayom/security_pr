"""Batch PR triage script — fetch N PRs, run Tiers 1+2, generate report.

Usage:
    python -u scripts/triage_batch.py --owner openclaw --repo openclaw --count 150
"""

import argparse
import asyncio
import json
import sys
import os
import time

# Force unbuffered output for progress reporting
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load .env BEFORE importing gatekeeper modules (pydantic-settings reads env at import time)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path for editable install fallback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_ai_auditor.gatekeeper.github_client import GitHubClient
from mcp_ai_auditor.gatekeeper.ingest import ingest_pr
from mcp_ai_auditor.gatekeeper.dedup import compute_embedding, cosine_similarity
from mcp_ai_auditor.gatekeeper.heuristics import run_heuristics
from mcp_ai_auditor.gatekeeper.vision import load_vision_document
from mcp_ai_auditor.gatekeeper.models import (
    PRMetadata, TierOutcome, Verdict, FlagSeverity,
    AssessmentScorecard, DedupResult, DimensionScore,
)
from mcp_ai_auditor.gatekeeper.config import gatekeeper_settings


async def fetch_pr_numbers(client: GitHubClient, owner: str, repo: str, count: int) -> tuple[list[int], int]:
    """Fetch the most recent open PR numbers. Returns (numbers, total_open).

    Uses page-limited fetching instead of full pagination to conserve rate limit.
    GitHub returns PRs sorted by created_at desc by default.
    """
    print(f"Fetching open PRs from {owner}/{repo}...")
    pages_needed = (count + 99) // 100  # 100 per page
    all_prs = []

    for page in range(1, pages_needed + 1):
        for attempt in range(3):
            resp = await client.client.get(
                f"/repos/{owner}/{repo}/pulls",
                params={"state": "open", "per_page": "100", "page": str(page), "sort": "created", "direction": "desc"},
            )
            if resp.status_code in (403, 429):
                wait = 30 * (attempt + 1)
                print(f"  Rate limited on page {page}, waiting {wait}s (attempt {attempt + 1}/3)...")
                await asyncio.sleep(wait)
                continue
            break
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_prs.extend(data)
        print(f"  Page {page}: got {len(data)} PRs (total so far: {len(all_prs)})")

    # Get total open PR count from the first response header (Link: last page)
    # Or we can use the repos endpoint
    repo_resp = await client.client.get(f"/repos/{owner}/{repo}")
    repo_resp.raise_for_status()
    # open_issues_count includes issues + PRs, so it's an upper bound
    # For exact count, we'd need the search API
    total_open_approx = repo_resp.json().get("open_issues_count", len(all_prs))
    # Get a more precise count via search (1 API call)
    try:
        search_resp = await client.client.get(
            "/search/issues",
            params={"q": f"repo:{owner}/{repo} type:pr state:open", "per_page": "1"},
        )
        if search_resp.status_code == 200:
            total_open = search_resp.json().get("total_count", total_open_approx)
        else:
            total_open = total_open_approx
    except Exception:
        total_open = total_open_approx

    print(f"  Total open PRs: {total_open}")

    numbers = [p["number"] for p in all_prs[:count]]
    if numbers:
        print(f"  Selected {len(numbers)} most recent (#{numbers[0]} to #{numbers[-1]})")
    return numbers, total_open


async def ingest_batch_with_progress(
    client: GitHubClient, owner: str, repo: str, numbers: list[int], concurrency: int = 3
) -> list[PRMetadata]:
    """Ingest PRs with progress reporting, rate limit awareness, and error tolerance."""
    results: list[PRMetadata] = []
    errors = 0
    total = len(numbers)
    start = time.time()

    # Process in small sequential batches to manage rate limits
    batch_size = concurrency
    for batch_start in range(0, total, batch_size):
        batch_nums = numbers[batch_start:batch_start + batch_size]
        sem = asyncio.Semaphore(concurrency)
        batch_results: list[PRMetadata | None] = [None] * len(batch_nums)

        async def _ingest(idx: int, number: int):
            nonlocal errors
            async with sem:
                try:
                    pr = await ingest_pr(owner, repo, number, client)
                    batch_results[idx] = pr
                except Exception as exc:
                    err_str = str(exc)
                    if "rate limit" in err_str.lower() or "403" in err_str or "429" in err_str:
                        # Wait for rate limit reset
                        print(f"  [RATE LIMIT] PR #{number} — waiting 60s...")
                        await asyncio.sleep(60)
                        try:
                            pr = await ingest_pr(owner, repo, number, client)
                            batch_results[idx] = pr
                            return
                        except Exception as exc2:
                            errors += 1
                            print(f"  [WARN] PR #{number}: retry failed: {exc2}")
                            return
                    errors += 1
                    print(f"  [WARN] PR #{number}: {err_str[:120]}")

        await asyncio.gather(*[_ingest(i, n) for i, n in enumerate(batch_nums)])
        batch_ok = [r for r in batch_results if r is not None]
        results.extend(batch_ok)

        done = batch_start + len(batch_nums)
        if done % 25 == 0 or done >= total:
            elapsed = time.time() - start
            rate = len(results) / elapsed if elapsed > 0 else 0
            print(f"  Ingested {len(results)}/{done} attempted, {total} total ({errors} errors) [{rate:.1f} PR/s]")

        # Small delay between batches to stay within rate limits
        if batch_start + batch_size < total:
            await asyncio.sleep(0.5)

    print(f"  Done: {len(results)} successful, {errors} failed out of {total}")
    return results


def compute_embeddings_with_progress(prs: list[PRMetadata]) -> list[list[float]]:
    """Compute embeddings with progress reporting."""
    print(f"Computing embeddings for {len(prs)} PRs...")
    embeddings = []
    for i, pr in enumerate(prs):
        emb = compute_embedding(pr)
        embeddings.append(emb)
        if (i + 1) % 50 == 0 or (i + 1) == len(prs):
            print(f"  Embedded {i + 1}/{len(prs)}")
    return embeddings


def find_duplicate_clusters(
    prs: list[PRMetadata],
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list[list[tuple[int, str, str, float]]]:
    """Find clusters of duplicate PRs above threshold.

    Returns list of clusters, each cluster is list of (pr_number, title, author, similarity).
    """
    n = len(prs)
    # Build adjacency list
    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                adj[i].append((j, sim))
                adj[j].append((i, sim))

    # BFS to find connected components
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited or start not in adj:
            continue
        cluster = []
        queue = [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            # Find max similarity to any other node in this cluster
            max_sim = 0.0
            for other, sim in adj.get(node, []):
                max_sim = max(max_sim, sim)
                if other not in visited:
                    visited.add(other)
                    queue.append(other)
            cluster.append((
                prs[node].number,
                prs[node].title,
                prs[node].author.login,
                max_sim if node != start else 0.0,  # anchor has 0 similarity marker
            ))
        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def run_all_heuristics(
    prs: list[PRMetadata],
    vision_focus_areas: list[str] | None = None,
) -> list[tuple[PRMetadata, list]]:
    """Run Tier 2 heuristics on all PRs. Returns (pr, flags) pairs."""
    results = []
    for pr in prs:
        hr = run_heuristics(pr, recent_prs=prs, extra_sensitive_paths=vision_focus_areas)
        results.append((pr, hr))
    return results


def generate_report(
    owner: str,
    repo: str,
    prs: list[PRMetadata],
    embeddings: list[list[float]],
    heuristic_results: list[tuple[PRMetadata, object]],
    clusters_090: list,
    clusters_085: list,
    clusters_080: list,
    total_open: int,
    vision_doc_name: str,
    elapsed_seconds: float,
) -> str:
    """Generate a markdown triage report."""

    # Classify verdicts
    verdicts = {"FAST_TRACK": [], "REVIEW_REQUIRED": [], "RECOMMEND_CLOSE": []}

    # Mark duplicates (from 0.90 clusters)
    dup_prs = set()
    for cluster in clusters_090:
        for pr_num, _, _, sim in cluster[1:]:  # skip anchor
            dup_prs.add(pr_num)

    for pr, hr in heuristic_results:
        if pr.number in dup_prs:
            verdicts["RECOMMEND_CLOSE"].append((pr, hr))
        elif hr.outcome == TierOutcome.GATED:
            verdicts["REVIEW_REQUIRED"].append((pr, hr))
        else:
            verdicts["FAST_TRACK"].append((pr, hr))

    n = len(prs)
    ft = len(verdicts["FAST_TRACK"])
    rr = len(verdicts["REVIEW_REQUIRED"])
    rc = len(verdicts["RECOMMEND_CLOSE"])

    # Flag frequency
    flag_counter = Counter()
    for _, hr in heuristic_results:
        for f in hr.flags:
            flag_counter[f.rule_id] += 1

    # Highest-risk PRs (most flags, or high-severity flags)
    risk_ranked = sorted(
        heuristic_results,
        key=lambda x: (
            sum(1 for f in x[1].flags if f.severity == FlagSeverity.HIGH),
            len(x[1].flags),
            x[1].suspicion_score,
        ),
        reverse=True,
    )

    # Account age stats
    new_accounts = sum(
        1 for pr in prs
        if pr.author.account_created_at
        and (datetime.now(timezone.utc) - pr.author.account_created_at).days < 90
    )

    first_timers = sum(1 for pr in prs if pr.author.contributions_to_repo == 0)

    lines = []
    lines.append(f"# OpenClaw PR Triage Report")
    lines.append(f"")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Tool:** mcp-ai-auditor Gatekeeper v0.1.0 (Tiers 1+2)")
    lines.append(f"**Scope:** {n} most recent open PRs out of {total_open:,} total")
    lines.append(f"**Runtime:** {elapsed_seconds:.0f} seconds")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Problem statement
    lines.append(f"## The Problem")
    lines.append(f"")
    lines.append(f"OpenClaw has **{total_open:,} open PRs**. Every open PR costs maintainer time. "
                 f"Many are duplicates, many lack tests, and some touch security-critical paths "
                 f"from brand-new accounts. Automated triage identifies which PRs need attention "
                 f"and which can be fast-tracked or closed.")
    lines.append(f"")

    # Method
    lines.append(f"## What We Did")
    lines.append(f"")
    lines.append(f"We ran the {n} most recent open PRs through a two-tier triage pipeline:")
    lines.append(f"")
    lines.append(f"- **Tier 1 (Embedding Dedup):** Computed semantic embeddings using "
                 f"`all-MiniLM-L6-v2`. Compared all {n*(n-1)//2:,} PR pairs by cosine similarity.")
    lines.append(f"- **Tier 2 (Suspicion Heuristics):** Applied 7 deterministic rules against "
                 f"PR metadata: account age, contribution history, sensitive path changes, "
                 f"test-to-code ratio, unjustified dependency changes, large-diff hiding, "
                 f"and temporal clustering.")
    if vision_doc_name:
        lines.append(f"- **Vision focus areas** from `{vision_doc_name}` fed into Tier 2 as "
                     f"additional sensitive paths (extensions, credentials, gateway, etc.).")
    lines.append(f"")
    lines.append(f"No LLM calls were used. Everything runs locally, costs $0, and completed "
                 f"in {elapsed_seconds:.0f} seconds.")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Results
    lines.append(f"## Results")
    lines.append(f"")
    lines.append(f"### Verdict Distribution")
    lines.append(f"")
    lines.append(f"| Verdict | Count | % | Meaning |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| **FAST_TRACK** | {ft} | {ft*100//n}% | Passed all checks. Safe for quick review. |")
    lines.append(f"| **REVIEW_REQUIRED** | {rr} | {rr*100//n}% | Flagged by heuristics. Needs human attention. |")
    lines.append(f"| **RECOMMEND_CLOSE** | {rc} | {rc*100//n}% | Likely duplicate. Consider closing. |")
    lines.append(f"")
    lines.append(f"A maintainer reviewing this batch could immediately **skip {ft} PRs** "
                 f"that the tool cleared, focus attention on {rr} flagged PRs, "
                 f"and close {rc} duplicates.")
    lines.append(f"")

    # Duplicate clusters
    lines.append(f"### Duplicate Clusters Found")
    lines.append(f"")
    lines.append(f"At 0.90 cosine similarity: **{len(clusters_090)} clusters** "
                 f"containing **{sum(len(c) for c in clusters_090)} PRs**")
    lines.append(f"At 0.85: **{len(clusters_085)} clusters** "
                 f"({sum(len(c) for c in clusters_085)} PRs)")
    lines.append(f"At 0.80: **{len(clusters_080)} clusters** "
                 f"({sum(len(c) for c in clusters_080)} PRs)")
    lines.append(f"")

    if clusters_090:
        for i, cluster in enumerate(clusters_090, 1):
            # Use first PR as anchor label
            anchor = cluster[0]
            lines.append(f"**Cluster {i}: {anchor[1][:60]}**")
            lines.append(f"| PR | Title | Author | Similarity |")
            lines.append(f"|---|---|---|---|")
            for pr_num, title, author, sim in cluster:
                sim_str = "anchor" if sim == 0.0 else f"{sim:.3f}"
                lines.append(f"| #{pr_num} | {title[:70]} | {author} | {sim_str} |")
            lines.append(f"")

    # Extrapolation
    if total_open > n and rc > 0:
        dup_rate = rc / n
        estimated = int(total_open * dup_rate)
        lines.append(f"### Extrapolation")
        lines.append(f"")
        lines.append(f"If {dup_rate:.1%} of {n} PRs are duplicates, the full backlog of "
                     f"{total_open:,} PRs likely contains **~{estimated} closable duplicates**.")
        lines.append(f"")

    # Highest-risk PRs
    lines.append(f"### Highest-Risk PRs")
    lines.append(f"")
    lines.append(f"PRs with the most suspicion flags (sorted by HIGH-severity flag count):")
    lines.append(f"")
    lines.append(f"| PR | Title | Flags | Score | Risk Factors |")
    lines.append(f"|---|---|---|---|---|")
    for pr, hr in risk_ranked[:15]:
        if not hr.flags:
            break
        high = sum(1 for f in hr.flags if f.severity == FlagSeverity.HIGH)
        flag_names = ", ".join(f.rule_id for f in hr.flags)
        title_short = pr.title[:55] + ("..." if len(pr.title) > 55 else "")
        lines.append(f"| #{pr.number} | {title_short} | {len(hr.flags)} ({high}H) | "
                     f"{hr.suspicion_score:.2f} | {flag_names} |")
    lines.append(f"")

    # Flag frequency
    lines.append(f"### Flag Frequency")
    lines.append(f"")
    lines.append(f"| Flag | Count | % of PRs | What It Means |")
    lines.append(f"|---|---|---|---|")
    flag_descriptions = {
        "temporal_clustering": "Multiple new-account PRs within 24h window",
        "first_contribution": "No prior merged PRs on this repo",
        "sensitive_paths": "Touches security-relevant code paths",
        "low_test_ratio": "Code added without proportional tests",
        "new_account": "GitHub account < 90 days old",
        "large_diff_hiding": "Large PR with small sensitive changes buried in bulk",
        "unjustified_deps": "Dependency changes without explanation in PR body",
    }
    for flag_id, count in flag_counter.most_common():
        desc = flag_descriptions.get(flag_id, flag_id)
        lines.append(f"| {flag_id} | {count} | {count*100//n}% | {desc} |")
    lines.append(f"")

    # Contributor stats
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Contributor Profile")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| First-time contributors | {first_timers} ({first_timers*100//n}%) |")
    lines.append(f"| New accounts (< 90 days) | {new_accounts} ({new_accounts*100//n}%) |")
    lines.append(f"| Unique authors | {len(set(pr.author.login for pr in prs))} |")

    # Files touched
    sensitive_prs = sum(1 for _, hr in heuristic_results
                       if any(f.rule_id == "sensitive_paths" for f in hr.flags))
    no_test_prs = sum(1 for _, hr in heuristic_results
                      if any(f.rule_id == "low_test_ratio" for f in hr.flags))
    lines.append(f"| PRs touching sensitive paths | {sensitive_prs} ({sensitive_prs*100//n}%) |")
    lines.append(f"| PRs with low test ratio | {no_test_prs} ({no_test_prs*100//n}%) |")
    lines.append(f"")

    # What this means
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## What This Means for OpenClaw")
    lines.append(f"")
    lines.append(f"1. **{ft*100//n}% of PRs can be fast-tracked.** These passed all heuristic "
                 f"checks — no suspicious signals, no duplicates. Maintainers can review these "
                 f"with confidence that automated screening found nothing concerning.")
    lines.append(f"")
    lines.append(f"2. **The duplicate problem is real.** {len(clusters_090)} duplicate clusters "
                 f"at 0.90 threshold, {len(clusters_080)} at 0.80. Extrapolated across the full "
                 f"backlog, this represents hundreds of closable or consolidatable PRs.")
    lines.append(f"")
    lines.append(f"3. **{first_timers*100//n}% of PRs are from first-time contributors.** "
                 f"OpenClaw's growth is almost entirely drive-by contributions. "
                 f"This is common for viral OSS but makes review expensive.")
    lines.append(f"")
    if sensitive_prs > 0:
        lines.append(f"4. **{sensitive_prs*100//n}% of PRs touch security-sensitive paths.** "
                     f"Combined with the first-contribution rate, this means many PRs involve "
                     f"unknown people modifying critical code (auth, extensions, gateway, credentials).")
        lines.append(f"")
    if no_test_prs > 0:
        lines.append(f"5. **{no_test_prs*100//n}% of code PRs lack tests.** "
                     f"For a project with test requirements, this is a persistent enforcement gap.")
        lines.append(f"")

    # How it works
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## How to Use This")
    lines.append(f"")
    lines.append(f"This runs as a **free GitHub Action** on every new PR:")
    lines.append(f"")
    lines.append(f"```yaml")
    lines.append(f"- uses: pranayom/security_pr@v1")
    lines.append(f"  with:")
    lines.append(f"    github_token: ${{{{ secrets.GITHUB_TOKEN }}}}")
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"Or install via pip:")
    lines.append(f"")
    lines.append(f"```bash")
    lines.append(f"pip install \"mcp-ai-auditor[gatekeeper]\"")
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"Every PR gets a verdict (`FAST_TRACK`, `REVIEW_REQUIRED`, `RECOMMEND_CLOSE`) "
                 f"and specific flags explaining why. Maintainers remain the authority — "
                 f"the tool recommends, never auto-closes.")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Generated by [mcp-ai-auditor](https://github.com/pranayom/security_pr) v0.1.0 "
                 f"| Tiers 1+2 | $0 cost | {elapsed_seconds:.0f}s runtime*")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Batch PR triage")
    parser.add_argument("--owner", default="nicoseng", help="Repo owner")
    parser.add_argument("--repo", default="OpenClaw", help="Repo name")
    parser.add_argument("--count", type=int, default=300, help="Number of PRs to triage")
    parser.add_argument("--vision", default="", help="Path to vision document YAML")
    parser.add_argument("--output", default="", help="Output report path")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent API requests")
    args = parser.parse_args()

    start_time = time.time()

    # Load vision document if provided
    vision = None
    vision_name = ""
    if args.vision and os.path.exists(args.vision):
        vision = load_vision_document(args.vision)
        vision_name = os.path.basename(args.vision)
        print(f"Loaded vision document: {vision_name} ({len(vision.focus_areas)} focus areas)")

    async with GitHubClient(rate_limit_buffer=0) as client:
        # Step 1: Get PR numbers (also returns total open count)
        pr_numbers, total_open = await fetch_pr_numbers(client, args.owner, args.repo, args.count)

        # Step 2: Ingest PRs
        print(f"\nIngesting {len(pr_numbers)} PRs (concurrency={args.concurrency})...")
        prs = await ingest_batch_with_progress(
            client, args.owner, args.repo, pr_numbers, args.concurrency
        )

    # Step 3: Compute embeddings
    print(f"\n--- Tier 1: Dedup ---")
    embeddings = compute_embeddings_with_progress(prs)

    # Find clusters at multiple thresholds
    print("Finding duplicate clusters...")
    clusters_090 = find_duplicate_clusters(prs, embeddings, 0.90)
    clusters_085 = find_duplicate_clusters(prs, embeddings, 0.85)
    clusters_080 = find_duplicate_clusters(prs, embeddings, 0.80)
    print(f"  0.90: {len(clusters_090)} clusters ({sum(len(c) for c in clusters_090)} PRs)")
    print(f"  0.85: {len(clusters_085)} clusters ({sum(len(c) for c in clusters_085)} PRs)")
    print(f"  0.80: {len(clusters_080)} clusters ({sum(len(c) for c in clusters_080)} PRs)")

    # Step 4: Run heuristics
    print(f"\n--- Tier 2: Heuristics ---")
    focus_areas = vision.focus_areas if vision else None
    heuristic_results = run_all_heuristics(prs, focus_areas)
    gated = sum(1 for _, hr in heuristic_results if hr.outcome == TierOutcome.GATED)
    print(f"  GATED: {gated}, PASS: {len(prs) - gated}")

    # Step 5: Generate report
    elapsed = time.time() - start_time
    report = generate_report(
        args.owner, args.repo, prs, embeddings, heuristic_results,
        clusters_090, clusters_085, clusters_080,
        total_open, vision_name, elapsed,
    )

    # Save report
    output_path = args.output or f"reports/openclaw-triage-{datetime.now().strftime('%Y-%m-%d')}.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print(f"Total time: {elapsed:.0f}s")

    # Also dump raw data as JSON for reference
    json_path = output_path.replace(".md", ".json")
    raw_data = {
        "metadata": {
            "owner": args.owner,
            "repo": args.repo,
            "count": len(prs),
            "total_open": total_open,
            "elapsed_seconds": elapsed,
            "date": datetime.now().isoformat(),
        },
        "verdicts": {
            "fast_track": len([1 for _, hr in heuristic_results
                              if hr.outcome != TierOutcome.GATED
                              and not any(pr.number in {
                                  item[0] for c in clusters_090 for item in c[1:]
                              } for pr, _ in [(_, hr)])]),
            "review_required": gated,
            "recommend_close": sum(len(c) - 1 for c in clusters_090),
        },
        "clusters_090": [
            [{"pr": num, "title": title, "author": author, "similarity": round(sim, 4)}
             for num, title, author, sim in cluster]
            for cluster in clusters_090
        ],
        "highest_risk": [
            {
                "pr": pr.number,
                "title": pr.title,
                "author": pr.author.login,
                "score": round(hr.suspicion_score, 3),
                "flags": [f.rule_id for f in hr.flags],
            }
            for pr, hr in sorted(
                heuristic_results,
                key=lambda x: x[1].suspicion_score,
                reverse=True,
            )[:20]
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
