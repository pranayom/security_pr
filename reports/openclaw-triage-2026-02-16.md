# OpenClaw PR Triage Report

**Date:** 2026-02-16
**Tool:** mcp-ai-auditor Gatekeeper Module (Tiers 1+2)
**Scope:** 100 most recent open PRs out of 3,368 total

---

## The Problem

OpenClaw has **3,368 open PRs**. Every open PR costs maintainer time to evaluate. Many are duplicates, many lack tests, and some touch security-critical paths from brand-new accounts. With Peter's transition to OpenAI and the project moving to a foundation, there is no bandwidth to triage this manually.

## What We Did

We ran 100 recent open PRs through an automated two-tier triage pipeline:

- **Tier 1 (Embedding Dedup):** Computed semantic embeddings for each PR's title, description, diff, and changed files using `all-MiniLM-L6-v2`. Compared all pairs by cosine similarity to find duplicates.
- **Tier 2 (Suspicion Heuristics):** Applied 7 deterministic rules against PR metadata: account age, contribution history, sensitive path changes, test-to-code ratio, unjustified dependency changes, large-diff hiding patterns, and temporal clustering.

No LLM calls were used. Everything runs locally, costs $0, and completes in ~30 seconds for 100 PRs.

---

## Results

### Verdict Distribution

| Verdict | Count | % | Meaning |
|---|---|---|---|
| **FAST_TRACK** | 64 | 64% | Passed all checks. Safe for quick review. |
| **REVIEW_REQUIRED** | 30 | 30% | Flagged by heuristics. Needs human attention. |
| **RECOMMEND_CLOSE** | 6 | 6% | Likely duplicate. Consider closing. |

A maintainer reviewing this batch could immediately **skip 64 PRs** that the tool cleared, focus attention on 30 that have specific flags, and close 6 duplicates.

### Duplicate Clusters Found

At the default threshold (0.90 cosine similarity), we found **3 duplicate clusters** containing **6 PRs**:

**Cluster 1: Telegram hyphen-to-underscore fix**
| PR | Title | Author | Similarity |
|---|---|---|---|
| #18711 | fix(telegram): replace hyphens with underscores in export-session | MisterGuy420 | anchor |
| #18704 | fix(telegram): normalize command names by replacing hyphens | Limitless2023 | 0.905 |

Two different contributors independently submitted the same Telegram fix.

**Cluster 2: Slack arg menu enhancements**
| PR | Title | Author | Similarity |
|---|---|---|---|
| #18458 | Slack: add header and context blocks to arg menus | - | anchor |
| #18439 | Slack: add overflow menus for slash arg choices | - | 0.903 |

**Cluster 3: Slack interaction normalization**
| PR | Title | Author | Similarity |
|---|---|---|---|
| #18423 | Slack: dedupe normalized interaction selections | - | anchor |
| #18413 | Slack: expand interaction payload normalization coverage | - | 0.907 |

At a slightly lower threshold (0.85), we found **5 clusters**. At 0.80, we found a **mega-cluster of 6 Slack PRs** all doing interaction payload normalization work that should likely be coordinated or consolidated.

### Extrapolation

If 6% of 100 PRs are duplicates, the full backlog of 3,368 PRs likely contains **~200 closable duplicates**. At the 0.80 threshold, there are likely **~300+ consolidation opportunities**.

### Highest-Risk PRs

These PRs combine multiple suspicion signals and warrant priority review:

| PR | Title | Flags | Risk Factors |
|---|---|---|---|
| #18733 | feat(infra): add LLM endpoint concurrency limits | 5 flags | New account + sensitive paths + large diff hiding |
| #18705 | Community improvements | 4 flags | 1,837 additions across 64 files, sensitive changes hidden in bulk |
| #18668 | feat(ui/chat): chat UX overhaul | 5 flags | Large diff with hidden sensitive changes, no tests |
| #18662 | feat(ui): 3-state mode system | 5 flags | Large diff with hidden sensitive changes, no tests |
| #18657 | feat(mattermost): add native slash command support | 5 flags | New account (< 90 days) + first contribution + sensitive paths |
| #18287 | fix(logging): improved redaction for config | 5 flags | New account + sensitive paths + large diff hiding |
| #18675 | feat: Comprehensive Security Implementation | 3 flags | First-time contributor submitting security code |

PR #18675 is particularly notable: a first-time contributor submitting a "Comprehensive Security Implementation" that touches sensitive paths. This is a textbook supply chain attack pattern (regardless of whether this specific PR is malicious).

### Flag Frequency

| Flag | Count | % of PRs | What It Means |
|---|---|---|---|
| temporal_clustering | 94 | 94% | High-volume repo: many new-account PRs in every 24h window |
| first_contribution | 89 | 89% | Most contributors have never had a PR merged here |
| sensitive_paths | 40 | 40% | PR touches security-relevant code (auth, extensions, gateway, credentials) |
| low_test_ratio | 22 | 22% | Code added without proportional tests |
| new_account | 18 | 18% | GitHub account created less than 90 days ago |
| large_diff_hiding | 5 | 5% | Large PR with small sensitive changes buried in bulk |

---

## What This Means for OpenClaw

1. **The duplicate problem is real and measurable.** Even in 100 PRs we found 6 clear duplicates and 10+ consolidation candidates. Across 3,368 open PRs, this likely represents hundreds of closable items.

2. **89% of PRs are from first-time contributors.** OpenClaw's growth is almost entirely drive-by contributions. This is common for viral OSS projects but makes review expensive because every PR requires evaluating an unknown contributor.

3. **40% of PRs touch security-sensitive paths** (extensions, gateway, credentials, auth). Combined with the first-contribution rate, this means nearly half the PR queue involves unknown people modifying critical code.

4. **22% of code PRs have no tests.** For a project that requires `pnpm test` to pass, this represents a test enforcement gap.

5. **The Slack interaction PRs need coordination.** Multiple contributors are independently working on Slack interaction normalization. A tracking issue or feature branch could consolidate this work.

---

## How This Tool Works

```
PR arrives
    |
    v
[Tier 1: Embedding Dedup]  -- sentence-transformers, cosine similarity
    |                         Duplicates -> RECOMMEND_CLOSE (stop)
    v
[Tier 2: Heuristics]       -- 7 deterministic rules, weighted scoring
    |                         Score >= 0.6 -> REVIEW_REQUIRED (stop)
    v
[Tier 3: Vision Alignment] -- Claude --print (optional, local only)
    |                         Compares PR against project Vision Document
    v
    FAST_TRACK
```

Tiers 1+2 are free, deterministic, and run in seconds. Tier 3 is optional and uses a local Claude subscription for semantic analysis (not used in this report).

---

## Next Steps

- **Run on full 3,368 PR backlog** to find all duplicates and consolidation opportunities
- **Vision Document review** — we drafted a Vision Document for OpenClaw covering 7 principles, 15 anti-patterns, and 15 security-sensitive path patterns. We'd welcome maintainer feedback.
- **GitHub Action** — package Tiers 1+2 as a GitHub Action that runs on every new PR, so duplicates and high-risk PRs are flagged automatically before a maintainer ever sees them.

---

*Generated by [mcp-ai-auditor](https://github.com/anthropics/mcp-ai-auditor) Gatekeeper Module*
