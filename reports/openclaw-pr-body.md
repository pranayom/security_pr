Adds a GitHub Action that runs a three-tier analysis on every new PR:
- Tier 1: Embedding-based duplicate detection (sentence-transformers, cosine similarity)
- Tier 2: Heuristic suspicion scoring (account age, sensitive paths, test ratio, dependency changes)
- Tier 3: LLM vision alignment against project principles (optional, free via OpenRouter)

Posts a scorecard comment with verdict and flags on each PR. Includes an OpenClaw-specific vision document covering 7 principles, 15 anti-patterns, and 15 security-sensitive path patterns.

## Summary

- **Problem:** OpenClaw has 3,368 open PRs. 89% are from first-time contributors, 40% touch security-sensitive paths, and duplicates are submitted independently. No automated triage exists.
- **Why it matters:** Every open PR costs maintainer time. With the leadership transition and foundation move, there is no bandwidth to triage manually. I ran this tool against 100 recent open PRs and found 6 clear duplicates (extrapolating to ~200 across the full backlog), 30 PRs needing closer review, and 64 that could be fast-tracked.
- **What changed:** Added `.github/workflows/pr-triage.yml` (workflow) and `.github/openclaw-vision.yaml` (project vision document). The workflow runs an external action (`pranayom/security_pr@master`) on every `pull_request` event.
- **What did NOT change (scope boundary):** No changes to source code, tests, build config, or dependencies. This only adds CI workflow and a YAML config file.

## Change Type (select all)

- [ ] Bug fix
- [x] Feature
- [ ] Refactor
- [ ] Docs
- [ ] Security hardening
- [x] Chore/infra

## Scope (select all touched areas)

- [ ] Gateway / orchestration
- [ ] Skills / tool execution
- [ ] Auth / tokens
- [ ] Memory / storage
- [ ] Integrations
- [ ] API / contracts
- [ ] UI / DX
- [x] CI/CD / infra

## Linked Issue/PR

- Related: addresses PR backlog triage at scale

## User-visible / Behavior Changes

- New PR comment posted automatically on every pull request with a triage scorecard (verdict + flags)
- No changes to existing behavior, builds, or tests

## Security Impact (required)

- New permissions/capabilities? `Yes` — workflow requests `pull-requests: write` to post PR comments
- Secrets/tokens handling changed? `Yes` — optional `OPENROUTER_API_KEY` secret for Tier 3 (free LLM). Not required; Tier 3 is skipped without it.
- New/changed network calls? `Yes` — the action calls GitHub API (PR metadata/diff) and optionally OpenRouter API (vision alignment)
- Command/tool execution surface changed? `No`
- Data access scope changed? `No` — only reads public PR metadata and diffs already visible to anyone
- **Risk + mitigation:** The workflow runs an external action from `pranayom/security_pr@master`. Maintainers should review that repo and consider pinning to a specific commit SHA for supply chain safety. The action only reads PR data and posts a comment — it does not modify code, merge, or close PRs.

## Repro + Verification

### Environment

- OS: Ubuntu (GitHub Actions runner)
- Runtime/container: Python 3.11
- Model/provider: OpenRouter `openai/gpt-oss-120b:free` (optional, for Tier 3)
- Integration/channel: GitHub Actions
- Relevant config: `.github/workflows/pr-triage.yml`, `.github/openclaw-vision.yaml`

### Steps

1. Open a new PR on this repo (or push to an existing PR)
2. The `PR Triage` workflow triggers automatically
3. A comment is posted on the PR with the scorecard

### Expected

- Comment appears with verdict (FAST_TRACK / REVIEW_REQUIRED / RECOMMEND_CLOSE), dimension scores, and any flags

### Actual

- Tested locally against PR #18675 ("Comprehensive Security Implementation"):
  - Full pipeline (Tiers 1+2+3): **REVIEW_REQUIRED** — flagged for "Architecture-First Contribution" violation (large architectural PR from first-time contributor without prior discussion)
  - Tiers 1+2 only: **FAST_TRACK**

## Evidence

- [x] Trace/log snippets — ran pipeline against 100 real OpenClaw PRs: 64% fast-track, 30% review required, 6% recommend close. Found 3 duplicate clusters at 0.90 threshold.
- [x] Perf numbers — 100 PRs assessed in ~30 seconds (Tiers 1+2). Tier 3 adds ~10s per PR.

### Key findings from 100-PR analysis:

| Flag | Frequency | Meaning |
|---|---|---|
| temporal_clustering | 94% | Many new-account PRs in every 24h window |
| first_contribution | 89% | Most contributors have never had a PR merged |
| sensitive_paths | 40% | PR touches auth, extensions, gateway, credentials |
| low_test_ratio | 22% | Code added without proportional tests |
| new_account | 18% | GitHub account < 90 days old |
| large_diff_hiding | 5% | Large PR with sensitive changes buried in bulk |

## Human Verification (required)

- Verified scenarios: Ran full pipeline locally against PR #18675, #18733, #18705 and 97 other real open PRs. Verified all three operating modes (full pipeline, Tiers 1+2 + vision, Tiers 1+2 only).
- Edge cases checked: Draft PRs (406 on diff → graceful fallback), rate-limited GitHub Search API (graceful degradation), missing OpenRouter key (Tier 3 skipped cleanly)
- What I did **not** verify: The workflow has not been tested running inside GitHub Actions on this repo (only locally). First real run will be on this PR itself.

## Compatibility / Migration

- Backward compatible? `Yes` — adds new files only, no existing files modified
- Config/env changes? `Yes` — optional `OPENROUTER_API_KEY` repo secret for Tier 3
- Migration needed? `No`

## Failure Recovery (if this breaks)

- How to disable/revert this change quickly: Delete `.github/workflows/pr-triage.yml` or add `if: false` to the job
- Files/config to restore: Remove `.github/workflows/pr-triage.yml` and `.github/openclaw-vision.yaml`
- Known bad symptoms: If the external action (`pranayom/security_pr`) is unavailable, the workflow will fail but will NOT block PR merges (no required status check configured)

## Risks and Mitigations

- Risk: External action dependency (`pranayom/security_pr@master`) — supply chain concern
  - Mitigation: Maintainers can pin to a specific commit SHA, or vendor the action into this repo. The action source is fully open and auditable.
- Risk: Vision document may not perfectly reflect maintainer intent
  - Mitigation: The vision document is a draft. Feedback and corrections are welcome — it's a YAML file maintainers can edit directly.
- Risk: Free OpenRouter models may become unavailable or rate-limited
  - Mitigation: Model is configurable via `openrouter_model` input. Tier 3 is optional; Tiers 1+2 work without it.
