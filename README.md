# PR Triage — Gatekeeper

A free GitHub Action that triages pull requests using a three-tier pipeline: embedding-based duplicate detection, suspicion heuristics, and optional LLM vision alignment.

**Every PR gets a verdict: `FAST_TRACK`, `REVIEW_REQUIRED`, or `RECOMMEND_CLOSE`.**

Tested on [OpenClaw](https://github.com/openclaw/openclaw) (3,368 open PRs): cut the maintainer review queue by 36% and found 6% duplicate PRs in 30 seconds. [See the full report.](https://gist.github.com/pranayom)

---

## Installation

```bash
# Core MCP tools (vulnerability scanner, data flow, CVE checker)
pip install mcp-ai-auditor

# With PR triage / gatekeeper pipeline
pip install "mcp-ai-auditor[gatekeeper]"

# For development
pip install -e ".[dev,gatekeeper]"
```

### CLI usage

```bash
auditor scan /path/to/project        # vulnerability scan
auditor trace /path/to/file.py       # data flow analysis
auditor cve /path/to/requirements.txt # CVE check
```

### MCP server

```bash
python -m mcp_ai_auditor.mcp         # start the MCP server
```

---

## Quick Start (GitHub Action)

Copy this workflow into `.github/workflows/pr-triage.yml` in your repo:

```yaml
name: PR Triage

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  pull-requests: write
  contents: read

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pranayom/security_pr@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

That's it. Every new PR gets a scorecard comment with a verdict and flags.

---

## How It Works

```
PR opened
    |
    v
[Tier 1: Embedding Dedup]     — sentence-transformers, cosine similarity
    |                            Duplicates -> RECOMMEND_CLOSE (stop)
    v
[Tier 2: Suspicion Heuristics] — 7 deterministic rules, weighted scoring
    |                            Flagged -> REVIEW_REQUIRED (stop)
    v
[Tier 3: Vision Alignment]     — LLM compares PR against Vision Document (optional)
    |
    v
FAST_TRACK
```

Tiers run strictly in sequence. Each tier is a gate — failures don't proceed to the next tier. This reserves LLM time for the minority of PRs where semantic judgment is actually useful.

### Tier 1 — Embedding Dedup (free, local)
Computes semantic embeddings for PR title + description + diff using `all-MiniLM-L6-v2`. Flags duplicates above a cosine similarity threshold (default: 0.90).

### Tier 2 — Suspicion Heuristics (free, deterministic)
Seven rules scored against PR metadata:

| Rule | What it catches |
|---|---|
| `new_account` | GitHub account < 90 days old |
| `first_contribution` | No previously merged PRs on this repo |
| `sensitive_paths` | Changes to auth, credentials, CI/CD, extensions |
| `low_test_ratio` | Code added without proportional tests |
| `unjustified_deps` | Dependency changes without explanation |
| `large_diff_hiding` | Large PR with small sensitive changes buried in bulk |
| `temporal_clustering` | Multiple new-account PRs within a short window |

### Tier 3 — Vision Alignment (optional, $0 via OpenRouter)
Compares the PR diff against your project's Vision Document (a YAML file defining principles, anti-patterns, and focus areas). Uses OpenRouter free models. Requires an `OPENROUTER_API_KEY` (free at [openrouter.ai/keys](https://openrouter.ai/keys)).

---

## Inputs

| Input | Required | Default | Description |
|---|---|---|---|
| `github_token` | Yes | — | GitHub token for API access (usually `secrets.GITHUB_TOKEN`) |
| `vision_document` | No | `""` | Path to YAML vision document (relative to repo root) |
| `openrouter_api_key` | No | `""` | OpenRouter API key for Tier 3 ($0 cost). Tier 3 skipped if not provided. |
| `openrouter_model` | No | `openai/gpt-oss-120b:free` | OpenRouter model for Tier 3 |
| `duplicate_threshold` | No | `0.9` | Cosine similarity threshold for duplicate detection |
| `suspicion_threshold` | No | `0.6` | Suspicion score threshold for flagging |
| `enforce_vision` | No | `false` | Enable Tier 3 vision alignment (set to `true` after reviewing your vision doc) |
| `post_comment` | No | `true` | Post scorecard as a PR comment |

## Outputs

| Output | Description |
|---|---|
| `verdict` | `FAST_TRACK`, `REVIEW_REQUIRED`, or `RECOMMEND_CLOSE` |
| `scorecard_json` | Full scorecard as JSON for downstream CI steps |

---

## Vision Documents

A Vision Document is an optional YAML file that defines what your project is trying to be. It enables Tier 3, where an LLM evaluates whether a PR aligns with your project's direction.

Example structure:

```yaml
project: my-project
principles:
  - name: "Security First"
    description: "All changes touching auth or credentials require security review"
  - name: "Test Everything"
    description: "Every feature PR must include tests"

anti_patterns:
  - "Adding dependencies without justification"
  - "Modifying CI/CD without maintainer approval"

focus_areas:
  - "src/auth/"
  - "src/credentials/"
  - ".github/"
```

Place it at `.github/vision.yaml` and set `vision_document: ".github/vision.yaml"` in the action inputs.

---

## Example Scorecard Comment

When the action runs on a PR, it posts a comment like:

> ## &#x26A0; PR Triage: **REVIEW REQUIRED**
>
> > First-time contributor modifying security-sensitive paths without tests.
>
> | Dimension | Score | Summary |
> |---|---|---|
> | Hygiene & Dedup | `++++++++--` 0.80 | No duplicates found |
> | Supply Chain Suspicion | `++++------` 0.40 | New account + sensitive paths |
>
> ### Flags
> - [**HIGH**] **Sensitive Paths**: PR modifies `src/auth/oauth.ts`, `src/credentials/store.ts`
> - [MEDIUM] **First Contribution**: No previously merged PRs from this author
> - [MEDIUM] **Low Test Ratio**: 245 lines added, 0 test lines

---

## Evidence: OpenClaw Triage

We ran this tool against 100 of OpenClaw's 3,368 open PRs:

| Verdict | Count | Meaning |
|---|---|---|
| FAST_TRACK | 64 (64%) | Safe for quick review |
| REVIEW_REQUIRED | 30 (30%) | Flagged — needs human attention |
| RECOMMEND_CLOSE | 6 (6%) | Likely duplicate |

- Found 3 duplicate clusters (6 PRs) at 0.90 threshold
- 89% of PRs from first-time contributors
- 40% touch security-sensitive paths
- Extrapolated: ~200 closable duplicates in the full backlog

---

## Cost

$0. All tiers run for free:
- Tier 1: `sentence-transformers` on CPU (GitHub Actions runner)
- Tier 2: Pure Python rules
- Tier 3: OpenRouter free models (optional, free API key)

---

## License

MIT
