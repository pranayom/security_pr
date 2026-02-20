"""FastMCP server exposing OSS maintainer toolkit tools."""

import asyncio
import json

from mcp.server.fastmcp import FastMCP

from oss_maintainer_toolkit.scanners.vulnerability_scanner import scan_vulnerabilities
from oss_maintainer_toolkit.analysis.data_flow import trace_data_flow
from oss_maintainer_toolkit.cve.checker import check_cve

mcp = FastMCP("oss-maintainer-toolkit")


@mcp.tool()
def scan_vulnerabilities_tool(target: str) -> str:
    """Scan files for security vulnerabilities using regex pattern matching.

    Detects OWASP Top 10 patterns: SQL injection, command injection,
    hardcoded secrets, XSS, path traversal, insecure deserialization,
    weak cryptography, SSL issues, and debug mode.

    Args:
        target: Path to a file or directory to scan.
    """
    result = scan_vulnerabilities(target)
    return result.model_dump_json(indent=2)


@mcp.tool()
def trace_data_flow_tool(target: str) -> str:
    """Trace tainted data flows from user inputs to dangerous sinks.

    AST-based Python taint analysis tracking data from Flask/Django
    request objects and input() through variable assignments to sinks
    like SQL execution, OS commands, eval, and deserialization.

    Args:
        target: Path to a Python file or directory to analyze.
    """
    result = trace_data_flow(target)
    return result.model_dump_json(indent=2)


@mcp.tool()
async def check_cve_tool(target: str) -> str:
    """Check project dependencies for known CVEs via OSV.dev.

    Parses requirements.txt and package.json files, queries the free
    OSV.dev batch API for known vulnerabilities, and returns detailed
    CVE records with severity, affected versions, and fix versions.

    Args:
        target: Path to a dependency file or directory containing them.
    """
    result = await check_cve(target)
    return result.model_dump_json(indent=2)


@mcp.tool()
async def assess_contribution_risk_tool(
    owner: str,
    repo: str,
    pr_number: int,
    vision_document_path: str = "",
    enable_tier3: bool = True,
) -> str:
    """Assess a GitHub pull request for contribution risk using a three-tier gated pipeline.

    Tier 1: Embedding-based dedup (flags duplicate PRs).
    Tier 2: Heuristic suspicion scoring (new accounts, sensitive paths, dep changes, etc.).
    Tier 3: LLM vision alignment via OpenRouter free models or claude --print (optional).

    Returns a JSON scorecard with verdict (FAST_TRACK / REVIEW_REQUIRED / RECOMMEND_CLOSE),
    per-dimension scores, and specific flags with explanations.

    Args:
        owner: GitHub repo owner (e.g. "nicoseng").
        repo: GitHub repo name (e.g. "OpenClaw").
        pr_number: Pull request number.
        vision_document_path: Path to YAML vision document (optional, enables Tier 3).
        enable_tier3: Whether to run Tier 3 vision alignment (default True).
    """
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr
    from oss_maintainer_toolkit.gatekeeper.pipeline import run_pipeline

    async with GitHubClient() as client:
        pr = await ingest_pr(owner, repo, pr_number, client)

    scorecard = await run_pipeline(
        pr,
        vision_document_path=vision_document_path,
        enable_tier3=enable_tier3,
        llm_provider="",  # uses config default (openrouter)
    )
    return scorecard.model_dump_json(indent=2)


@mcp.tool()
async def triage_issue_tool(
    owner: str,
    repo: str,
    issue_number: int,
    vision_document_path: str = "",
    enable_tier3: bool = True,
) -> str:
    """Triage a GitHub issue using the three-tier gated pipeline.

    Tier 1: Embedding-based dedup (flags duplicate issues).
    Tier 2: Heuristic quality scoring (vague descriptions, new accounts, missing repro, etc.).
    Tier 3: LLM vision alignment via OpenRouter free models or claude --print (optional).

    Returns a JSON scorecard with verdict (FAST_TRACK / REVIEW_REQUIRED / RECOMMEND_CLOSE),
    per-dimension scores, and specific flags with explanations.

    Args:
        owner: GitHub repo owner (e.g. "nicoseng").
        repo: GitHub repo name (e.g. "OpenClaw").
        issue_number: Issue number.
        vision_document_path: Path to YAML vision document (optional, enables Tier 3).
        enable_tier3: Whether to run Tier 3 vision alignment (default True).
    """
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue
    from oss_maintainer_toolkit.gatekeeper.issue_pipeline import run_issue_pipeline

    async with GitHubClient() as client:
        issue = await ingest_issue(owner, repo, issue_number, client)

    scorecard = await run_issue_pipeline(
        issue,
        vision_document_path=vision_document_path,
        enable_tier3=enable_tier3,
        llm_provider="",
    )
    return scorecard.model_dump_json(indent=2)


@mcp.tool()
async def link_issues_to_prs_tool(
    owner: str,
    repo: str,
    threshold: float = 0.0,
) -> str:
    """Link GitHub issues to pull requests using embedding similarity.

    Pure Tier 1 analysis: computes cosine similarity between PR diffs and
    issue descriptions to suggest which PRs address which issues. Fills
    the gap when contributors don't write "Fixes #N" explicitly.

    Returns a JSON report with suggested links (sorted by similarity),
    explicit links already present, and orphan issues with no linked PRs.

    Args:
        owner: GitHub repo owner (e.g. "nicoseng").
        repo: GitHub repo name (e.g. "OpenClaw").
        threshold: Similarity threshold (0 = use config default of 0.45).
    """
    from oss_maintainer_toolkit.gatekeeper.cache import PRCache
    from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch
    from oss_maintainer_toolkit.gatekeeper.issue_cache import IssueCache
    from oss_maintainer_toolkit.gatekeeper.issue_dedup import compute_issue_embedding
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue_batch
    from oss_maintainer_toolkit.gatekeeper.linking import find_issue_pr_links
    from oss_maintainer_toolkit.gatekeeper.models import PRMetadata, IssueMetadata

    async with GitHubClient() as client:
        # Fetch all open PRs and issues
        raw_prs = await client.list_open_prs(owner, repo)
        raw_issues = await client.list_open_issues(owner, repo)

        pr_numbers = [p["number"] for p in raw_prs]
        issue_numbers = [i["number"] for i in raw_issues]

        # Batch ingest
        prs = list(await ingest_batch(owner, repo, pr_numbers, client))
        issues = list(await ingest_issue_batch(owner, repo, issue_numbers, client))

    # Compute embeddings
    pr_embeddings = [compute_embedding(pr) for pr in prs]
    issue_embeddings = [compute_issue_embedding(issue) for issue in issues]

    report = find_issue_pr_links(prs, pr_embeddings, issues, issue_embeddings, threshold)
    return report.model_dump_json(indent=2)


@mcp.tool()
async def detect_stale_items_tool(
    owner: str,
    repo: str,
    threshold: float = 0.0,
    inactive_days: int = 0,
    since_days: int = 90,
) -> str:
    """Detect semantically stale PRs and issues using embedding similarity + metadata.

    Four detection signals (Tier 1 + Tier 2 only, no LLM, $0 cost):
    - Superseded PRs: open PR similar to a recently merged PR.
    - Already-addressed issues: open issue similar to a merged PR.
    - Blocked PRs: open PR references still-open issues.
    - Inactive items: open PRs/issues with no activity beyond threshold.

    Returns a JSON report with all detected stale items grouped by signal.

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        threshold: Similarity threshold (0 = config default 0.75).
        inactive_days: Inactivity threshold in days (0 = config default 90).
        since_days: How far back to look for merged PRs (default 90 days).
    """
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch
    from oss_maintainer_toolkit.gatekeeper.issue_dedup import compute_issue_embedding
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue_batch
    from oss_maintainer_toolkit.gatekeeper.staleness import detect_stale_items

    async with GitHubClient() as client:
        raw_open_prs = await client.list_open_prs(owner, repo)
        raw_issues = await client.list_open_issues(owner, repo)
        raw_merged_prs = await client.list_recently_merged_prs(owner, repo, since_days)

        open_pr_numbers = [p["number"] for p in raw_open_prs]
        issue_numbers = [i["number"] for i in raw_issues]
        merged_pr_numbers = [p["number"] for p in raw_merged_prs]

        open_prs = list(await ingest_batch(owner, repo, open_pr_numbers, client))
        open_issues = list(await ingest_issue_batch(owner, repo, issue_numbers, client))
        merged_prs = list(await ingest_batch(owner, repo, merged_pr_numbers, client))

    open_pr_embeddings = [compute_embedding(pr) for pr in open_prs]
    open_issue_embeddings = [compute_issue_embedding(issue) for issue in open_issues]
    merged_pr_embeddings = [compute_embedding(pr) for pr in merged_prs]

    report = detect_stale_items(
        open_prs, open_pr_embeddings,
        open_issues, open_issue_embeddings,
        merged_prs, merged_pr_embeddings,
        threshold=threshold,
        inactive_days=inactive_days,
    )
    return report.model_dump_json(indent=2)


@mcp.tool()
async def classify_labels_tool(
    owner: str,
    repo: str,
    item_type: str,
    item_number: int,
    vision_document_path: str = "",
    threshold: float = 0.0,
) -> str:
    """Suggest labels for a GitHub PR or issue using embedding similarity + keyword heuristics.

    Tier 1 + Tier 2 only (no LLM, $0 cost). Classifies items against a label taxonomy
    sourced from the vision document and/or GitHub repo labels.

    Returns a JSON report with suggested labels sorted by confidence, including
    embedding similarity scores and keyword matches.

    Args:
        owner: GitHub repo owner (e.g. "nicoseng").
        repo: GitHub repo name (e.g. "OpenClaw").
        item_type: "pr" or "issue".
        item_number: PR or issue number.
        vision_document_path: Path to YAML vision document (optional, provides richer taxonomy).
        threshold: Minimum confidence threshold (0 = config default 0.35).
    """
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.labeling import (
        classify_item,
        compute_item_embedding,
        compute_label_embeddings,
        github_labels_to_taxonomy,
        merge_taxonomies,
    )

    # Load taxonomies
    vision_labels = []
    if vision_document_path:
        from oss_maintainer_toolkit.gatekeeper.vision import load_vision_document
        vision_doc = load_vision_document(vision_document_path)
        vision_labels = vision_doc.label_taxonomy

    async with GitHubClient() as client:
        # Fetch GitHub labels
        raw_labels = await client.list_repo_labels(owner, repo)
        github_labels = github_labels_to_taxonomy(raw_labels)

        # Fetch item
        if item_type == "pr":
            from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr
            item = await ingest_pr(owner, repo, item_number, client)
        else:
            from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue
            item = await ingest_issue(owner, repo, item_number, client)

    # Merge taxonomies
    taxonomy = merge_taxonomies(vision_labels, github_labels)

    if vision_labels and github_labels:
        taxonomy_source = "merged"
    elif vision_labels:
        taxonomy_source = "vision"
    else:
        taxonomy_source = "github"

    # Compute embeddings and classify
    item_embedding = compute_item_embedding(item)
    label_embeddings = compute_label_embeddings(taxonomy)
    report = classify_item(
        item, item_embedding, taxonomy, label_embeddings, threshold=threshold,
    )
    report.taxonomy_source = taxonomy_source
    return report.model_dump_json(indent=2)


@mcp.tool()
async def contributor_profile_tool(
    owner: str,
    repo: str,
    username: str,
    max_prs: int = 0,
) -> str:
    """Build a contributor profile from their PR history in a repository.

    Analyzes merge rate, test inclusion rate, areas of expertise, and activity
    patterns. Tier 1 + Tier 2 only (metadata analysis, no LLM, $0 cost).

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        username: GitHub username to profile.
        max_prs: Max PRs to analyze (0 = config default 50).
    """
    from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
    from oss_maintainer_toolkit.gatekeeper.contributor_profiles import build_contributor_profile
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch

    limit = max_prs if max_prs > 0 else gatekeeper_settings.contributor_max_prs

    async with GitHubClient() as client:
        raw_prs = await client.search_user_prs(owner, repo, username, max_results=limit)
        pr_numbers = [p["number"] for p in raw_prs]
        prs = list(await ingest_batch(owner, repo, pr_numbers, client))

    profile = build_contributor_profile(owner, repo, username, prs)
    return profile.model_dump_json(indent=2)


@mcp.tool()
async def suggest_reviewers_tool(
    owner: str,
    repo: str,
    pr_number: int,
    max_suggestions: int = 0,
) -> str:
    """Suggest reviewers for a GitHub PR based on CODEOWNERS and review history.

    Combines CODEOWNERS file matching and past review patterns on similar files.
    Tier 2 only (heuristic rules, no LLM, $0 cost).

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        pr_number: Pull request number.
        max_suggestions: Max reviewers to suggest (0 = config default 5).
    """
    from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr, ingest_batch
    from oss_maintainer_toolkit.gatekeeper.review_routing import parse_codeowners, suggest_reviewers

    async with GitHubClient() as client:
        pr = await ingest_pr(owner, repo, pr_number, client)

        # Try to load CODEOWNERS
        codeowners_rules = None
        for path in ["CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"]:
            content = await client.get_file_content(owner, repo, path)
            if content is not None:
                codeowners_rules = parse_codeowners(content)
                break

        # Fetch recent merged PRs for review history
        recent_limit = gatekeeper_settings.review_recent_prs
        raw_merged = await client.list_recently_merged_prs(owner, repo, since_days=90)
        merged_numbers = [p["number"] for p in raw_merged[:recent_limit]]
        merged_prs = list(await ingest_batch(owner, repo, merged_numbers, client))

        # Fetch reviews for merged PRs
        reviews_by_pr: dict[int, list[str]] = {}
        for mp in merged_prs:
            reviews = await client.list_pr_reviews(owner, repo, mp.number)
            reviewers = list({r["user"]["login"] for r in reviews if r.get("user")})
            if reviewers:
                reviews_by_pr[mp.number] = reviewers

    report = suggest_reviewers(
        pr,
        codeowners_rules=codeowners_rules,
        recent_prs=merged_prs,
        reviews_by_pr=reviews_by_pr,
        max_suggestions=max_suggestions,
    )
    return report.model_dump_json(indent=2)


@mcp.tool()
async def detect_conflicts_tool(
    owner: str,
    repo: str,
    threshold: float = 0.0,
    file_overlap_weight: float = 0.0,
) -> str:
    """Detect conflicting PR pairs via file overlap and embedding similarity.

    Identifies open PRs that modify overlapping files or similar code regions.
    Tier 1 + Tier 2 only (no LLM, $0 cost).

    Returns a JSON report with conflict pairs sorted by confidence.

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        threshold: Minimum confidence to report (0 = config default 0.3).
        file_overlap_weight: Weight for file overlap vs embedding (0 = config default 0.5).
    """
    from oss_maintainer_toolkit.gatekeeper.conflict_detection import detect_conflicts
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch

    async with GitHubClient() as client:
        raw_prs = await client.list_open_prs(owner, repo)
        pr_numbers = [p["number"] for p in raw_prs]
        prs = list(await ingest_batch(owner, repo, pr_numbers, client))

    embeddings = [compute_embedding(pr) for pr in prs]
    report = detect_conflicts(
        prs, embeddings,
        file_overlap_weight=file_overlap_weight,
        threshold=threshold,
    )
    return report.model_dump_json(indent=2)


@mcp.tool()
async def audit_backlog_tool(
    owner: str,
    repo: str,
    count: int = 100,
    concurrency: int = 3,
    vision_document_path: str = "",
) -> str:
    """Audit a repository's PR backlog using batch triage.

    Fetches the N most recent open PRs, runs embedding dedup (Tier 1) and
    heuristic rules (Tier 2), and produces a structured report with verdict
    distribution, duplicate clusters, highest-risk PRs, and contributor stats.

    Tier 1 + Tier 2 only (no LLM, $0 cost).

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        count: Number of PRs to analyze (default 100).
        concurrency: Concurrent API requests (default 3).
        vision_document_path: Path to YAML vision document (optional, adds focus areas).
    """
    from oss_maintainer_toolkit.gatekeeper.audit_backlog import run_audit
    from oss_maintainer_toolkit.gatekeeper.audit_scorecard import audit_report_to_markdown

    report = await run_audit(
        owner, repo,
        count=count,
        concurrency=concurrency,
        vision_document_path=vision_document_path,
    )
    return audit_report_to_markdown(report)


@mcp.tool()
async def generate_vision_tool(
    owner: str,
    repo: str,
    max_merged: int = 10,
    max_rejected: int = 10,
) -> str:
    """Generate a Vision Document for a GitHub repository using LLM analysis.

    Fetches README, CONTRIBUTING.md, and recent merged/rejected PRs to deduce
    the project's unwritten governance rules. Returns a YAML vision document.

    Requires an LLM API key configured via AUDITOR_GK_LLM_API_KEY or
    provider-specific keys (AUDITOR_GK_OPENROUTER_API_KEY, etc.).

    Args:
        owner: GitHub repo owner.
        repo: GitHub repo name.
        max_merged: Max merged PRs to analyze (default 10).
        max_rejected: Max rejected PRs to analyze (default 10).
    """
    from oss_maintainer_toolkit.gatekeeper.vision_generation import (
        generate_vision_document,
        vision_document_to_yaml,
    )

    doc = await generate_vision_document(
        owner, repo,
        max_merged=max_merged,
        max_rejected=max_rejected,
    )
    return vision_document_to_yaml(doc, owner, repo)


if __name__ == "__main__":
    mcp.run()
