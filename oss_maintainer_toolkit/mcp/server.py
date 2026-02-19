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


if __name__ == "__main__":
    mcp.run()
