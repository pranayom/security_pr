"""Typer CLI for oss-maintainer-toolkit."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from oss_maintainer_toolkit.scanners.vulnerability_scanner import scan_vulnerabilities
from oss_maintainer_toolkit.analysis.data_flow import trace_data_flow
from oss_maintainer_toolkit.cve.checker import check_cve

app = typer.Typer(
    name="maintainer",
    help="OSS Maintainer Toolkit — automated triage for PRs, issues, contributors, and review queues.",
)
console = Console()


@app.command()
def scan(target: str = typer.Argument(help="File or directory to scan")):
    """Scan files for security vulnerabilities."""
    result = scan_vulnerabilities(target)

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")

    if not result.findings:
        console.print(f"[green]No vulnerabilities found[/green] in {result.files_scanned} files.")
        return

    table = Table(title=f"Vulnerability Scan — {result.total_findings} findings in {result.files_scanned} files")
    table.add_column("Severity", style="bold")
    table.add_column("Category")
    table.add_column("File:Line")
    table.add_column("Description")

    severity_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "cyan",
        "info": "dim",
    }

    for f in sorted(result.findings, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity.value)):
        style = severity_styles.get(f.severity.value, "")
        table.add_row(
            f"[{style}]{f.severity.value.upper()}[/{style}]",
            f.category,
            f"{f.file}:{f.line}",
            f.description,
        )

    console.print(table)


@app.command()
def trace(target: str = typer.Argument(help="Python file or directory to analyze")):
    """Trace tainted data flows from sources to sinks."""
    result = trace_data_flow(target)

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")

    if not result.flows:
        console.print(f"[green]No taint flows found[/green] in {result.files_analyzed} files.")
        return

    table = Table(title=f"Data Flow Analysis — {result.total_flows} taint flows in {result.files_analyzed} files")
    table.add_column("Variable", style="bold cyan")
    table.add_column("Source")
    table.add_column("Sink")
    table.add_column("Description")

    for flow in result.flows:
        table.add_row(
            flow.variable,
            f"L{flow.source_line}: {flow.source_code[:60]}",
            f"L{flow.sink_line}: {flow.sink_code[:60]}",
            flow.description,
        )

    console.print(table)


@app.command()
def cve(target: str = typer.Argument(help="Dependency file or directory")):
    """Check dependencies for known CVEs."""
    result = asyncio.run(check_cve(target))

    if result.errors:
        for err in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {err}")

    if not result.vulnerabilities:
        console.print(f"[green]No known CVEs found[/green] in {result.dependencies_checked} dependencies.")
        return

    table = Table(title=f"CVE Check — {result.total_vulnerabilities} vulnerabilities in {result.dependencies_checked} dependencies")
    table.add_column("Severity", style="bold")
    table.add_column("CVE ID")
    table.add_column("Package")
    table.add_column("Version")
    table.add_column("Fixed In")
    table.add_column("Summary")

    severity_styles = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "cyan",
    }

    for v in sorted(result.vulnerabilities, key=lambda x: ["critical", "high", "medium", "low", "info"].index(x.severity.value)):
        style = severity_styles.get(v.severity.value, "")
        table.add_row(
            f"[{style}]{v.severity.value.upper()}[/{style}]",
            v.id,
            v.affected_package,
            v.affected_version,
            v.fixed_version or "—",
            v.summary[:80],
        )

    console.print(table)


@app.command()
def assess(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    pr_number: int = typer.Argument(help="Pull request number"),
    vision: str = typer.Option("", "--vision", help="Path to YAML vision document"),
    no_tier3: bool = typer.Option(False, "--no-tier3", help="Skip Tier 3 vision alignment"),
    provider: str = typer.Option("", "--provider", help="LLM provider: auto, openrouter, openai, anthropic, gemini, generic, claude_cli"),
    api_key: str = typer.Option("", "--api-key", help="Unified API key (auto-detects provider from prefix)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON scorecard"),
):
    """Assess a GitHub PR for contribution risk (three-tier gated pipeline)."""
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr
    from oss_maintainer_toolkit.gatekeeper.pipeline import run_pipeline
    from oss_maintainer_toolkit.gatekeeper.scorecard import render_scorecard, scorecard_to_json

    async def _run():
        async with GitHubClient() as client:
            pr = await ingest_pr(owner, repo, pr_number, client)

        scorecard = await run_pipeline(
            pr,
            vision_document_path=vision,
            enable_tier3=not no_tier3,
            llm_provider=provider,
            llm_api_key=api_key,
        )
        return scorecard

    scorecard = asyncio.run(_run())

    if json_output:
        console.print(scorecard_to_json(scorecard))
    else:
        render_scorecard(scorecard, console)


@app.command(name="triage-issue")
def triage_issue(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    issue_number: int = typer.Argument(help="Issue number"),
    vision: str = typer.Option("", "--vision", help="Path to YAML vision document"),
    no_tier3: bool = typer.Option(False, "--no-tier3", help="Skip Tier 3 vision alignment"),
    provider: str = typer.Option("", "--provider", help="LLM provider: auto, openrouter, openai, anthropic, gemini, generic, claude_cli"),
    api_key: str = typer.Option("", "--api-key", help="Unified API key (auto-detects provider from prefix)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON scorecard"),
):
    """Triage a GitHub issue (three-tier gated pipeline)."""
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue
    from oss_maintainer_toolkit.gatekeeper.issue_pipeline import run_issue_pipeline
    from oss_maintainer_toolkit.gatekeeper.issue_scorecard import issue_scorecard_to_json, render_issue_scorecard

    async def _run():
        async with GitHubClient() as client:
            issue = await ingest_issue(owner, repo, issue_number, client)

        scorecard = await run_issue_pipeline(
            issue,
            vision_document_path=vision,
            enable_tier3=not no_tier3,
            llm_provider=provider,
            llm_api_key=api_key,
        )
        return scorecard

    scorecard = asyncio.run(_run())

    if json_output:
        console.print(issue_scorecard_to_json(scorecard))
    else:
        render_issue_scorecard(scorecard, console)


@app.command(name="link-issues")
def link_issues(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    threshold: float = typer.Option(0.0, "--threshold", help="Similarity threshold (0 = config default 0.45)"),
    max_prs: int = typer.Option(0, "--max-prs", help="Max PRs to analyze (0 = all)"),
    max_issues: int = typer.Option(0, "--max-issues", help="Max issues to analyze (0 = all)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Link issues to PRs using embedding similarity (Tier 1 only)."""
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch
    from oss_maintainer_toolkit.gatekeeper.issue_dedup import compute_issue_embedding
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue_batch
    from oss_maintainer_toolkit.gatekeeper.linking import find_issue_pr_links
    from oss_maintainer_toolkit.gatekeeper.linking_scorecard import linking_report_to_json, render_linking_report

    async def _run():
        async with GitHubClient() as client:
            raw_prs = await client.list_open_prs(owner, repo)
            raw_issues = await client.list_open_issues(owner, repo)

            pr_numbers = [p["number"] for p in raw_prs]
            issue_numbers = [i["number"] for i in raw_issues]

            if max_prs > 0:
                pr_numbers = pr_numbers[:max_prs]
            if max_issues > 0:
                issue_numbers = issue_numbers[:max_issues]

            prs = list(await ingest_batch(owner, repo, pr_numbers, client))
            issues = list(await ingest_issue_batch(owner, repo, issue_numbers, client))

        pr_embeddings = [compute_embedding(pr) for pr in prs]
        issue_embeddings = [compute_issue_embedding(issue) for issue in issues]

        return find_issue_pr_links(prs, pr_embeddings, issues, issue_embeddings, threshold)

    report = asyncio.run(_run())

    if json_output:
        console.print(linking_report_to_json(report))
    else:
        render_linking_report(report, console)


@app.command(name="stale-detect")
def stale_detect(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    threshold: float = typer.Option(0.0, "--threshold", help="Similarity threshold (0 = config default 0.75)"),
    inactive_days: int = typer.Option(0, "--inactive-days", help="Inactivity threshold in days (0 = config default 90)"),
    since_days: int = typer.Option(90, "--since-days", help="How far back to look for merged PRs"),
    max_prs: int = typer.Option(0, "--max-prs", help="Max open PRs to analyze (0 = all)"),
    max_issues: int = typer.Option(0, "--max-issues", help="Max open issues to analyze (0 = all)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Detect semantically stale PRs and issues (Tier 1 + Tier 2 only, $0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch
    from oss_maintainer_toolkit.gatekeeper.issue_dedup import compute_issue_embedding
    from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue_batch
    from oss_maintainer_toolkit.gatekeeper.staleness import detect_stale_items
    from oss_maintainer_toolkit.gatekeeper.staleness_scorecard import (
        render_staleness_report,
        staleness_report_to_json,
    )

    async def _run():
        async with GitHubClient() as client:
            raw_open_prs = await client.list_open_prs(owner, repo)
            raw_issues = await client.list_open_issues(owner, repo)
            raw_merged_prs = await client.list_recently_merged_prs(owner, repo, since_days)

            open_pr_numbers = [p["number"] for p in raw_open_prs]
            issue_numbers = [i["number"] for i in raw_issues]
            merged_pr_numbers = [p["number"] for p in raw_merged_prs]

            if max_prs > 0:
                open_pr_numbers = open_pr_numbers[:max_prs]
            if max_issues > 0:
                issue_numbers = issue_numbers[:max_issues]

            open_prs = list(await ingest_batch(owner, repo, open_pr_numbers, client))
            open_issues = list(await ingest_issue_batch(owner, repo, issue_numbers, client))
            merged_prs = list(await ingest_batch(owner, repo, merged_pr_numbers, client))

        open_pr_embeddings = [compute_embedding(pr) for pr in open_prs]
        open_issue_embeddings = [compute_issue_embedding(issue) for issue in open_issues]
        merged_pr_embeddings = [compute_embedding(pr) for pr in merged_prs]

        return detect_stale_items(
            open_prs, open_pr_embeddings,
            open_issues, open_issue_embeddings,
            merged_prs, merged_pr_embeddings,
            threshold=threshold,
            inactive_days=inactive_days,
        )

    report = asyncio.run(_run())

    if json_output:
        console.print(staleness_report_to_json(report))
    else:
        render_staleness_report(report, console)


if __name__ == "__main__":
    app()
