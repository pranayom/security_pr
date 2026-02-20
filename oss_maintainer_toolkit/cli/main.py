"""Typer CLI for oss-maintainer-toolkit."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from typing import Optional

from typing import Optional

app = typer.Typer(
    name="maintainer",
    help="OSS Maintainer Toolkit — automated triage for PRs, issues, contributors, and review queues.",
)
console = Console()


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


@app.command(name="classify-labels")
def classify_labels(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    item_type: str = typer.Argument(help="'pr' or 'issue'"),
    item_number: int = typer.Argument(help="PR or issue number"),
    vision: str = typer.Option("", "--vision", help="Path to YAML vision document"),
    threshold: float = typer.Option(0.0, "--threshold", help="Min confidence (0 = config default 0.35)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Suggest labels for a PR or issue (Tier 1 + Tier 2, $0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.labeling import (
        classify_item,
        compute_item_embedding,
        compute_label_embeddings,
        github_labels_to_taxonomy,
        merge_taxonomies,
    )
    from oss_maintainer_toolkit.gatekeeper.labeling_scorecard import (
        labeling_report_to_json,
        render_labeling_report,
    )

    async def _run():
        vision_labels = []
        if vision:
            from oss_maintainer_toolkit.gatekeeper.vision import load_vision_document
            vision_doc = load_vision_document(vision)
            vision_labels = vision_doc.label_taxonomy

        async with GitHubClient() as client:
            raw_labels = await client.list_repo_labels(owner, repo)
            github_labels = github_labels_to_taxonomy(raw_labels)

            if item_type == "pr":
                from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr
                item = await ingest_pr(owner, repo, item_number, client)
            else:
                from oss_maintainer_toolkit.gatekeeper.issue_ingest import ingest_issue
                item = await ingest_issue(owner, repo, item_number, client)

        taxonomy = merge_taxonomies(vision_labels, github_labels)

        if vision_labels and github_labels:
            taxonomy_source = "merged"
        elif vision_labels:
            taxonomy_source = "vision"
        else:
            taxonomy_source = "github"

        item_embedding = compute_item_embedding(item)
        label_embeddings = compute_label_embeddings(taxonomy)
        report = classify_item(
            item, item_embedding, taxonomy, label_embeddings, threshold=threshold,
        )
        report.taxonomy_source = taxonomy_source
        return report

    report = asyncio.run(_run())

    if json_output:
        console.print(labeling_report_to_json(report))
    else:
        render_labeling_report(report, console)


@app.command(name="contributor-profile")
def contributor_profile(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    username: str = typer.Argument(help="GitHub username to profile"),
    max_prs: int = typer.Option(0, "--max-prs", help="Max PRs to analyze (0 = config default 50)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON profile"),
):
    """Build a contributor profile from PR history (Tier 1 + Tier 2, $0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
    from oss_maintainer_toolkit.gatekeeper.contributor_profiles import build_contributor_profile
    from oss_maintainer_toolkit.gatekeeper.contributor_scorecard import (
        contributor_profile_to_json,
        render_contributor_profile,
    )
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch

    async def _run():
        limit = max_prs if max_prs > 0 else gatekeeper_settings.contributor_max_prs
        async with GitHubClient() as client:
            raw_prs = await client.search_user_prs(owner, repo, username, max_results=limit)
            pr_numbers = [p["number"] for p in raw_prs]
            prs = list(await ingest_batch(owner, repo, pr_numbers, client))
        return build_contributor_profile(owner, repo, username, prs)

    profile = asyncio.run(_run())

    if json_output:
        console.print(contributor_profile_to_json(profile))
    else:
        render_contributor_profile(profile, console)


@app.command(name="suggest-reviewers")
def suggest_reviewers_cmd(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    pr_number: int = typer.Argument(help="Pull request number"),
    max_suggestions: int = typer.Option(0, "--max", help="Max reviewers (0 = config default 5)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Suggest reviewers for a PR based on CODEOWNERS and review history ($0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_pr, ingest_batch
    from oss_maintainer_toolkit.gatekeeper.review_routing import parse_codeowners, suggest_reviewers
    from oss_maintainer_toolkit.gatekeeper.review_routing_scorecard import (
        render_review_routing_report,
        review_routing_report_to_json,
    )

    async def _run():
        async with GitHubClient() as client:
            pr = await ingest_pr(owner, repo, pr_number, client)

            codeowners_rules = None
            for path in ["CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"]:
                content = await client.get_file_content(owner, repo, path)
                if content is not None:
                    codeowners_rules = parse_codeowners(content)
                    break

            recent_limit = gatekeeper_settings.review_recent_prs
            raw_merged = await client.list_recently_merged_prs(owner, repo, since_days=90)
            merged_numbers = [p["number"] for p in raw_merged[:recent_limit]]
            merged_prs = list(await ingest_batch(owner, repo, merged_numbers, client))

            reviews_by_pr: dict[int, list[str]] = {}
            for mp in merged_prs:
                reviews = await client.list_pr_reviews(owner, repo, mp.number)
                reviewers = list({r["user"]["login"] for r in reviews if r.get("user")})
                if reviewers:
                    reviews_by_pr[mp.number] = reviewers

        return suggest_reviewers(
            pr,
            codeowners_rules=codeowners_rules,
            recent_prs=merged_prs,
            reviews_by_pr=reviews_by_pr,
            max_suggestions=max_suggestions,
        )

    report = asyncio.run(_run())

    if json_output:
        console.print(review_routing_report_to_json(report))
    else:
        render_review_routing_report(report, console)


@app.command(name="audit-backlog")
def audit_backlog(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    count: int = typer.Option(100, "--count", "-n", help="Number of PRs to analyze"),
    concurrency: int = typer.Option(3, "--concurrency", help="Concurrent API requests"),
    vision: str = typer.Option("", "--vision", help="Path to YAML vision document"),
    output: str = typer.Option("", "--output", "-o", help="Save markdown report to file"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Audit a repository's PR backlog — batch triage with dedup + heuristics ($0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.audit_backlog import run_audit
    from oss_maintainer_toolkit.gatekeeper.audit_scorecard import (
        audit_report_to_json,
        audit_report_to_markdown,
        render_audit_report,
    )

    async def _run():
        return await run_audit(
            owner, repo,
            count=count,
            concurrency=concurrency,
            vision_document_path=vision,
        )

    try:
        report = asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if output:
        md = audit_report_to_markdown(report)
        with open(output, "w", encoding="utf-8") as f:
            f.write(md)
        console.print(f"[green]Report written to {output}[/green]")
    elif json_output:
        console.print(audit_report_to_json(report))
    else:
        render_audit_report(report, console)


@app.command(name="generate-vision")
def generate_vision(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout)"),
    provider: str = typer.Option("", "--provider", help="LLM provider: auto, openrouter, openai, anthropic, gemini, generic"),
    api_key: str = typer.Option("", "--api-key", help="Unified API key (auto-detects provider from prefix)"),
    max_merged: int = typer.Option(10, "--max-merged", help="Max merged PRs to analyze"),
    max_rejected: int = typer.Option(10, "--max-rejected", help="Max rejected PRs to analyze"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON instead of YAML"),
):
    """Generate a Vision Document for a GitHub repository using LLM analysis."""
    from oss_maintainer_toolkit.gatekeeper.vision_generation import (
        generate_vision_document,
        vision_document_to_yaml,
    )

    async def _run():
        return await generate_vision_document(
            owner, repo,
            provider=provider,
            api_key=api_key,
            max_merged=max_merged,
            max_rejected=max_rejected,
        )

    try:
        doc = asyncio.run(_run())
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if json_output:
        result = doc.model_dump_json(indent=2)
    else:
        result = vision_document_to_yaml(doc, owner, repo)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result)
        console.print(f"[green]Vision document written to {output}[/green]")
    else:
        console.print(result)


@app.command(name="detect-conflicts")
def detect_conflicts_cmd(
    owner: str = typer.Argument(help="GitHub repo owner"),
    repo: str = typer.Argument(help="GitHub repo name"),
    threshold: float = typer.Option(0.0, "--threshold", help="Min confidence (0 = config default 0.3)"),
    file_overlap_weight: float = typer.Option(0.0, "--file-weight", help="File overlap weight (0 = config default 0.5)"),
    max_prs: int = typer.Option(0, "--max-prs", help="Max open PRs to analyze (0 = all)"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON report"),
):
    """Detect conflicting PR pairs via file overlap + embedding similarity ($0 cost)."""
    from oss_maintainer_toolkit.gatekeeper.conflict_detection import detect_conflicts
    from oss_maintainer_toolkit.gatekeeper.conflict_scorecard import (
        conflict_report_to_json,
        render_conflict_report,
    )
    from oss_maintainer_toolkit.gatekeeper.dedup import compute_embedding
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
    from oss_maintainer_toolkit.gatekeeper.ingest import ingest_batch

    async def _run():
        async with GitHubClient() as client:
            raw_prs = await client.list_open_prs(owner, repo)
            pr_numbers = [p["number"] for p in raw_prs]
            if max_prs > 0:
                pr_numbers = pr_numbers[:max_prs]
            prs = list(await ingest_batch(owner, repo, pr_numbers, client))

        embeddings = [compute_embedding(pr) for pr in prs]
        return detect_conflicts(
            prs, embeddings,
            file_overlap_weight=file_overlap_weight,
            threshold=threshold,
        )

    report = asyncio.run(_run())

    if json_output:
        console.print(conflict_report_to_json(report))
    else:
        render_conflict_report(report, console)


if __name__ == "__main__":
    app()
