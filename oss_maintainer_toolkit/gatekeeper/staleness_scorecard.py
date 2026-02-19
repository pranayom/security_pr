"""Staleness report formatting — JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import StalenessReport


def staleness_report_to_json(report: StalenessReport) -> str:
    """Serialize staleness report to JSON."""
    return report.model_dump_json(indent=2)


def render_staleness_report(report: StalenessReport, console: Console | None = None) -> None:
    """Render a Rich-formatted staleness report to the console."""
    if console is None:
        console = Console()

    total_stale = (
        len(report.superseded_prs)
        + len(report.addressed_issues)
        + len(report.blocked_prs)
        + len(report.inactive_prs)
        + len(report.inactive_issues)
    )

    header = (
        f"[bold]Smart Stale Detection Report[/bold]\n\n"
        f"Repo: {report.owner}/{report.repo}\n"
        f"Open PRs: {report.total_open_prs}  |  Open Issues: {report.total_open_issues}  |  "
        f"Merged PRs checked: {report.total_merged_prs_checked}\n"
        f"Similarity threshold: {report.threshold:.2f}  |  "
        f"Inactive days: {report.inactive_days}\n"
        f"Total stale items: {total_stale}"
    )
    console.print(Panel(header, title="Smart Stale Detection", border_style="yellow"))

    # Superseded PRs
    if report.superseded_prs:
        table = Table(title="Superseded PRs")
        table.add_column("Open PR #", style="bold cyan")
        table.add_column("Superseded By", style="bold green")
        table.add_column("Similarity")
        table.add_column("Title")

        for item in report.superseded_prs:
            sim_str = f"{item.similarity:.3f}"
            table.add_row(
                f"#{item.number}",
                f"#{item.related_number}",
                sim_str,
                item.title[:50],
            )
        console.print(table)

    # Already-addressed issues
    if report.addressed_issues:
        table = Table(title="Already-Addressed Issues")
        table.add_column("Issue #", style="bold cyan")
        table.add_column("Addressed By PR", style="bold green")
        table.add_column("Similarity")
        table.add_column("Title")

        for item in report.addressed_issues:
            sim_str = f"{item.similarity:.3f}"
            table.add_row(
                f"#{item.number}",
                f"#{item.related_number}",
                sim_str,
                item.title[:50],
            )
        console.print(table)

    # Blocked PRs
    if report.blocked_prs:
        table = Table(title="Blocked PRs")
        table.add_column("PR #", style="bold cyan")
        table.add_column("Blocked By Issue", style="bold red")
        table.add_column("Explanation")

        for item in report.blocked_prs:
            table.add_row(
                f"#{item.number}",
                f"#{item.related_number}",
                item.explanation[:60],
            )
        console.print(table)

    # Inactive PRs
    if report.inactive_prs:
        table = Table(title="Inactive PRs")
        table.add_column("PR #", style="bold cyan")
        table.add_column("Last Activity", style="dim")
        table.add_column("Title")

        for item in report.inactive_prs:
            last = item.last_activity.strftime("%Y-%m-%d") if item.last_activity else "—"
            table.add_row(
                f"#{item.number}",
                last,
                item.title[:50],
            )
        console.print(table)

    # Inactive Issues
    if report.inactive_issues:
        table = Table(title="Inactive Issues")
        table.add_column("Issue #", style="bold cyan")
        table.add_column("Last Activity", style="dim")
        table.add_column("Title")

        for item in report.inactive_issues:
            last = item.last_activity.strftime("%Y-%m-%d") if item.last_activity else "—"
            table.add_row(
                f"#{item.number}",
                last,
                item.title[:50],
            )
        console.print(table)

    if total_stale == 0:
        console.print("[green]No stale items detected.[/green]")
