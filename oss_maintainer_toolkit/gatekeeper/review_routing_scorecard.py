"""Review routing report formatting -- JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import ReviewRoutingReport


def review_routing_report_to_json(report: ReviewRoutingReport) -> str:
    """Serialize review routing report to JSON."""
    return report.model_dump_json(indent=2)


def render_review_routing_report(report: ReviewRoutingReport, console: Console | None = None) -> None:
    """Render a Rich-formatted review routing report to the console."""
    if console is None:
        console = Console()

    codeowners_str = "Yes" if report.codeowners_found else "No"
    header = (
        f"[bold]Review Routing Report[/bold]\n\n"
        f"Repo: {report.owner}/{report.repo}\n"
        f"PR #{report.pr_number}: {report.pr_title}\n"
        f"Changed files: {len(report.changed_files)}  |  "
        f"CODEOWNERS: {codeowners_str}  |  "
        f"Recent PRs checked: {report.recent_reviewers_checked}\n"
        f"Suggestions: {len(report.suggestions)}"
    )
    console.print(Panel(header, title="Review Routing", border_style="blue"))

    if report.suggestions:
        table = Table(title="Suggested Reviewers")
        table.add_column("Reviewer", style="bold cyan")
        table.add_column("Score")
        table.add_column("Reasons")

        for s in report.suggestions:
            score_str = f"{s.score:.2f}"
            if s.score >= 0.7:
                score_str = f"[green]{score_str}[/green]"
            elif s.score >= 0.4:
                score_str = f"[yellow]{score_str}[/yellow]"
            else:
                score_str = f"[dim]{score_str}[/dim]"

            reasons_str = "; ".join(s.reasons) if s.reasons else "-"
            table.add_row(f"@{s.username}", score_str, reasons_str)

        console.print(table)
    else:
        console.print("[dim]No reviewer suggestions available.[/dim]")
