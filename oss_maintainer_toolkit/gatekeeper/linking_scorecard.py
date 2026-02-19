"""Linking report formatting — JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import LinkingReport


def linking_report_to_json(report: LinkingReport) -> str:
    """Serialize linking report to JSON."""
    return report.model_dump_json(indent=2)


def render_linking_report(report: LinkingReport, console: Console | None = None) -> None:
    """Render a Rich-formatted linking report to the console."""
    if console is None:
        console = Console()

    # Header panel
    header = (
        f"[bold]Issue-to-PR Linking Report[/bold]\n\n"
        f"Repo: {report.owner}/{report.repo}\n"
        f"PRs analyzed: {report.total_prs}  |  Issues analyzed: {report.total_issues}\n"
        f"Threshold: {report.threshold:.2f}\n"
        f"Suggestions: {len(report.suggestions)}  |  "
        f"Explicit links: {len(report.explicit_links)}  |  "
        f"Orphan issues: {len(report.orphan_issues)}"
    )
    console.print(Panel(header, title="Issue-to-PR Linking", border_style="blue"))

    # Suggested links table
    if report.suggestions:
        link_table = Table(title="Suggested PR-Issue Links")
        link_table.add_column("PR #", style="bold cyan")
        link_table.add_column("Issue #", style="bold green")
        link_table.add_column("Similarity")
        link_table.add_column("PR Title")
        link_table.add_column("Issue Title")

        for s in report.suggestions:
            sim_str = f"{s.similarity:.3f}"
            if s.similarity >= 0.7:
                sim_str = f"[green]{sim_str}[/green]"
            elif s.similarity >= 0.5:
                sim_str = f"[yellow]{sim_str}[/yellow]"
            else:
                sim_str = f"[dim]{sim_str}[/dim]"

            link_table.add_row(
                f"#{s.pr_number}",
                f"#{s.issue_number}",
                sim_str,
                s.pr_title[:50],
                s.issue_title[:50],
            )

        console.print(link_table)

    # Orphan issues table
    if report.orphan_issues:
        orphan_table = Table(title="Orphan Issues (no linked PRs)")
        orphan_table.add_column("Issue #", style="bold yellow")

        for issue_num in report.orphan_issues:
            orphan_table.add_row(f"#{issue_num}")

        console.print(orphan_table)

    if not report.suggestions and not report.orphan_issues:
        console.print("[green]All issues have explicit PR links.[/green]")
