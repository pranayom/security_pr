"""Conflict detection report formatting -- JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import ConflictReport


def conflict_report_to_json(report: ConflictReport) -> str:
    """Serialize conflict report to JSON."""
    return report.model_dump_json(indent=2)


def render_conflict_report(report: ConflictReport, console: Console | None = None) -> None:
    """Render a Rich-formatted conflict detection report to the console."""
    if console is None:
        console = Console()

    header = (
        f"[bold]Cross-PR Conflict Detection Report[/bold]\n\n"
        f"Repo: {report.owner}/{report.repo}\n"
        f"Open PRs analyzed: {report.total_open_prs}\n"
        f"File overlap weight: {report.file_overlap_weight:.2f}  |  "
        f"Threshold: {report.threshold:.2f}\n"
        f"Conflict pairs: {len(report.conflict_pairs)}"
    )
    console.print(Panel(header, title="Conflict Detection", border_style="red"))

    if report.conflict_pairs:
        table = Table(title="Conflicting PR Pairs")
        table.add_column("PR A", style="bold cyan")
        table.add_column("PR B", style="bold cyan")
        table.add_column("Confidence")
        table.add_column("Shared Files")
        table.add_column("Semantic Sim")

        for pair in report.conflict_pairs:
            conf_str = f"{pair.confidence:.3f}"
            if pair.confidence >= 0.6:
                conf_str = f"[red]{conf_str}[/red]"
            elif pair.confidence >= 0.4:
                conf_str = f"[yellow]{conf_str}[/yellow]"
            else:
                conf_str = f"[dim]{conf_str}[/dim]"

            files_str = ", ".join(pair.overlapping_files[:3])
            if len(pair.overlapping_files) > 3:
                files_str += f" (+{len(pair.overlapping_files) - 3})"
            if not pair.overlapping_files:
                files_str = "-"

            table.add_row(
                f"#{pair.pr_a} {pair.pr_a_title[:30]}",
                f"#{pair.pr_b} {pair.pr_b_title[:30]}",
                conf_str,
                files_str,
                f"{pair.semantic_similarity:.3f}",
            )

        console.print(table)
    else:
        console.print("[green]No conflicting PR pairs detected.[/green]")
