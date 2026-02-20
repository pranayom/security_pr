"""Labeling report formatting -- JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import LabelingReport


def labeling_report_to_json(report: LabelingReport) -> str:
    """Serialize labeling report to JSON."""
    return report.model_dump_json(indent=2)


def render_labeling_report(report: LabelingReport, console: Console | None = None) -> None:
    """Render a Rich-formatted labeling report to the console."""
    if console is None:
        console = Console()

    existing = ", ".join(report.existing_labels) if report.existing_labels else "(none)"
    header = (
        f"[bold]Label Classification Report[/bold]\n\n"
        f"Repo: {report.owner}/{report.repo}\n"
        f"Item: {report.item_type.upper()} #{report.item_number} — {report.item_title}\n"
        f"Existing labels: {existing}\n"
        f"Taxonomy: {report.taxonomy_size} labels ({report.taxonomy_source})  |  "
        f"Threshold: {report.threshold:.2f}\n"
        f"Suggestions: {len(report.suggestions)}"
    )
    console.print(Panel(header, title="Label Automation", border_style="magenta"))

    if report.suggestions:
        table = Table(title="Suggested Labels")
        table.add_column("Label", style="bold cyan")
        table.add_column("Confidence")
        table.add_column("Embedding Sim")
        table.add_column("Keyword Matches")
        table.add_column("Source", style="dim")

        for s in report.suggestions:
            conf_str = f"{s.confidence:.3f}"
            if s.confidence >= 0.6:
                conf_str = f"[green]{conf_str}[/green]"
            elif s.confidence >= 0.4:
                conf_str = f"[yellow]{conf_str}[/yellow]"
            else:
                conf_str = f"[dim]{conf_str}[/dim]"

            kw_str = ", ".join(s.keyword_matches) if s.keyword_matches else "-"

            table.add_row(
                s.label,
                conf_str,
                f"{s.embedding_similarity:.3f}",
                kw_str,
                s.source,
            )

        console.print(table)
    else:
        console.print("[dim]No labels above threshold.[/dim]")
