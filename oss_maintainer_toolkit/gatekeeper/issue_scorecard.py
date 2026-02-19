"""Issue scorecard formatting — JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import FlagSeverity, IssueScorecard, Verdict


_VERDICT_STYLES = {
    Verdict.FAST_TRACK: ("bold green", "FAST TRACK"),
    Verdict.REVIEW_REQUIRED: ("bold yellow", "REVIEW REQUIRED"),
    Verdict.RECOMMEND_CLOSE: ("bold red", "RECOMMEND CLOSE"),
}

_SEVERITY_STYLES = {
    FlagSeverity.HIGH: "red",
    FlagSeverity.MEDIUM: "yellow",
    FlagSeverity.LOW: "cyan",
}


def issue_scorecard_to_json(scorecard: IssueScorecard) -> str:
    """Serialize issue scorecard to JSON."""
    return scorecard.model_dump_json(indent=2)


def render_issue_scorecard(scorecard: IssueScorecard, console: Console | None = None) -> None:
    """Render a Rich-formatted issue scorecard to the console."""
    if console is None:
        console = Console()

    style, label = _VERDICT_STYLES.get(
        scorecard.verdict,
        ("bold", scorecard.verdict.value.upper()),
    )

    # Header panel
    header = (
        f"[{style}]{label}[/{style}]\n\n"
        f"Issue: {scorecard.owner}/{scorecard.repo}#{scorecard.issue_number}\n"
        f"Confidence: {scorecard.confidence:.0%}\n\n"
        f"{scorecard.summary}"
    )
    console.print(Panel(header, title="Issue Triage Scorecard", border_style=style))

    # Dimensions table
    if scorecard.dimensions:
        dim_table = Table(title="Dimensions")
        dim_table.add_column("Dimension", style="bold")
        dim_table.add_column("Score")
        dim_table.add_column("Summary")

        for dim in scorecard.dimensions:
            score_str = f"{dim.score:.2f}"
            if dim.score >= 0.7:
                score_str = f"[green]{score_str}[/green]"
            elif dim.score >= 0.4:
                score_str = f"[yellow]{score_str}[/yellow]"
            else:
                score_str = f"[red]{score_str}[/red]"

            dim_table.add_row(dim.dimension, score_str, dim.summary)

        console.print(dim_table)

    # Flags table
    if scorecard.flags:
        flag_table = Table(title="Flags")
        flag_table.add_column("Severity", style="bold")
        flag_table.add_column("Rule")
        flag_table.add_column("Title")
        flag_table.add_column("Explanation")

        for flag in scorecard.flags:
            sev_style = _SEVERITY_STYLES.get(flag.severity, "")
            flag_table.add_row(
                f"[{sev_style}]{flag.severity.value.upper()}[/{sev_style}]",
                flag.rule_id,
                flag.title,
                flag.explanation,
            )

        console.print(flag_table)
