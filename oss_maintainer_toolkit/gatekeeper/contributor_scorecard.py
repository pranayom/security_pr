"""Contributor profile formatting -- JSON output and Rich terminal rendering."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oss_maintainer_toolkit.gatekeeper.models import ContributorProfile


def contributor_profile_to_json(profile: ContributorProfile) -> str:
    """Serialize contributor profile to JSON."""
    return profile.model_dump_json(indent=2)


def render_contributor_profile(profile: ContributorProfile, console: Console | None = None) -> None:
    """Render a Rich-formatted contributor profile to the console."""
    if console is None:
        console = Console()

    first = profile.first_contribution.strftime("%Y-%m-%d") if profile.first_contribution else "-"
    last = profile.last_contribution.strftime("%Y-%m-%d") if profile.last_contribution else "-"
    areas = ", ".join(profile.areas_of_expertise) if profile.areas_of_expertise else "(none)"

    header = (
        f"[bold]Contributor Profile: {profile.username}[/bold]\n\n"
        f"Repo: {profile.owner}/{profile.repo}\n"
        f"PRs analyzed: {profile.prs_analyzed}  |  Reviews: {profile.review_count}\n"
        f"Active: {first} to {last}"
    )
    console.print(Panel(header, title="Contributor Profile", border_style="green"))

    # Metrics table
    table = Table(title="Contribution Metrics")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    merge_style = "[green]" if profile.merge_rate >= 0.7 else "[yellow]" if profile.merge_rate >= 0.4 else "[red]"
    test_style = "[green]" if profile.test_inclusion_rate >= 0.5 else "[yellow]" if profile.test_inclusion_rate >= 0.2 else "[red]"

    table.add_row("Total PRs", str(profile.total_prs))
    table.add_row("Merged", str(profile.merged_prs))
    table.add_row("Open", str(profile.open_prs))
    table.add_row("Closed (unmerged)", str(profile.closed_prs))
    table.add_row("Merge Rate", f"{merge_style}{profile.merge_rate:.1%}[/]")
    table.add_row("Test Inclusion Rate", f"{test_style}{profile.test_inclusion_rate:.1%}[/]")
    table.add_row("Avg Additions", f"{profile.avg_additions:.0f}")
    table.add_row("Avg Deletions", f"{profile.avg_deletions:.0f}")
    table.add_row("Areas of Expertise", areas)

    console.print(table)
