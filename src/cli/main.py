"""Typer CLI for mcp-ai-auditor security tools."""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from src.scanners.vulnerability_scanner import scan_vulnerabilities
from src.analysis.data_flow import trace_data_flow
from src.cve.checker import check_cve

app = typer.Typer(
    name="auditor",
    help="AI-powered security auditor — scan for vulnerabilities, trace data flows, check CVEs.",
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


if __name__ == "__main__":
    app()
