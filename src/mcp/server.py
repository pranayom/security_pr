"""FastMCP server exposing security audit tools."""

import asyncio
import json

from mcp.server.fastmcp import FastMCP

from src.scanners.vulnerability_scanner import scan_vulnerabilities
from src.analysis.data_flow import trace_data_flow
from src.cve.checker import check_cve

mcp = FastMCP("mcp-ai-auditor")


@mcp.tool()
def scan_vulnerabilities_tool(target: str) -> str:
    """Scan files for security vulnerabilities using regex pattern matching.

    Detects OWASP Top 10 patterns: SQL injection, command injection,
    hardcoded secrets, XSS, path traversal, insecure deserialization,
    weak cryptography, SSL issues, and debug mode.

    Args:
        target: Path to a file or directory to scan.
    """
    result = scan_vulnerabilities(target)
    return result.model_dump_json(indent=2)


@mcp.tool()
def trace_data_flow_tool(target: str) -> str:
    """Trace tainted data flows from user inputs to dangerous sinks.

    AST-based Python taint analysis tracking data from Flask/Django
    request objects and input() through variable assignments to sinks
    like SQL execution, OS commands, eval, and deserialization.

    Args:
        target: Path to a Python file or directory to analyze.
    """
    result = trace_data_flow(target)
    return result.model_dump_json(indent=2)


@mcp.tool()
async def check_cve_tool(target: str) -> str:
    """Check project dependencies for known CVEs via OSV.dev.

    Parses requirements.txt and package.json files, queries the free
    OSV.dev batch API for known vulnerabilities, and returns detailed
    CVE records with severity, affected versions, and fix versions.

    Args:
        target: Path to a dependency file or directory containing them.
    """
    result = await check_cve(target)
    return result.model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run()
