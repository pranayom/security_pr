"""Tests for the MCP server tool functions."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from src.config import settings
from src.mcp.server import scan_vulnerabilities_tool, trace_data_flow_tool, check_cve_tool


FIXTURES = Path(__file__).parent / "fixtures"


class TestMCPTools:
    def test_scan_vulnerabilities_tool_returns_json(self):
        result = scan_vulnerabilities_tool(str(FIXTURES / "sample_vulnerable.py"))
        data = json.loads(result)
        assert data["files_scanned"] == 1
        assert data["total_findings"] > 0
        assert len(data["findings"]) == data["total_findings"]

    def test_trace_data_flow_tool_returns_json(self):
        result = trace_data_flow_tool(str(FIXTURES / "sample_taint.py"))
        data = json.loads(result)
        assert data["files_analyzed"] == 1
        assert data["total_flows"] > 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_check_cve_tool_returns_json(self):
        respx.post(f"{settings.osv_api_url}/querybatch").mock(
            return_value=httpx.Response(200, json={
                "results": [{"vulns": []} for _ in range(6)]
            })
        )

        result = await check_cve_tool(str(FIXTURES / "requirements_vulnerable.txt"))
        data = json.loads(result)
        assert data["dependencies_checked"] == 6

    def test_scan_tool_nonexistent(self):
        result = scan_vulnerabilities_tool("/nonexistent")
        data = json.loads(result)
        assert data["files_scanned"] == 0
        assert len(data["errors"]) > 0
