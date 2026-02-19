"""Tests for the CVE checker (OSV.dev integration, mocked)."""

from pathlib import Path

import httpx
import pytest
import respx

from oss_maintainer_toolkit.config import settings
from oss_maintainer_toolkit.cve.parsers import parse_requirements_txt, parse_package_json, find_and_parse_dependencies
from oss_maintainer_toolkit.cve.checker import check_cve, query_osv_batch
from oss_maintainer_toolkit.models import Dependency


FIXTURES = Path(__file__).parent / "fixtures"


# --- Parser Tests ---

class TestParsers:
    def test_parse_requirements_txt(self):
        deps = parse_requirements_txt(FIXTURES / "requirements_vulnerable.txt")
        assert len(deps) == 6
        names = {d.name for d in deps}
        assert "django" in names
        assert "requests" in names

        django = next(d for d in deps if d.name == "django")
        assert django.version == "2.2.1"

    def test_parse_package_json(self):
        deps = parse_package_json(FIXTURES / "package_vulnerable.json")
        assert len(deps) == 4  # 3 deps + 1 devDep
        names = {d.name for d in deps}
        assert "lodash" in names
        assert "express" in names
        assert "mocha" in names

    def test_find_and_parse_single_file(self):
        deps = find_and_parse_dependencies(str(FIXTURES / "requirements_vulnerable.txt"))
        assert len(deps) == 6

    def test_find_and_parse_directory(self):
        deps = find_and_parse_dependencies(str(FIXTURES))
        assert len(deps) >= 6  # at least requirements_vulnerable.txt

    def test_find_and_parse_nonexistent(self):
        deps = find_and_parse_dependencies("/nonexistent")
        assert len(deps) == 0


# --- CVE Checker Tests (mocked HTTP) ---

# Sample OSV batch response
MOCK_BATCH_RESPONSE = {
    "results": [
        {
            "vulns": [
                {"id": "GHSA-test-1234-abcd"}
            ]
        },
        {"vulns": []},  # no vulns for second package
    ]
}

MOCK_VULN_DETAIL = {
    "id": "GHSA-test-1234-abcd",
    "summary": "SQL Injection in Django",
    "details": "A SQL injection vulnerability exists in Django 2.2.1",
    "severity": [{"type": "CVSS_V3", "score": "9.8"}],
    "affected": [
        {
            "package": {"name": "django", "ecosystem": "PyPI"},
            "ranges": [
                {
                    "type": "ECOSYSTEM",
                    "events": [
                        {"introduced": "0"},
                        {"fixed": "2.2.2"}
                    ]
                }
            ]
        }
    ],
    "references": [
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2019-test", "type": "ADVISORY"}
    ]
}


class TestCVEChecker:
    @respx.mock
    @pytest.mark.asyncio
    async def test_query_osv_batch(self):
        respx.post(f"{settings.osv_api_url}/querybatch").mock(
            return_value=httpx.Response(200, json=MOCK_BATCH_RESPONSE)
        )
        respx.get(f"{settings.osv_api_url}/vulns/GHSA-test-1234-abcd").mock(
            return_value=httpx.Response(200, json=MOCK_VULN_DETAIL)
        )

        deps = [
            Dependency(name="django", version="2.2.1", source_file="requirements.txt"),
            Dependency(name="requests", version="2.19.1", source_file="requirements.txt"),
        ]
        records = await query_osv_batch(deps)
        assert len(records) == 1
        assert records[0].id == "GHSA-test-1234-abcd"
        assert records[0].affected_package == "django"
        assert records[0].fixed_version == "2.2.2"
        assert records[0].severity.value == "critical"

    @respx.mock
    @pytest.mark.asyncio
    async def test_check_cve_with_file(self):
        respx.post(f"{settings.osv_api_url}/querybatch").mock(
            return_value=httpx.Response(200, json={
                "results": [{"vulns": []} for _ in range(6)]
            })
        )

        result = await check_cve(str(FIXTURES / "requirements_vulnerable.txt"))
        assert result.dependencies_checked == 6
        assert result.total_vulnerabilities == 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_check_cve_api_error(self):
        respx.post(f"{settings.osv_api_url}/querybatch").mock(
            return_value=httpx.Response(500)
        )

        result = await check_cve(str(FIXTURES / "requirements_vulnerable.txt"))
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_check_cve_no_deps(self):
        result = await check_cve(str(FIXTURES / "sample_clean.py"))
        assert result.dependencies_checked == 0

    @pytest.mark.asyncio
    async def test_empty_dependencies(self):
        records = await query_osv_batch([])
        assert len(records) == 0
