"""Tests for the Gatekeeper PR ingestion module."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from mcp_ai_auditor.gatekeeper.cache import PRCache
from mcp_ai_auditor.gatekeeper.github_client import GitHubClient
from mcp_ai_auditor.gatekeeper.ingest import (
    _extract_linked_issues,
    _normalize_pr,
    _parse_datetime,
    ingest_pr,
    ingest_batch,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://api.github.com"


class TestHelpers:
    def test_extract_linked_issues_fixes(self):
        issues = _extract_linked_issues("Fixes #15 and closes #20")
        assert issues == [15, 20]

    def test_extract_linked_issues_hash_only(self):
        issues = _extract_linked_issues("Related to #7 and #12")
        assert issues == [7, 12]

    def test_extract_linked_issues_empty(self):
        assert _extract_linked_issues("") == []
        assert _extract_linked_issues("No issues here") == []

    def test_extract_linked_issues_resolves(self):
        issues = _extract_linked_issues("Resolves #3")
        assert issues == [3]

    def test_parse_datetime_iso(self):
        dt = _parse_datetime("2025-12-01T10:00:00Z")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 12

    def test_parse_datetime_none(self):
        assert _parse_datetime(None) is None
        assert _parse_datetime("") is None

    def test_normalize_pr(self):
        pr_data = json.loads((FIXTURES / "sample_pr_metadata.json").read_text())
        files_data = json.loads((FIXTURES / "sample_pr_files.json").read_text())
        user_data = json.loads((FIXTURES / "sample_user.json").read_text())

        pr = _normalize_pr(pr_data, files_data, "diff text here", user_data)

        assert pr.number == 42
        assert pr.title == "Add input validation to login endpoint"
        assert pr.author.login == "contributor123"
        assert pr.author.account_created_at is not None
        assert len(pr.files) == 2
        assert pr.diff_text == "diff text here"
        assert 15 in pr.linked_issues
        assert 20 in pr.linked_issues
        assert "enhancement" in pr.labels
        assert pr.total_additions == 55
        assert pr.total_deletions == 5


class TestIngestPR:
    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_pr_from_github(self):
        pr_data = json.loads((FIXTURES / "sample_pr_metadata.json").read_text())
        files_data = json.loads((FIXTURES / "sample_pr_files.json").read_text())
        user_data = json.loads((FIXTURES / "sample_user.json").read_text())

        # Same URL serves JSON or diff based on Accept header
        def pr_handler(request):
            if "diff" in request.headers.get("accept", ""):
                return httpx.Response(200, text="diff --git a/file.py b/file.py")
            return httpx.Response(200, json=pr_data)

        respx.get(f"{BASE_URL}/repos/nicoseng/OpenClaw/pulls/42").mock(
            side_effect=pr_handler
        )
        respx.get(f"{BASE_URL}/repos/nicoseng/OpenClaw/pulls/42/files").mock(
            return_value=httpx.Response(200, json=files_data)
        )
        respx.get(f"{BASE_URL}/users/contributor123").mock(
            return_value=httpx.Response(200, json=user_data)
        )
        respx.get(url__startswith=f"{BASE_URL}/search/issues").mock(
            return_value=httpx.Response(200, json={"total_count": 5})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            pr = await ingest_pr("nicoseng", "OpenClaw", 42, client)

        assert pr.number == 42
        assert pr.author.login == "contributor123"
        assert pr.author.contributions_to_repo == 5

    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_pr_with_cache_hit(self):
        cache = PRCache(db_path=":memory:")
        cached_data = {
            "owner": "nicoseng",
            "repo": "OpenClaw",
            "number": 42,
            "title": "Cached PR",
            "body": "",
            "author": {"login": "cached_user", "contributions_to_repo": 0},
            "files": [],
            "diff_text": "",
            "linked_issues": [],
            "labels": [],
            "total_additions": 0,
            "total_deletions": 0,
        }
        cache.put_pr("nicoseng", "OpenClaw", 42, cached_data)

        # No HTTP mocks needed — should hit cache
        async with GitHubClient(api_url=BASE_URL) as client:
            pr = await ingest_pr("nicoseng", "OpenClaw", 42, client, cache=cache)

        assert pr.title == "Cached PR"
        assert pr.author.login == "cached_user"
        cache.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_batch(self):
        for num in [1, 2]:
            pr_data = {
                "number": num,
                "title": f"PR #{num}",
                "body": "",
                "user": {"login": f"user{num}"},
                "labels": [],
                "base": {"repo": {"name": "repo", "owner": {"login": "owner"}}},
                "created_at": "2025-12-01T10:00:00Z",
            }

            def make_pr_handler(data):
                def handler(request):
                    if "diff" in request.headers.get("accept", ""):
                        return httpx.Response(200, text="")
                    return httpx.Response(200, json=data)
                return handler

            respx.get(f"{BASE_URL}/repos/owner/repo/pulls/{num}").mock(
                side_effect=make_pr_handler(pr_data)
            )
            respx.get(f"{BASE_URL}/repos/owner/repo/pulls/{num}/files").mock(
                return_value=httpx.Response(200, json=[])
            )
            respx.get(f"{BASE_URL}/users/user{num}").mock(
                return_value=httpx.Response(200, json={
                    "login": f"user{num}", "created_at": "2025-01-01T00:00:00Z"
                })
            )

        respx.get(url__startswith=f"{BASE_URL}/search/issues").mock(
            return_value=httpx.Response(200, json={"total_count": 0})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            prs = await ingest_batch("owner", "repo", [1, 2], client)

        assert len(prs) == 2
