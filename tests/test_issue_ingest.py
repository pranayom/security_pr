"""Tests for the Gatekeeper issue ingestion module."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
from oss_maintainer_toolkit.gatekeeper.issue_cache import IssueCache
from oss_maintainer_toolkit.gatekeeper.issue_ingest import (
    _normalize_issue,
    _parse_datetime,
    ingest_issue,
    ingest_issue_batch,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://api.github.com"


class TestHelpers:
    def test_parse_datetime_iso(self):
        dt = _parse_datetime("2025-12-01T10:00:00Z")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 12

    def test_parse_datetime_none(self):
        assert _parse_datetime(None) is None
        assert _parse_datetime("") is None

    def test_normalize_issue(self):
        issue_data = json.loads((FIXTURES / "sample_issue_metadata.json").read_text())
        user_data = json.loads((FIXTURES / "sample_user.json").read_text())

        issue = _normalize_issue(issue_data, user_data, contributions_to_repo=3)

        assert issue.number == 101
        assert issue.title == "Login page crashes on empty password"
        assert issue.author.login == "contributor123"
        assert issue.author.account_created_at is not None
        assert issue.author.contributions_to_repo == 3
        assert issue.state == "open"
        assert "bug" in issue.labels
        assert "login" in issue.labels
        assert "maintainer1" in issue.assignees
        assert issue.milestone == "v2.0"
        assert issue.reactions["+1"] == 5
        assert issue.comment_count == 3
        assert issue.owner == "nicoseng"
        assert issue.repo == "OpenClaw"

    def test_normalize_issue_minimal(self):
        """Issue with minimal data normalizes without errors."""
        issue_data = {
            "number": 1,
            "title": "Minimal",
            "user": {"login": "testuser"},
            "state": "open",
            "labels": [],
            "assignees": [],
            "comments": 0,
        }
        issue = _normalize_issue(issue_data)
        assert issue.number == 1
        assert issue.author.login == "testuser"
        assert issue.body == ""
        assert issue.labels == []


class TestIngestIssue:
    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_issue_from_github(self):
        issue_data = json.loads((FIXTURES / "sample_issue_metadata.json").read_text())
        user_data = json.loads((FIXTURES / "sample_user.json").read_text())

        respx.get(f"{BASE_URL}/repos/nicoseng/OpenClaw/issues/101").mock(
            return_value=httpx.Response(200, json=issue_data)
        )
        respx.get(f"{BASE_URL}/users/contributor123").mock(
            return_value=httpx.Response(200, json=user_data)
        )
        respx.get(url__startswith=f"{BASE_URL}/search/issues").mock(
            return_value=httpx.Response(200, json={"total_count": 3})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            issue = await ingest_issue("nicoseng", "OpenClaw", 101, client)

        assert issue.number == 101
        assert issue.author.login == "contributor123"
        assert issue.author.contributions_to_repo == 3
        assert issue.owner == "nicoseng"
        assert issue.repo == "OpenClaw"

    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_issue_with_cache_hit(self):
        cache = IssueCache(db_path=":memory:")
        cached_data = {
            "owner": "nicoseng",
            "repo": "OpenClaw",
            "number": 101,
            "title": "Cached Issue",
            "body": "",
            "author": {"login": "cached_user", "contributions_to_repo": 0},
            "state": "open",
            "labels": [],
            "assignees": [],
            "milestone": "",
            "reactions": {},
            "comment_count": 0,
        }
        cache.put_issue("nicoseng", "OpenClaw", 101, cached_data)

        async with GitHubClient(api_url=BASE_URL) as client:
            issue = await ingest_issue("nicoseng", "OpenClaw", 101, client, cache=cache)

        assert issue.title == "Cached Issue"
        assert issue.author.login == "cached_user"
        cache.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_ingest_issue_batch(self):
        for num in [1, 2]:
            issue_data = {
                "number": num,
                "title": f"Issue #{num}",
                "body": "",
                "user": {"login": f"user{num}"},
                "state": "open",
                "labels": [],
                "assignees": [],
                "comments": 0,
                "created_at": "2025-12-01T10:00:00Z",
            }

            respx.get(f"{BASE_URL}/repos/owner/repo/issues/{num}").mock(
                return_value=httpx.Response(200, json=issue_data)
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
            issues = await ingest_issue_batch("owner", "repo", [1, 2], client)

        assert len(issues) == 2
