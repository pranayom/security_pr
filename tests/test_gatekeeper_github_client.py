"""Tests for the Gatekeeper GitHub client (mocked HTTP)."""

import httpx
import pytest
import respx

from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient


BASE_URL = "https://api.github.com"


class TestGitHubClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_pr(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/pulls/42").mock(
            return_value=httpx.Response(200, json={"number": 42, "title": "Test PR"})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            pr = await client.get_pr("owner", "repo", 42)

        assert pr["number"] == 42
        assert pr["title"] == "Test PR"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_pr_files(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/pulls/42/files").mock(
            return_value=httpx.Response(200, json=[
                {"filename": "src/main.py", "additions": 10},
                {"filename": "tests/test_main.py", "additions": 20},
            ])
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            files = await client.get_pr_files("owner", "repo", 42)

        assert len(files) == 2
        assert files[0]["filename"] == "src/main.py"

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_pr_diff(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/pulls/42").mock(
            return_value=httpx.Response(200, text="diff --git a/file.py b/file.py\n+added line")
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            diff = await client.get_pr_diff("owner", "repo", 42)

        assert "diff --git" in diff
        assert "+added line" in diff

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_user(self):
        respx.get(f"{BASE_URL}/users/testuser").mock(
            return_value=httpx.Response(200, json={
                "login": "testuser",
                "created_at": "2025-01-01T00:00:00Z",
            })
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            user = await client.get_user("testuser")

        assert user["login"] == "testuser"

    @respx.mock
    @pytest.mark.asyncio
    async def test_pagination(self):
        responses = iter([
            httpx.Response(
                200,
                json=[{"number": 1}],
                headers={"link": f'<{BASE_URL}/repos/owner/repo/pulls?page=2>; rel="next"'},
            ),
            httpx.Response(200, json=[{"number": 2}]),
        ])
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            side_effect=lambda req: next(responses)
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            prs = await client.list_open_prs("owner", "repo")

        assert len(prs) == 2
        assert prs[0]["number"] == 1
        assert prs[1]["number"] == 2

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_check(self):
        respx.get(f"{BASE_URL}/rate_limit").mock(
            return_value=httpx.Response(200, json={
                "rate": {"limit": 60, "remaining": 55, "reset": 1234567890}
            })
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            result = await client.check_rate_limit()

        assert result["rate"]["remaining"] == 55

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_low_raises(self):
        """When remaining is below buffer, _check_remaining raises."""
        respx.get(f"{BASE_URL}/repos/owner/repo/pulls/42").mock(
            return_value=httpx.Response(
                200,
                json={"number": 42},
                headers={"x-ratelimit-remaining": "2"},
            )
        )

        async with GitHubClient(api_url=BASE_URL, rate_limit_buffer=10) as client:
            with pytest.raises(httpx.HTTPStatusError, match="Rate limit low"):
                await client.get_pr("owner", "repo", 42)

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_header_with_token(self):
        route = respx.get(f"{BASE_URL}/repos/owner/repo/pulls/42").mock(
            return_value=httpx.Response(200, json={"number": 42})
        )

        async with GitHubClient(token="ghp_test123", api_url=BASE_URL) as client:
            await client.get_pr("owner", "repo", 42)

        assert route.calls[0].request.headers["Authorization"] == "Bearer ghp_test123"

    @pytest.mark.asyncio
    async def test_client_not_entered_raises(self):
        client = GitHubClient(api_url=BASE_URL)
        with pytest.raises(RuntimeError, match="async context manager"):
            _ = client.client

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error_propagates(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/pulls/999").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get_pr("owner", "repo", 999)

    # --- Issue endpoint tests ---

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_issue(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/issues/101").mock(
            return_value=httpx.Response(200, json={"number": 101, "title": "Test Issue"})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            issue = await client.get_issue("owner", "repo", 101)

        assert issue["number"] == 101
        assert issue["title"] == "Test Issue"

    @respx.mock
    @pytest.mark.asyncio
    async def test_list_open_issues_excludes_prs(self):
        """list_open_issues should filter out items with pull_request key."""
        respx.get(f"{BASE_URL}/repos/owner/repo/issues").mock(
            return_value=httpx.Response(200, json=[
                {"number": 1, "title": "Real Issue"},
                {"number": 2, "title": "Actually a PR", "pull_request": {"url": "..."}},
                {"number": 3, "title": "Another Issue"},
            ])
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            issues = await client.list_open_issues("owner", "repo")

        assert len(issues) == 2
        assert issues[0]["number"] == 1
        assert issues[1]["number"] == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_count_user_issues(self):
        respx.get(url__startswith=f"{BASE_URL}/search/issues").mock(
            return_value=httpx.Response(200, json={"total_count": 7})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            count = await client.count_user_issues("owner", "repo", "testuser")

        assert count == 7

    @respx.mock
    @pytest.mark.asyncio
    async def test_count_user_issues_rate_limit_fallback(self):
        """Returns 0 on rate limit errors."""
        respx.get(url__startswith=f"{BASE_URL}/search/issues").mock(
            return_value=httpx.Response(429, json={"message": "rate limited"})
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            count = await client.count_user_issues("owner", "repo", "testuser")

        assert count == 0
