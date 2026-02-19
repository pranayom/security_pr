"""Async GitHub REST API client for PR ingestion."""

from __future__ import annotations

import httpx

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings


class GitHubClient:
    """Async context manager wrapping httpx.AsyncClient for GitHub API."""

    def __init__(
        self,
        token: str = "",
        api_url: str = "",
        rate_limit_buffer: int = 0,
    ):
        self.token = token or gatekeeper_settings.github_token
        self.api_url = (api_url or gatekeeper_settings.github_api_url).rstrip("/")
        self.rate_limit_buffer = rate_limit_buffer or gatekeeper_settings.rate_limit_buffer
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GitHubClient:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers=headers,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("GitHubClient must be used as async context manager")
        return self._client

    async def check_rate_limit(self) -> dict:
        """Check current rate limit status."""
        resp = await self.client.get("/rate_limit")
        resp.raise_for_status()
        return resp.json()

    async def _check_remaining(self, resp: httpx.Response) -> None:
        """Raise if rate limit remaining is below buffer."""
        remaining = int(resp.headers.get("x-ratelimit-remaining", "999"))
        if remaining < self.rate_limit_buffer:
            raise httpx.HTTPStatusError(
                f"Rate limit low: {remaining} remaining (buffer={self.rate_limit_buffer})",
                request=resp.request,
                response=resp,
            )

    async def _paginate(self, url: str, params: dict | None = None) -> list[dict]:
        """Follow Link header pagination to collect all pages."""
        results: list[dict] = []
        next_url: str | None = url
        current_params = params

        while next_url:
            resp = await self.client.get(next_url, params=current_params)
            resp.raise_for_status()
            await self._check_remaining(resp)

            data = resp.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)

            # Parse Link header for next page
            link_header = resp.headers.get("link", "")
            next_url = None
            current_params = None  # params are embedded in the Link URL
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip().strip("<>")
                    break

        return results

    async def get_pr(self, owner: str, repo: str, number: int) -> dict:
        """Fetch a single pull request."""
        resp = await self.client.get(f"/repos/{owner}/{repo}/pulls/{number}")
        resp.raise_for_status()
        await self._check_remaining(resp)
        return resp.json()

    async def get_pr_files(self, owner: str, repo: str, number: int) -> list[dict]:
        """Fetch files changed in a pull request (paginated)."""
        return await self._paginate(
            f"/repos/{owner}/{repo}/pulls/{number}/files",
            params={"per_page": "100"},
        )

    async def get_pr_diff(self, owner: str, repo: str, number: int) -> str:
        """Fetch the raw diff for a pull request.

        Returns empty string if diff is unavailable (e.g. draft PRs, merge conflicts).
        """
        resp = await self.client.get(
            f"/repos/{owner}/{repo}/pulls/{number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        if resp.status_code == 406:
            return ""
        resp.raise_for_status()
        return resp.text

    async def get_user(self, username: str) -> dict:
        """Fetch user profile details."""
        resp = await self.client.get(f"/users/{username}")
        resp.raise_for_status()
        return resp.json()

    async def list_open_prs(self, owner: str, repo: str) -> list[dict]:
        """List open pull requests (paginated)."""
        return await self._paginate(
            f"/repos/{owner}/{repo}/pulls",
            params={"state": "open", "per_page": "100"},
        )

    async def count_user_prs(self, owner: str, repo: str, username: str) -> int:
        """Count merged PRs by a user in a repo via Search API.

        Returns 0 on rate limit or other errors to avoid blocking ingestion.
        """
        query = f"repo:{owner}/{repo} author:{username} type:pr is:merged"
        try:
            resp = await self.client.get(
                "/search/issues",
                params={"q": query, "per_page": "1"},
            )
            if resp.status_code in (403, 422, 429):
                return 0
            resp.raise_for_status()
            return resp.json().get("total_count", 0)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            return 0

    # --- Issue endpoints ---

    async def get_issue(self, owner: str, repo: str, number: int) -> dict:
        """Fetch a single issue."""
        resp = await self.client.get(f"/repos/{owner}/{repo}/issues/{number}")
        resp.raise_for_status()
        await self._check_remaining(resp)
        return resp.json()

    async def list_open_issues(self, owner: str, repo: str) -> list[dict]:
        """List open issues (paginated), excluding pull requests."""
        items = await self._paginate(
            f"/repos/{owner}/{repo}/issues",
            params={"state": "open", "per_page": "100"},
        )
        # GitHub's issues endpoint returns PRs too — filter them out
        return [item for item in items if "pull_request" not in item]

    async def count_user_issues(self, owner: str, repo: str, username: str) -> int:
        """Count issues authored by a user in a repo via Search API.

        Returns 0 on rate limit or other errors to avoid blocking ingestion.
        """
        query = f"repo:{owner}/{repo} author:{username} type:issue"
        try:
            resp = await self.client.get(
                "/search/issues",
                params={"q": query, "per_page": "1"},
            )
            if resp.status_code in (403, 422, 429):
                return 0
            resp.raise_for_status()
            return resp.json().get("total_count", 0)
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            return 0
