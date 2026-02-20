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

    async def list_recently_merged_prs(
        self, owner: str, repo: str, since_days: int = 90,
    ) -> list[dict]:
        """List recently merged pull requests (paginated).

        Fetches closed PRs and filters to those with merged_at set,
        limited to the last `since_days` days.
        """
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        items = await self._paginate(
            f"/repos/{owner}/{repo}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": "100",
            },
        )
        merged = []
        for item in items:
            merged_at = item.get("merged_at")
            if not merged_at:
                continue
            # Parse ISO datetime
            dt = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            if dt >= cutoff:
                merged.append(item)
        return merged

    async def list_closed_unmerged_prs(
        self, owner: str, repo: str, max_results: int = 10,
    ) -> list[dict]:
        """List recently closed-but-not-merged pull requests.

        These represent rejected or abandoned PRs — useful for inferring
        what the project does NOT accept.
        """
        items = await self._paginate(
            f"/repos/{owner}/{repo}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": "100",
            },
        )
        rejected = []
        for item in items:
            if item.get("merged_at") is None:
                rejected.append(item)
                if len(rejected) >= max_results:
                    break
        return rejected

    async def list_repo_labels(self, owner: str, repo: str) -> list[dict]:
        """List all labels for a repository (paginated)."""
        return await self._paginate(
            f"/repos/{owner}/{repo}/labels",
            params={"per_page": "100"},
        )

    async def search_user_prs(
        self, owner: str, repo: str, username: str, max_results: int = 50,
    ) -> list[dict]:
        """Search for PRs by a user in a repo via Search API.

        Returns raw search result items (limited to max_results).
        Returns empty list on rate limit or errors.
        """
        query = f"repo:{owner}/{repo} author:{username} type:pr"
        try:
            results: list[dict] = []
            page = 1
            while len(results) < max_results:
                resp = await self.client.get(
                    "/search/issues",
                    params={"q": query, "per_page": "100", "page": str(page)},
                )
                if resp.status_code in (403, 422, 429):
                    break
                resp.raise_for_status()
                items = resp.json().get("items", [])
                if not items:
                    break
                results.extend(items)
                page += 1
            return results[:max_results]
        except (httpx.HTTPStatusError, httpx.TimeoutException):
            return []

    async def list_pr_reviews(self, owner: str, repo: str, number: int) -> list[dict]:
        """List reviews on a pull request (paginated)."""
        return await self._paginate(
            f"/repos/{owner}/{repo}/pulls/{number}/reviews",
            params={"per_page": "100"},
        )

    async def get_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Fetch raw file content from the default branch.

        Returns None if file doesn't exist (404).
        """
        resp = await self.client.get(
            f"/repos/{owner}/{repo}/contents/{path}",
            headers={"Accept": "application/vnd.github.raw+json"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

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
