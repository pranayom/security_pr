"""PR ingestion — fetch, normalize, and cache pull request data."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime

from src.gatekeeper.models import PRAuthor, PRFileChange, PRMetadata
from src.gatekeeper.github_client import GitHubClient
from src.gatekeeper.cache import PRCache


_ISSUE_PATTERN = re.compile(
    r"(?:fix(?:es)?|close[sd]?|resolve[sd]?)\s+#(\d+)|#(\d+)",
    re.IGNORECASE,
)


def _extract_linked_issues(body: str) -> list[int]:
    """Extract issue numbers from PR body text."""
    if not body:
        return []
    issues: set[int] = set()
    for match in _ISSUE_PATTERN.finditer(body):
        num = match.group(1) or match.group(2)
        if num:
            issues.add(int(num))
    return sorted(issues)


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse GitHub ISO datetime string."""
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _normalize_pr(
    pr_data: dict,
    files_data: list[dict],
    diff_text: str,
    user_data: dict | None = None,
) -> PRMetadata:
    """Transform raw GitHub API responses into PRMetadata."""
    user = pr_data.get("user", {})
    author = PRAuthor(
        login=user.get("login", "unknown"),
        account_created_at=_parse_datetime(user_data.get("created_at")) if user_data else None,
        contributions_to_repo=0,
    )

    files = [
        PRFileChange(
            filename=f.get("filename", ""),
            status=f.get("status", "modified"),
            additions=f.get("additions", 0),
            deletions=f.get("deletions", 0),
            patch=f.get("patch", ""),
        )
        for f in files_data
    ]

    body = pr_data.get("body", "") or ""
    total_additions = sum(f.additions for f in files)
    total_deletions = sum(f.deletions for f in files)

    labels = [label.get("name", "") for label in pr_data.get("labels", [])]

    return PRMetadata(
        owner=pr_data.get("base", {}).get("repo", {}).get("owner", {}).get("login", ""),
        repo=pr_data.get("base", {}).get("repo", {}).get("name", ""),
        number=pr_data.get("number", 0),
        title=pr_data.get("title", ""),
        body=body,
        author=author,
        files=files,
        diff_text=diff_text,
        created_at=_parse_datetime(pr_data.get("created_at")),
        updated_at=_parse_datetime(pr_data.get("updated_at")),
        linked_issues=_extract_linked_issues(body),
        labels=labels,
        total_additions=total_additions,
        total_deletions=total_deletions,
    )


async def ingest_pr(
    owner: str,
    repo: str,
    number: int,
    client: GitHubClient,
    cache: PRCache | None = None,
) -> PRMetadata:
    """Fetch, normalize, and optionally cache a pull request.

    Checks cache first; if miss, fetches from GitHub API.
    """
    if cache:
        cached = cache.get_pr(owner, repo, number)
        if cached:
            return PRMetadata(**cached)

    pr_data, files_data, diff_text = await asyncio.gather(
        client.get_pr(owner, repo, number),
        client.get_pr_files(owner, repo, number),
        client.get_pr_diff(owner, repo, number),
    )

    user_login = pr_data.get("user", {}).get("login", "")
    user_data = None
    if user_login:
        user_data = await client.get_user(user_login)

    pr_metadata = _normalize_pr(pr_data, files_data, diff_text, user_data)

    if cache:
        cache.put_pr(owner, repo, number, pr_metadata.model_dump(mode="json"))

    return pr_metadata


async def ingest_batch(
    owner: str,
    repo: str,
    pr_numbers: list[int],
    client: GitHubClient,
    cache: PRCache | None = None,
    concurrency: int = 5,
) -> list[PRMetadata]:
    """Concurrently ingest multiple PRs with a semaphore."""
    sem = asyncio.Semaphore(concurrency)

    async def _ingest_one(number: int) -> PRMetadata:
        async with sem:
            return await ingest_pr(owner, repo, number, client, cache)

    return await asyncio.gather(*[_ingest_one(n) for n in pr_numbers])
