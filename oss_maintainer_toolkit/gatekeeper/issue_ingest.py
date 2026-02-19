"""Issue ingestion — fetch, normalize, and cache GitHub issue data."""

from __future__ import annotations

import asyncio
from datetime import datetime

from oss_maintainer_toolkit.gatekeeper.models import IssueAuthor, IssueMetadata
from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
from oss_maintainer_toolkit.gatekeeper.issue_cache import IssueCache


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse GitHub ISO datetime string."""
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _normalize_issue(
    issue_data: dict,
    user_data: dict | None = None,
    contributions_to_repo: int = 0,
) -> IssueMetadata:
    """Transform raw GitHub API responses into IssueMetadata."""
    user = issue_data.get("user", {})
    author = IssueAuthor(
        login=user.get("login", "unknown"),
        account_created_at=_parse_datetime(user_data.get("created_at")) if user_data else None,
        contributions_to_repo=contributions_to_repo,
    )

    labels = [label.get("name", "") for label in issue_data.get("labels", [])]
    assignees = [a.get("login", "") for a in issue_data.get("assignees", [])]
    milestone = ""
    if issue_data.get("milestone"):
        milestone = issue_data["milestone"].get("title", "")

    reactions = {}
    if issue_data.get("reactions"):
        r = issue_data["reactions"]
        for key in ("+1", "-1", "laugh", "hooray", "confused", "heart", "rocket", "eyes"):
            if r.get(key, 0) > 0:
                reactions[key] = r[key]

    # Derive owner/repo from repository_url or let caller pass explicitly
    repo_url = issue_data.get("repository_url", "")
    owner = ""
    repo = ""
    if repo_url:
        parts = repo_url.rstrip("/").split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1]

    return IssueMetadata(
        owner=owner,
        repo=repo,
        number=issue_data.get("number", 0),
        title=issue_data.get("title", ""),
        body=issue_data.get("body", "") or "",
        author=author,
        state=issue_data.get("state", "open"),
        labels=labels,
        assignees=assignees,
        milestone=milestone,
        reactions=reactions,
        comment_count=issue_data.get("comments", 0),
        created_at=_parse_datetime(issue_data.get("created_at")),
        updated_at=_parse_datetime(issue_data.get("updated_at")),
        closed_at=_parse_datetime(issue_data.get("closed_at")),
    )


async def ingest_issue(
    owner: str,
    repo: str,
    number: int,
    client: GitHubClient,
    cache: IssueCache | None = None,
) -> IssueMetadata:
    """Fetch, normalize, and optionally cache a GitHub issue.

    Checks cache first; if miss, fetches from GitHub API.
    """
    if cache:
        cached = cache.get_issue(owner, repo, number)
        if cached:
            return IssueMetadata(**cached)

    issue_data = await client.get_issue(owner, repo, number)

    user_login = issue_data.get("user", {}).get("login", "")
    user_data = None
    contributions = 0
    if user_login:
        user_data, contributions = await asyncio.gather(
            client.get_user(user_login),
            client.count_user_issues(owner, repo, user_login),
        )

    issue = _normalize_issue(issue_data, user_data, contributions)
    # Override owner/repo with the known values (API may not include repository_url)
    issue = issue.model_copy(update={"owner": owner, "repo": repo})

    if cache:
        cache.put_issue(owner, repo, number, issue.model_dump(mode="json"))

    return issue


async def ingest_issue_batch(
    owner: str,
    repo: str,
    issue_numbers: list[int],
    client: GitHubClient,
    cache: IssueCache | None = None,
    concurrency: int = 5,
) -> list[IssueMetadata]:
    """Concurrently ingest multiple issues with a semaphore."""
    sem = asyncio.Semaphore(concurrency)

    async def _ingest_one(number: int) -> IssueMetadata:
        async with sem:
            return await ingest_issue(owner, repo, number, client, cache)

    return await asyncio.gather(*[_ingest_one(n) for n in issue_numbers])
