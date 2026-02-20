"""Contributor Profiles: track contribution patterns per author.

Analyzes a contributor's PR history to compute metrics like merge rate,
test inclusion rate, areas of expertise, and review participation.
Tier 1 + Tier 2 only (metadata analysis), no LLM, $0 cost.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import ContributorProfile, PRMetadata


def _extract_top_directory(filename: str) -> str:
    """Extract top-level directory from a filename."""
    parts = filename.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def _has_test_files(pr: PRMetadata) -> bool:
    """Check if a PR includes test files."""
    for f in pr.files:
        name_lower = f.filename.lower()
        if "test" in name_lower or "spec" in name_lower:
            return True
    return False


def build_contributor_profile(
    owner: str,
    repo: str,
    username: str,
    prs: list[PRMetadata],
    review_count: int = 0,
) -> ContributorProfile:
    """Build a contributor profile from their PR history.

    Args:
        owner: Repository owner.
        repo: Repository name.
        username: Contributor's GitHub login.
        prs: List of PRMetadata objects for PRs authored by this user.
        review_count: Number of PRs this user has reviewed (pre-computed).

    Returns:
        ContributorProfile with computed metrics.
    """
    profile = ContributorProfile(
        owner=owner,
        repo=repo,
        username=username,
        prs_analyzed=len(prs),
        review_count=review_count,
    )

    if not prs:
        return profile

    merged = 0
    open_count = 0
    closed = 0
    test_prs = 0
    total_additions = 0
    total_deletions = 0
    dir_counter: Counter[str] = Counter()
    dates: list[datetime] = []

    for pr in prs:
        # Count by state
        if pr.merged_at:
            merged += 1
        elif pr.state == "open":
            open_count += 1
        else:
            closed += 1

        # Test inclusion
        if _has_test_files(pr):
            test_prs += 1

        # Size
        total_additions += pr.total_additions
        total_deletions += pr.total_deletions

        # Areas
        for f in pr.files:
            dir_counter[_extract_top_directory(f.filename)] += 1

        # Dates
        if pr.created_at:
            dates.append(pr.created_at)

    total = len(prs)
    profile.total_prs = total
    profile.merged_prs = merged
    profile.open_prs = open_count
    profile.closed_prs = closed
    profile.merge_rate = merged / total if total > 0 else 0.0
    profile.test_inclusion_rate = test_prs / total if total > 0 else 0.0
    profile.avg_additions = total_additions / total if total > 0 else 0.0
    profile.avg_deletions = total_deletions / total if total > 0 else 0.0

    # Top 5 directories
    profile.areas_of_expertise = [d for d, _ in dir_counter.most_common(5)]

    # Date range
    if dates:
        profile.first_contribution = min(dates)
        profile.last_contribution = max(dates)

    return profile
