"""Tier 2: Suspicion heuristic rule engine for issue quality assessment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import (
    FlagSeverity,
    HeuristicsResult,
    IssueMetadata,
    SuspicionFlag,
    TierOutcome,
)

# Severity weight multipliers for score aggregation (same as PR heuristics)
_SEVERITY_WEIGHTS: dict[FlagSeverity, float] = {
    FlagSeverity.HIGH: 0.3,
    FlagSeverity.MEDIUM: 0.15,
    FlagSeverity.LOW: 0.05,
}

# Keywords that suggest an issue is a bug report
_BUG_KEYWORDS = {"bug", "error", "crash", "exception", "traceback", "fail", "broken", "issue"}

# Keywords that suggest a reproduction is included
_REPRO_KEYWORDS = {
    "reproduce", "repro", "steps to", "step 1", "expected", "actual",
    "stack trace", "traceback", "```",
}


def check_vague_description(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 1: Flag if issue body is too short to be actionable."""
    min_length = gatekeeper_settings.issue_min_body_length
    if len(issue.body.strip()) < min_length:
        return SuspicionFlag(
            rule_id="vague_description",
            severity=FlagSeverity.MEDIUM,
            title="Vague description",
            explanation=f"Issue body is {len(issue.body.strip())} chars (minimum: {min_length})",
            evidence=f"body_length={len(issue.body.strip())}",
        )
    return None


def check_new_account(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 2: Flag if contributor account is less than 90 days old."""
    if not issue.author.account_created_at:
        return None

    age = datetime.now(timezone.utc) - issue.author.account_created_at
    threshold_days = gatekeeper_settings.new_account_days

    if age < timedelta(days=threshold_days):
        return SuspicionFlag(
            rule_id="new_account",
            severity=FlagSeverity.MEDIUM,
            title="New account",
            explanation=f"Account created {age.days} days ago (threshold: {threshold_days} days)",
            evidence=f"Account created: {issue.author.account_created_at.isoformat()}",
        )
    return None


def check_first_contribution(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 3: Flag if this is the contributor's first contribution to the repo."""
    if issue.author.contributions_to_repo == 0:
        return SuspicionFlag(
            rule_id="first_contribution",
            severity=FlagSeverity.LOW,
            title="First contribution",
            explanation=f"User '{issue.author.login}' has no prior issues in this repo",
            evidence="contributions_to_repo=0",
        )
    return None


def check_missing_reproduction(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 4: Flag bug-like issues without reproduction steps."""
    title_lower = issue.title.lower()
    body_lower = issue.body.lower()
    labels_lower = [l.lower() for l in issue.labels]

    # Check if this looks like a bug report
    is_bug = any(kw in title_lower or kw in body_lower for kw in _BUG_KEYWORDS)
    is_bug = is_bug or any("bug" in l for l in labels_lower)

    if not is_bug:
        return None

    # Check if repro steps are present
    has_repro = any(kw in body_lower for kw in _REPRO_KEYWORDS)
    if has_repro:
        return None

    return SuspicionFlag(
        rule_id="missing_reproduction",
        severity=FlagSeverity.MEDIUM,
        title="Missing reproduction steps",
        explanation="Bug-like issue without reproduction steps or code snippets",
        evidence=f"title='{issue.title[:60]}'",
    )


def check_short_title(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 5a: Flag very short titles."""
    if len(issue.title.strip()) < 10:
        return SuspicionFlag(
            rule_id="short_title",
            severity=FlagSeverity.LOW,
            title="Short title",
            explanation=f"Issue title is only {len(issue.title.strip())} chars",
            evidence=f"title='{issue.title}'",
        )
    return None


def check_all_caps_title(issue: IssueMetadata) -> SuspicionFlag | None:
    """Rule 5b: Flag ALL CAPS titles (spam signal)."""
    # Only flag if title has enough letters to judge
    letters = [c for c in issue.title if c.isalpha()]
    if len(letters) >= 5 and issue.title == issue.title.upper():
        return SuspicionFlag(
            rule_id="all_caps_title",
            severity=FlagSeverity.LOW,
            title="ALL CAPS title",
            explanation="Issue title is in ALL CAPS, which may indicate spam or low quality",
            evidence=f"title='{issue.title[:60]}'",
        )
    return None


def check_temporal_clustering(
    issue: IssueMetadata,
    recent_issues: list[IssueMetadata] | None = None,
) -> SuspicionFlag | None:
    """Rule 6: Flag if 3+ new-account issues arrive within 24 hours."""
    if not recent_issues or not issue.created_at:
        return None

    threshold_days = gatekeeper_settings.new_account_days
    window = timedelta(hours=24)

    clustered = []
    for other in recent_issues:
        if other.number == issue.number:
            continue
        if not other.created_at or not other.author.account_created_at:
            continue

        account_age = datetime.now(timezone.utc) - other.author.account_created_at
        time_diff = abs(issue.created_at - other.created_at)

        if account_age < timedelta(days=threshold_days) and time_diff < window:
            clustered.append(other)

    min_cluster = 3 if len(recent_issues) < 50 else 5
    if len(clustered) >= min_cluster:
        return SuspicionFlag(
            rule_id="temporal_clustering",
            severity=FlagSeverity.HIGH,
            title="Temporal clustering of new-account issues",
            explanation=f"{len(clustered)} other new-account issues within 24h window",
            evidence=", ".join(f"Issue#{i.number} by {i.author.login}" for i in clustered[:5]),
        )
    return None


def run_issue_heuristics(
    issue: IssueMetadata,
    recent_issues: list[IssueMetadata] | None = None,
    threshold: float = 0.0,
) -> HeuristicsResult:
    """Run all issue heuristic rules and aggregate into a suspicion score.

    Args:
        issue: The issue to assess.
        recent_issues: Other recent issues (for temporal clustering).
        threshold: Suspicion threshold (0 = use config default).

    Returns:
        HeuristicsResult with outcome GATED if score >= threshold, PASS otherwise.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.issue_suspicion_threshold

    rules = [
        check_vague_description(issue),
        check_new_account(issue),
        check_first_contribution(issue),
        check_missing_reproduction(issue),
        check_short_title(issue),
        check_all_caps_title(issue),
        check_temporal_clustering(issue, recent_issues),
    ]

    flags = [f for f in rules if f is not None]

    # Weighted score aggregation
    score = sum(_SEVERITY_WEIGHTS.get(f.severity, 0.1) for f in flags)
    score = min(score, 1.0)  # Cap at 1.0

    outcome = TierOutcome.GATED if score >= threshold else TierOutcome.PASS

    return HeuristicsResult(
        outcome=outcome,
        suspicion_score=score,
        flags=flags,
    )
