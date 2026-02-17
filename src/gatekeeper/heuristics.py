"""Tier 2: Suspicion heuristic rule engine for PR risk assessment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.gatekeeper.config import gatekeeper_settings
from src.gatekeeper.models import (
    FlagSeverity,
    HeuristicsResult,
    PRMetadata,
    SuspicionFlag,
    TierOutcome,
)

# Severity weight multipliers for score aggregation
_SEVERITY_WEIGHTS: dict[FlagSeverity, float] = {
    FlagSeverity.HIGH: 0.3,
    FlagSeverity.MEDIUM: 0.15,
    FlagSeverity.LOW: 0.05,
}


def _is_sensitive_path(filename: str, sensitive_paths: list[str] | None = None) -> bool:
    """Check if a filename matches any sensitive path pattern."""
    paths = sensitive_paths or gatekeeper_settings.sensitive_paths
    filename_lower = filename.lower()
    return any(p.lower() in filename_lower for p in paths)


def check_new_account(pr: PRMetadata) -> SuspicionFlag | None:
    """Rule 1: Flag if contributor account is less than 90 days old."""
    if not pr.author.account_created_at:
        return None

    age = datetime.now(timezone.utc) - pr.author.account_created_at
    threshold_days = gatekeeper_settings.new_account_days

    if age < timedelta(days=threshold_days):
        return SuspicionFlag(
            rule_id="new_account",
            severity=FlagSeverity.MEDIUM,
            title="New account",
            explanation=f"Account created {age.days} days ago (threshold: {threshold_days} days)",
            evidence=f"Account created: {pr.author.account_created_at.isoformat()}",
        )
    return None


def check_first_contribution(pr: PRMetadata) -> SuspicionFlag | None:
    """Rule 2: Flag if this is the contributor's first contribution to the repo."""
    if pr.author.contributions_to_repo == 0:
        return SuspicionFlag(
            rule_id="first_contribution",
            severity=FlagSeverity.LOW,
            title="First contribution",
            explanation=f"User '{pr.author.login}' has no prior contributions to this repo",
            evidence=f"contributions_to_repo=0",
        )
    return None


def check_sensitive_paths(pr: PRMetadata, sensitive_paths: list[str] | None = None) -> SuspicionFlag | None:
    """Rule 3: Flag if PR touches security-sensitive paths."""
    sensitive_files = [f for f in pr.files if _is_sensitive_path(f.filename, sensitive_paths)]
    if not sensitive_files:
        return None

    filenames = [f.filename for f in sensitive_files]
    # Higher severity if touching auth/crypto directly
    has_high_risk = any(
        p in f.filename.lower()
        for f in sensitive_files
        for p in ("auth", "crypto", "security", "password", "login")
    )

    return SuspicionFlag(
        rule_id="sensitive_paths",
        severity=FlagSeverity.HIGH if has_high_risk else FlagSeverity.MEDIUM,
        title="Sensitive path changes",
        explanation=f"PR modifies {len(sensitive_files)} security-sensitive file(s)",
        evidence=", ".join(filenames[:5]),
    )


def check_test_ratio(pr: PRMetadata) -> SuspicionFlag | None:
    """Rule 4: Flag if code additions > 20 but test additions are below threshold."""
    code_additions = 0
    test_additions = 0

    for f in pr.files:
        if "test" in f.filename.lower() or "spec" in f.filename.lower():
            test_additions += f.additions
        else:
            code_additions += f.additions

    if code_additions <= 20:
        return None

    total = code_additions + test_additions
    ratio = test_additions / total if total > 0 else 0.0

    if ratio < gatekeeper_settings.min_test_ratio:
        return SuspicionFlag(
            rule_id="low_test_ratio",
            severity=FlagSeverity.MEDIUM,
            title="Low test coverage",
            explanation=f"Test ratio {ratio:.1%} is below threshold {gatekeeper_settings.min_test_ratio:.0%} "
                        f"({test_additions} test lines / {total} total additions)",
            evidence=f"code_additions={code_additions}, test_additions={test_additions}",
        )
    return None


def check_dependency_changes(pr: PRMetadata) -> SuspicionFlag | None:
    """Rule 5: Flag dependency file changes without body mentioning deps."""
    dep_files = [
        "requirements.txt", "package.json", "pyproject.toml",
        "Gemfile", "go.mod", "Cargo.toml", "pom.xml",
        "package-lock.json", "yarn.lock", "Pipfile",
    ]

    changed_dep_files = [
        f for f in pr.files
        if any(f.filename.endswith(d) or f.filename.endswith(d) for d in dep_files)
    ]

    if not changed_dep_files:
        return None

    body_lower = pr.body.lower()
    dep_keywords = ["depend", "upgrade", "bump", "update", "package", "library", "version"]
    mentions_deps = any(kw in body_lower for kw in dep_keywords)

    if not mentions_deps:
        return SuspicionFlag(
            rule_id="unjustified_deps",
            severity=FlagSeverity.HIGH,
            title="Unjustified dependency changes",
            explanation="Dependency files modified but PR description doesn't mention dependency changes",
            evidence=", ".join(f.filename for f in changed_dep_files),
        )
    return None


def check_large_diff_hiding(pr: PRMetadata, sensitive_paths: list[str] | None = None) -> SuspicionFlag | None:
    """Rule 6: Flag large diffs where <5% of changes are in sensitive paths."""
    total_changes = pr.total_additions + pr.total_deletions
    if total_changes < 500:
        return None

    sensitive_changes = sum(
        f.additions + f.deletions
        for f in pr.files
        if _is_sensitive_path(f.filename, sensitive_paths)
    )

    if sensitive_changes == 0:
        return None

    sensitive_ratio = sensitive_changes / total_changes
    if sensitive_ratio < 0.05:
        return SuspicionFlag(
            rule_id="large_diff_hiding",
            severity=FlagSeverity.HIGH,
            title="Large diff with hidden sensitive changes",
            explanation=f"Large diff ({total_changes} changes) with only {sensitive_ratio:.1%} in sensitive paths — "
                        "sensitive changes may be hidden in bulk",
            evidence=f"total_changes={total_changes}, sensitive_changes={sensitive_changes}",
        )
    return None


def check_temporal_clustering(
    pr: PRMetadata,
    recent_prs: list[PRMetadata] | None = None,
) -> SuspicionFlag | None:
    """Rule 7: Flag if 2+ new-account PRs arrive within 24 hours."""
    if not recent_prs or not pr.created_at:
        return None

    threshold_days = gatekeeper_settings.new_account_days
    window = timedelta(hours=24)

    # Only check PRs from new accounts within 24h of this PR
    clustered = []
    for other in recent_prs:
        if other.number == pr.number:
            continue
        if not other.created_at or not other.author.account_created_at:
            continue

        account_age = datetime.now(timezone.utc) - other.author.account_created_at
        time_diff = abs(pr.created_at - other.created_at)

        if account_age < timedelta(days=threshold_days) and time_diff < window:
            clustered.append(other)

    # Scale threshold with context size: need 3+ clustered for small sets, 5+ for large
    min_cluster = 3 if len(recent_prs) < 50 else 5
    if len(clustered) >= min_cluster:
        return SuspicionFlag(
            rule_id="temporal_clustering",
            severity=FlagSeverity.HIGH,
            title="Temporal clustering of new-account PRs",
            explanation=f"{len(clustered)} other new-account PRs within 24h window",
            evidence=", ".join(f"PR#{p.number} by {p.author.login}" for p in clustered[:5]),
        )
    return None


def run_heuristics(
    pr: PRMetadata,
    recent_prs: list[PRMetadata] | None = None,
    threshold: float = 0.0,
    extra_sensitive_paths: list[str] | None = None,
) -> HeuristicsResult:
    """Run all heuristic rules and aggregate into a suspicion score.

    Args:
        pr: The PR to assess.
        recent_prs: Other recent PRs (for temporal clustering).
        threshold: Suspicion threshold (0 = use config default).
        extra_sensitive_paths: Additional sensitive paths (e.g. from vision document focus_areas).

    Returns:
        HeuristicsResult with outcome GATED if score >= threshold, PASS otherwise.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.suspicion_threshold

    # Merge default + project-specific sensitive paths
    merged_sensitive = None
    if extra_sensitive_paths:
        merged_sensitive = gatekeeper_settings.sensitive_paths + extra_sensitive_paths

    rules = [
        check_new_account(pr),
        check_first_contribution(pr),
        check_sensitive_paths(pr, sensitive_paths=merged_sensitive),
        check_test_ratio(pr),
        check_dependency_changes(pr),
        check_large_diff_hiding(pr, sensitive_paths=merged_sensitive),
        check_temporal_clustering(pr, recent_prs),
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
