"""Tests for the Gatekeeper heuristic rule engine (Tier 2)."""

from datetime import datetime, timedelta, timezone

import pytest

from oss_maintainer_toolkit.gatekeeper.heuristics import (
    check_dependency_changes,
    check_first_contribution,
    check_large_diff_hiding,
    check_new_account,
    check_sensitive_paths,
    check_temporal_clustering,
    check_test_ratio,
    run_heuristics,
)
from oss_maintainer_toolkit.gatekeeper.models import (
    FlagSeverity,
    PRAuthor,
    PRFileChange,
    PRMetadata,
    TierOutcome,
)


def _make_pr(
    number: int = 1,
    login: str = "user",
    account_age_days: int | None = None,
    contributions: int = 5,
    files: list[PRFileChange] | None = None,
    body: str = "",
    total_additions: int = 0,
    total_deletions: int = 0,
    created_at: datetime | None = None,
) -> PRMetadata:
    account_created = None
    if account_age_days is not None:
        account_created = datetime.now(timezone.utc) - timedelta(days=account_age_days)

    files = files or []
    if not total_additions and files:
        total_additions = sum(f.additions for f in files)
    if not total_deletions and files:
        total_deletions = sum(f.deletions for f in files)

    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title="Test PR",
        body=body,
        author=PRAuthor(
            login=login,
            account_created_at=account_created,
            contributions_to_repo=contributions,
        ),
        files=files,
        diff_text="",
        created_at=created_at,
        total_additions=total_additions,
        total_deletions=total_deletions,
    )


class TestCheckNewAccount:
    def test_new_account_flagged(self):
        pr = _make_pr(account_age_days=30)
        flag = check_new_account(pr)
        assert flag is not None
        assert flag.rule_id == "new_account"
        assert flag.severity == FlagSeverity.MEDIUM

    def test_old_account_passes(self):
        pr = _make_pr(account_age_days=365)
        assert check_new_account(pr) is None

    def test_no_account_date_passes(self):
        pr = _make_pr()  # No account_age_days
        assert check_new_account(pr) is None


class TestCheckFirstContribution:
    def test_first_contribution_flagged(self):
        pr = _make_pr(contributions=0)
        flag = check_first_contribution(pr)
        assert flag is not None
        assert flag.rule_id == "first_contribution"
        assert flag.severity == FlagSeverity.LOW

    def test_returning_contributor_passes(self):
        pr = _make_pr(contributions=10)
        assert check_first_contribution(pr) is None


class TestCheckSensitivePaths:
    def test_auth_path_flagged_high(self):
        files = [PRFileChange(filename="src/auth/login.py", additions=5)]
        pr = _make_pr(files=files)
        flag = check_sensitive_paths(pr)
        assert flag is not None
        assert flag.severity == FlagSeverity.HIGH

    def test_ci_path_flagged_medium(self):
        files = [PRFileChange(filename=".github/workflows/ci.yml", additions=3)]
        pr = _make_pr(files=files)
        flag = check_sensitive_paths(pr)
        assert flag is not None
        assert flag.severity in (FlagSeverity.HIGH, FlagSeverity.MEDIUM)

    def test_normal_path_passes(self):
        files = [PRFileChange(filename="src/utils/helpers.py", additions=10)]
        pr = _make_pr(files=files)
        assert check_sensitive_paths(pr) is None


class TestCheckTestRatio:
    def test_low_test_ratio_flagged(self):
        files = [
            PRFileChange(filename="src/main.py", additions=50),
            PRFileChange(filename="tests/test_main.py", additions=2),
        ]
        pr = _make_pr(files=files)
        flag = check_test_ratio(pr)
        assert flag is not None
        assert flag.rule_id == "low_test_ratio"

    def test_good_test_ratio_passes(self):
        files = [
            PRFileChange(filename="src/main.py", additions=50),
            PRFileChange(filename="tests/test_main.py", additions=30),
        ]
        pr = _make_pr(files=files)
        assert check_test_ratio(pr) is None

    def test_small_diff_passes(self):
        """Changes <= 20 lines don't trigger test ratio check."""
        files = [PRFileChange(filename="src/main.py", additions=10)]
        pr = _make_pr(files=files)
        assert check_test_ratio(pr) is None


class TestCheckDependencyChanges:
    def test_dep_change_without_mention_flagged(self):
        files = [PRFileChange(filename="requirements.txt", additions=2)]
        pr = _make_pr(files=files, body="Fixed a bug in the login")
        flag = check_dependency_changes(pr)
        assert flag is not None
        assert flag.rule_id == "unjustified_deps"
        assert flag.severity == FlagSeverity.HIGH

    def test_dep_change_with_mention_passes(self):
        files = [PRFileChange(filename="requirements.txt", additions=2)]
        pr = _make_pr(files=files, body="Upgraded dependency to fix CVE")
        assert check_dependency_changes(pr) is None

    def test_no_dep_files_passes(self):
        files = [PRFileChange(filename="src/main.py", additions=10)]
        pr = _make_pr(files=files)
        assert check_dependency_changes(pr) is None


class TestCheckLargeDiffHiding:
    def test_large_diff_with_hidden_sensitive_flagged(self):
        files = [
            PRFileChange(filename="src/main.py", additions=400, deletions=100),
            PRFileChange(filename="src/auth/config.py", additions=5, deletions=2),
        ]
        pr = _make_pr(files=files, total_additions=405, total_deletions=102)
        flag = check_large_diff_hiding(pr)
        assert flag is not None
        assert flag.rule_id == "large_diff_hiding"

    def test_small_diff_passes(self):
        files = [PRFileChange(filename="src/auth/login.py", additions=10, deletions=5)]
        pr = _make_pr(files=files, total_additions=10, total_deletions=5)
        assert check_large_diff_hiding(pr) is None

    def test_no_sensitive_paths_passes(self):
        files = [PRFileChange(filename="src/main.py", additions=400, deletions=200)]
        pr = _make_pr(files=files, total_additions=400, total_deletions=200)
        assert check_large_diff_hiding(pr) is None


class TestCheckTemporalClustering:
    def test_clustering_flagged(self):
        now = datetime.now(timezone.utc)
        pr = _make_pr(number=1, created_at=now, account_age_days=30)
        recent = [
            _make_pr(number=2, login="user2", created_at=now - timedelta(hours=2), account_age_days=20),
            _make_pr(number=3, login="user3", created_at=now - timedelta(hours=5), account_age_days=10),
            _make_pr(number=4, login="user4", created_at=now - timedelta(hours=8), account_age_days=15),
        ]
        flag = check_temporal_clustering(pr, recent)
        assert flag is not None
        assert flag.rule_id == "temporal_clustering"

    def test_no_clustering_passes(self):
        now = datetime.now(timezone.utc)
        pr = _make_pr(number=1, created_at=now)
        recent = [
            _make_pr(number=2, login="user2", created_at=now - timedelta(days=5), account_age_days=200),
        ]
        assert check_temporal_clustering(pr, recent) is None

    def test_no_recent_prs_passes(self):
        pr = _make_pr(created_at=datetime.now(timezone.utc))
        assert check_temporal_clustering(pr, []) is None
        assert check_temporal_clustering(pr, None) is None


class TestRunHeuristics:
    def test_clean_pr_passes(self):
        files = [
            PRFileChange(filename="src/utils.py", additions=15),
            PRFileChange(filename="tests/test_utils.py", additions=10),
        ]
        pr = _make_pr(
            account_age_days=365,
            contributions=10,
            files=files,
            body="Refactored utility helpers",
        )
        result = run_heuristics(pr, threshold=0.6)
        assert result.outcome == TierOutcome.PASS
        assert result.suspicion_score < 0.6

    def test_suspicious_pr_gated(self):
        files = [
            PRFileChange(filename="src/auth/login.py", additions=50),
            PRFileChange(filename="requirements.txt", additions=3),
        ]
        pr = _make_pr(
            account_age_days=10,
            contributions=0,
            files=files,
            body="Fixed some stuff",
        )
        result = run_heuristics(pr, threshold=0.6)
        assert result.outcome == TierOutcome.GATED
        assert result.suspicion_score >= 0.6
        assert len(result.flags) >= 3  # new_account + first_contribution + sensitive + unjustified_deps

    def test_score_capped_at_1(self):
        """Even with many flags, score should not exceed 1.0."""
        files = [
            PRFileChange(filename="src/auth/login.py", additions=400, deletions=100),
            PRFileChange(filename="src/crypto/keys.py", additions=5, deletions=2),
            PRFileChange(filename="requirements.txt", additions=3),
        ]
        pr = _make_pr(
            account_age_days=5,
            contributions=0,
            files=files,
            body="misc changes",
            total_additions=408,
            total_deletions=102,
        )
        result = run_heuristics(pr, threshold=0.1)
        assert result.suspicion_score <= 1.0
