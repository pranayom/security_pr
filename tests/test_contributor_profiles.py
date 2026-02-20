"""Tests for the Gatekeeper contributor profiles feature."""

import json
from datetime import datetime, timezone

import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    ContributorProfile,
    PRAuthor,
    PRFileChange,
    PRMetadata,
)
from oss_maintainer_toolkit.gatekeeper.contributor_profiles import (
    _extract_top_directory,
    _has_test_files,
    build_contributor_profile,
)
from oss_maintainer_toolkit.gatekeeper.contributor_scorecard import (
    contributor_profile_to_json,
    render_contributor_profile,
)


def _make_pr(
    number: int = 1,
    title: str = "Fix bug",
    state: str = "closed",
    merged_at: datetime | None = None,
    files: list[str] | None = None,
    additions: int = 10,
    deletions: int = 5,
    created_at: datetime | None = None,
) -> PRMetadata:
    file_changes = [PRFileChange(filename=f) for f in (files or [])]
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        author=PRAuthor(login="contributor"),
        state=state,
        merged_at=merged_at,
        files=file_changes,
        total_additions=additions,
        total_deletions=deletions,
        created_at=created_at,
    )


# --- Helper tests ---

class TestHelpers:
    def test_extract_top_directory_nested(self):
        assert _extract_top_directory("src/auth/login.py") == "src"

    def test_extract_top_directory_root_file(self):
        assert _extract_top_directory("README.md") == "(root)"

    def test_extract_top_directory_backslash(self):
        assert _extract_top_directory("src\\auth\\login.py") == "src"

    def test_has_test_files_true(self):
        pr = _make_pr(files=["src/main.py", "tests/test_main.py"])
        assert _has_test_files(pr) is True

    def test_has_test_files_spec(self):
        pr = _make_pr(files=["src/main.ts", "src/main.spec.ts"])
        assert _has_test_files(pr) is True

    def test_has_test_files_false(self):
        pr = _make_pr(files=["src/main.py", "src/utils.py"])
        assert _has_test_files(pr) is False

    def test_has_test_files_empty(self):
        pr = _make_pr(files=[])
        assert _has_test_files(pr) is False


# --- Profile building ---

class TestBuildProfile:
    def test_empty_prs(self):
        profile = build_contributor_profile("owner", "repo", "user", [])
        assert profile.total_prs == 0
        assert profile.merge_rate == 0.0
        assert profile.prs_analyzed == 0

    def test_all_merged(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prs = [
            _make_pr(number=1, merged_at=dt, created_at=dt),
            _make_pr(number=2, merged_at=dt, created_at=dt),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert profile.total_prs == 2
        assert profile.merged_prs == 2
        assert profile.merge_rate == 1.0

    def test_mixed_states(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prs = [
            _make_pr(number=1, state="closed", merged_at=dt, created_at=dt),
            _make_pr(number=2, state="open", created_at=dt),
            _make_pr(number=3, state="closed", created_at=dt),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert profile.merged_prs == 1
        assert profile.open_prs == 1
        assert profile.closed_prs == 1
        assert abs(profile.merge_rate - 1.0 / 3.0) < 1e-6

    def test_test_inclusion_rate(self):
        prs = [
            _make_pr(number=1, files=["src/main.py", "tests/test_main.py"]),
            _make_pr(number=2, files=["src/utils.py"]),
            _make_pr(number=3, files=["src/auth.py", "tests/test_auth.py"]),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert abs(profile.test_inclusion_rate - 2.0 / 3.0) < 1e-6

    def test_avg_additions_deletions(self):
        prs = [
            _make_pr(number=1, additions=100, deletions=50),
            _make_pr(number=2, additions=200, deletions=100),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert profile.avg_additions == 150.0
        assert profile.avg_deletions == 75.0

    def test_areas_of_expertise(self):
        prs = [
            _make_pr(number=1, files=["src/auth/login.py", "src/auth/logout.py"]),
            _make_pr(number=2, files=["src/auth/tokens.py", "docs/api.md"]),
            _make_pr(number=3, files=["tests/test_auth.py"]),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert "src" in profile.areas_of_expertise
        assert len(profile.areas_of_expertise) <= 5

    def test_date_range(self):
        dt1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dt2 = datetime(2026, 6, 15, tzinfo=timezone.utc)
        prs = [
            _make_pr(number=1, created_at=dt1),
            _make_pr(number=2, created_at=dt2),
        ]
        profile = build_contributor_profile("o", "r", "u", prs)
        assert profile.first_contribution == dt1
        assert profile.last_contribution == dt2

    def test_review_count_passthrough(self):
        profile = build_contributor_profile("o", "r", "u", [], review_count=42)
        assert profile.review_count == 42

    def test_profile_metadata(self):
        profile = build_contributor_profile("myowner", "myrepo", "myuser", [])
        assert profile.owner == "myowner"
        assert profile.repo == "myrepo"
        assert profile.username == "myuser"


# --- Model tests ---

class TestContributorProfileModel:
    def test_defaults(self):
        p = ContributorProfile(owner="o", repo="r", username="u")
        assert p.total_prs == 0
        assert p.merge_rate == 0.0
        assert p.areas_of_expertise == []
        assert p.first_contribution is None


# --- Scorecard ---

class TestContributorScorecard:
    def test_json_serialization(self):
        profile = ContributorProfile(
            owner="owner", repo="repo", username="alice",
            total_prs=10, merged_prs=8, merge_rate=0.8,
            test_inclusion_rate=0.6,
            areas_of_expertise=["src", "tests"],
            prs_analyzed=10,
        )
        json_str = contributor_profile_to_json(profile)
        data = json.loads(json_str)
        assert data["username"] == "alice"
        assert data["total_prs"] == 10
        assert data["merge_rate"] == 0.8

    def test_rich_rendering(self):
        console = Console(record=True, width=120)
        profile = ContributorProfile(
            owner="owner", repo="repo", username="alice",
            total_prs=5, merged_prs=4, merge_rate=0.8,
            test_inclusion_rate=0.6,
            areas_of_expertise=["src"],
            prs_analyzed=5,
        )
        render_contributor_profile(profile, console)
        output = console.export_text()
        assert "Contributor Profile" in output
        assert "alice" in output
        assert "owner/repo" in output
        assert "Merge Rate" in output

    def test_rich_rendering_empty_profile(self):
        console = Console(record=True, width=120)
        profile = ContributorProfile(owner="o", repo="r", username="newbie")
        render_contributor_profile(profile, console)
        output = console.export_text()
        assert "newbie" in output
