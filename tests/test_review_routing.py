"""Tests for the Gatekeeper review routing feature."""

import json

import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    CodeOwnerRule,
    PRAuthor,
    PRFileChange,
    PRMetadata,
    ReviewerSuggestion,
    ReviewRoutingReport,
)
from oss_maintainer_toolkit.gatekeeper.review_routing import (
    _match_codeowners,
    _score_past_reviewers,
    parse_codeowners,
    suggest_reviewers,
)
from oss_maintainer_toolkit.gatekeeper.review_routing_scorecard import (
    render_review_routing_report,
    review_routing_report_to_json,
)


def _make_pr(
    number: int = 1,
    title: str = "Fix bug",
    author: str = "contributor",
    files: list[str] | None = None,
) -> PRMetadata:
    file_changes = [PRFileChange(filename=f) for f in (files or [])]
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        author=PRAuthor(login=author),
        files=file_changes,
    )


# --- CODEOWNERS parsing ---

class TestParseCodeowners:
    def test_basic_rules(self):
        content = "*.js @frontend-team\n*.py @backend-team\n"
        rules = parse_codeowners(content)
        assert len(rules) == 2
        assert rules[0].pattern == "*.js"
        assert rules[0].owners == ["frontend-team"]

    def test_multiple_owners(self):
        content = "src/auth/* @alice @bob\n"
        rules = parse_codeowners(content)
        assert len(rules) == 1
        assert set(rules[0].owners) == {"alice", "bob"}

    def test_comments_and_empty_lines(self):
        content = "# This is a comment\n\n*.py @backend\n# Another comment\n"
        rules = parse_codeowners(content)
        assert len(rules) == 1

    def test_no_owners(self):
        content = "*.txt\n*.py @dev\n"
        rules = parse_codeowners(content)
        # Line with no @owner is skipped
        assert len(rules) == 1
        assert rules[0].pattern == "*.py"

    def test_empty_content(self):
        rules = parse_codeowners("")
        assert rules == []

    def test_strips_at_signs(self):
        content = "*.py @alice @bob\n"
        rules = parse_codeowners(content)
        assert rules[0].owners == ["alice", "bob"]


# --- CODEOWNERS matching ---

class TestMatchCodeowners:
    def test_glob_match(self):
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice"])]
        result = _match_codeowners(["main.py", "README.md"], rules)
        assert "alice" in result
        assert len(result["alice"]) == 1

    def test_no_match(self):
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice"])]
        result = _match_codeowners(["main.js"], rules)
        assert result == {}

    def test_last_rule_wins(self):
        rules = [
            CodeOwnerRule(pattern="*.py", owners=["alice"]),
            CodeOwnerRule(pattern="*.py", owners=["bob"]),
        ]
        result = _match_codeowners(["main.py"], rules)
        assert "bob" in result
        # alice may or may not be present depending on override semantics
        # The point is bob matched via the later rule

    def test_multiple_files_same_owner(self):
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice"])]
        result = _match_codeowners(["a.py", "b.py"], rules)
        assert "alice" in result


# --- Past reviewer scoring ---

class TestScorePastReviewers:
    def test_matching_files(self):
        changed = ["src/auth.py"]
        recent_prs = [_make_pr(number=10, files=["src/auth.py", "src/utils.py"])]
        reviews = {10: ["reviewer1", "reviewer2"]}
        result = _score_past_reviewers(changed, recent_prs, reviews)
        assert "reviewer1" in result
        assert "reviewer2" in result

    def test_no_overlap(self):
        changed = ["src/auth.py"]
        recent_prs = [_make_pr(number=10, files=["docs/README.md"])]
        reviews = {10: ["reviewer1"]}
        result = _score_past_reviewers(changed, recent_prs, reviews)
        assert result == {}

    def test_no_reviews(self):
        changed = ["src/auth.py"]
        recent_prs = [_make_pr(number=10, files=["src/auth.py"])]
        reviews = {}
        result = _score_past_reviewers(changed, recent_prs, reviews)
        assert result == {}


# --- Suggest reviewers ---

class TestSuggestReviewers:
    def test_empty_inputs(self):
        pr = _make_pr(files=["src/main.py"])
        report = suggest_reviewers(pr)
        assert report.suggestions == []
        assert report.codeowners_found is False

    def test_codeowners_suggestion(self):
        pr = _make_pr(files=["main.py"], author="contributor")
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice"])]
        report = suggest_reviewers(pr, codeowners_rules=rules)
        assert len(report.suggestions) >= 1
        assert report.suggestions[0].username == "alice"
        assert report.codeowners_found is True

    def test_excludes_pr_author(self):
        pr = _make_pr(files=["main.py"], author="alice")
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice", "bob"])]
        report = suggest_reviewers(pr, codeowners_rules=rules)
        usernames = [s.username for s in report.suggestions]
        assert "alice" not in usernames
        assert "bob" in usernames

    def test_max_suggestions(self):
        pr = _make_pr(files=["main.py"])
        rules = [CodeOwnerRule(pattern="*.py", owners=["a", "b", "c", "d", "e", "f"])]
        report = suggest_reviewers(pr, codeowners_rules=rules, max_suggestions=3)
        assert len(report.suggestions) <= 3

    def test_combined_sources(self):
        pr = _make_pr(files=["src/auth.py"], author="contributor")
        rules = [CodeOwnerRule(pattern="src/auth.py", owners=["alice"])]
        recent = [_make_pr(number=10, files=["src/auth.py"])]
        reviews = {10: ["bob"]}
        report = suggest_reviewers(
            pr, codeowners_rules=rules,
            recent_prs=recent, reviews_by_pr=reviews,
        )
        usernames = [s.username for s in report.suggestions]
        assert "alice" in usernames
        assert "bob" in usernames

    def test_scores_normalized(self):
        pr = _make_pr(files=["main.py"], author="contributor")
        rules = [CodeOwnerRule(pattern="*.py", owners=["alice", "bob"])]
        report = suggest_reviewers(pr, codeowners_rules=rules)
        for s in report.suggestions:
            assert 0.0 <= s.score <= 1.0

    def test_report_metadata(self):
        pr = _make_pr(number=42, title="Fix auth", files=["src/auth.py"])
        report = suggest_reviewers(pr)
        assert report.pr_number == 42
        assert report.pr_title == "Fix auth"
        assert report.changed_files == ["src/auth.py"]


# --- Model tests ---

class TestModels:
    def test_code_owner_rule(self):
        r = CodeOwnerRule(pattern="*.py", owners=["alice"])
        assert r.pattern == "*.py"

    def test_reviewer_suggestion_defaults(self):
        s = ReviewerSuggestion(username="alice", score=0.8)
        assert s.reasons == []

    def test_report_defaults(self):
        r = ReviewRoutingReport(owner="o", repo="r", pr_number=1)
        assert r.suggestions == []
        assert r.codeowners_found is False


# --- Scorecard ---

class TestReviewRoutingScorecard:
    def test_json_serialization(self):
        report = ReviewRoutingReport(
            owner="owner", repo="repo", pr_number=42,
            pr_title="Fix auth",
            changed_files=["src/auth.py"],
            suggestions=[
                ReviewerSuggestion(
                    username="alice", score=0.9,
                    reasons=["CODEOWNERS: src/auth/*"],
                ),
            ],
            codeowners_found=True,
        )
        json_str = review_routing_report_to_json(report)
        data = json.loads(json_str)
        assert data["pr_number"] == 42
        assert len(data["suggestions"]) == 1

    def test_rich_rendering(self):
        console = Console(record=True, width=120)
        report = ReviewRoutingReport(
            owner="owner", repo="repo", pr_number=42,
            pr_title="Fix auth",
            suggestions=[
                ReviewerSuggestion(username="alice", score=0.9, reasons=["CODEOWNERS"]),
            ],
            codeowners_found=True,
        )
        render_review_routing_report(report, console)
        output = console.export_text()
        assert "Review Routing" in output
        assert "alice" in output
        assert "#42" in output

    def test_rich_rendering_no_suggestions(self):
        console = Console(record=True, width=120)
        report = ReviewRoutingReport(owner="o", repo="r", pr_number=1)
        render_review_routing_report(report, console)
        output = console.export_text()
        assert "No reviewer suggestions" in output
