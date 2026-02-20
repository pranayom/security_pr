"""Tests for the Gatekeeper label automation feature."""

import json
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    IssueAuthor,
    IssueMetadata,
    LabelDefinition,
    LabelingReport,
    LabelSuggestion,
    PRAuthor,
    PRFileChange,
    PRMetadata,
    VisionDocument,
)
from oss_maintainer_toolkit.gatekeeper.labeling import (
    _build_item_embedding_text,
    _build_label_embedding_text,
    _compute_keyword_scores,
    classify_item,
    github_labels_to_taxonomy,
    merge_taxonomies,
)
from oss_maintainer_toolkit.gatekeeper.labeling_scorecard import (
    labeling_report_to_json,
    render_labeling_report,
)
from oss_maintainer_toolkit.gatekeeper.vision import load_vision_document

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pr(
    number: int = 1,
    title: str = "Fix bug",
    body: str = "",
    labels: list[str] | None = None,
    files: list[str] | None = None,
) -> PRMetadata:
    file_changes = [PRFileChange(filename=f) for f in (files or [])]
    return PRMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=PRAuthor(login="dev"),
        labels=labels or [],
        files=file_changes,
    )


def _make_issue(
    number: int = 1,
    title: str = "Bug report",
    body: str = "",
    labels: list[str] | None = None,
) -> IssueMetadata:
    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(login="user"),
        labels=labels or [],
    )


def _make_label(
    name: str = "bug",
    description: str = "Something is broken",
    keywords: list[str] | None = None,
    source: str = "vision",
) -> LabelDefinition:
    return LabelDefinition(
        name=name,
        description=description,
        keywords=keywords or [],
        source=source,
    )


# --- Model tests ---

class TestLabelModels:
    def test_label_definition_defaults(self):
        lb = LabelDefinition(name="bug")
        assert lb.name == "bug"
        assert lb.description == ""
        assert lb.keywords == []
        assert lb.color == ""
        assert lb.source == "vision"

    def test_label_suggestion_fields(self):
        s = LabelSuggestion(
            label="bug",
            confidence=0.75,
            embedding_similarity=0.6,
            keyword_matches=["error", "crash"],
            source="vision",
        )
        assert s.label == "bug"
        assert s.confidence == 0.75
        assert s.keyword_matches == ["error", "crash"]

    def test_empty_labeling_report(self):
        r = LabelingReport(
            owner="o", repo="r", item_type="pr", item_number=1,
        )
        assert r.suggestions == []
        assert r.existing_labels == []
        assert r.taxonomy_size == 0


# --- Text building ---

class TestTextBuilding:
    def test_label_text_includes_all_parts(self):
        lb = _make_label(
            name="security",
            description="Security-related changes",
            keywords=["auth", "crypto"],
        )
        text = _build_label_embedding_text(lb)
        assert "security" in text
        assert "Security-related" in text
        assert "auth" in text
        assert "crypto" in text

    def test_label_text_no_description(self):
        lb = LabelDefinition(name="docs")
        text = _build_label_embedding_text(lb)
        assert text == "docs"

    def test_item_text_pr_includes_filenames(self):
        pr = _make_pr(
            title="Fix auth",
            body="Fixes login issue",
            files=["src/auth.py", "tests/test_auth.py"],
        )
        text = _build_item_embedding_text(pr)
        assert "Fix auth" in text
        assert "Fixes login" in text
        assert "src/auth.py" in text
        assert "tests/test_auth.py" in text

    def test_item_text_issue_no_files(self):
        issue = _make_issue(title="Login broken", body="Cannot log in")
        text = _build_item_embedding_text(issue)
        assert "Login broken" in text
        assert "Cannot log in" in text


# --- Keyword scoring ---

class TestKeywordScoring:
    def test_exact_match(self):
        labels = [_make_label(keywords=["bug", "error"])]
        scores = _compute_keyword_scores("Found a bug in the code", labels)
        score, matched = scores[0]
        assert "bug" in matched
        assert score > 0

    def test_no_partial_match(self):
        """'debug' should NOT match keyword 'bug' due to word boundary."""
        labels = [_make_label(keywords=["bug"])]
        scores = _compute_keyword_scores("debug mode enabled", labels)
        score, matched = scores[0]
        assert score == 0.0
        assert matched == []

    def test_case_insensitive(self):
        labels = [_make_label(keywords=["Bug"])]
        scores = _compute_keyword_scores("found a bug", labels)
        score, matched = scores[0]
        assert "Bug" in matched
        assert score == 1.0

    def test_score_fraction(self):
        labels = [_make_label(keywords=["bug", "error", "crash"])]
        scores = _compute_keyword_scores("a bug and a crash happened", labels)
        score, matched = scores[0]
        assert len(matched) == 2
        assert abs(score - 2.0 / 3.0) < 1e-6

    def test_empty_keywords(self):
        labels = [_make_label(keywords=[])]
        scores = _compute_keyword_scores("some text", labels)
        score, matched = scores[0]
        assert score == 0.0
        assert matched == []


# --- GitHub label conversion ---

class TestGithubLabelConversion:
    def test_basic_conversion(self):
        raw = [
            {"name": "bug", "description": "Something broken", "color": "d73a4a"},
            {"name": "enhancement", "description": "New feature", "color": "a2eeef"},
        ]
        taxonomy = github_labels_to_taxonomy(raw)
        assert len(taxonomy) == 2
        assert taxonomy[0].name == "bug"
        assert taxonomy[0].source == "github"
        assert taxonomy[1].name == "enhancement"

    def test_empty_description(self):
        raw = [{"name": "wontfix", "description": None, "color": "ffffff"}]
        taxonomy = github_labels_to_taxonomy(raw)
        assert len(taxonomy) == 1
        assert taxonomy[0].description == ""

    def test_skip_empty_name(self):
        raw = [{"name": "", "description": "orphan"}, {"name": "valid"}]
        taxonomy = github_labels_to_taxonomy(raw)
        assert len(taxonomy) == 1
        assert taxonomy[0].name == "valid"


# --- Classification ---

class TestClassification:
    def test_empty_taxonomy(self):
        pr = _make_pr()
        report = classify_item(pr, [1.0, 0.0], [], [], threshold=0.3)
        assert report.suggestions == []
        assert report.taxonomy_size == 0

    def test_above_threshold(self):
        pr = _make_pr(title="Fix security bug")
        labels = [_make_label(name="security", description="Security issues")]
        # Use identical embedding -> sim = 1.0
        emb = [1.0, 0.0]
        report = classify_item(
            pr, emb, labels, [emb], threshold=0.3, keyword_weight=0.0,
        )
        assert len(report.suggestions) >= 1
        assert report.suggestions[0].label == "security"
        assert report.suggestions[0].confidence >= 0.3

    def test_below_threshold(self):
        pr = _make_pr(title="Fix bug")
        labels = [_make_label(name="docs", description="Documentation")]
        # Orthogonal embeddings -> sim = 0.0
        report = classify_item(
            pr, [1.0, 0.0], labels, [[0.0, 1.0]],
            threshold=0.5, keyword_weight=0.0,
        )
        assert len(report.suggestions) == 0

    def test_max_suggestions_limits_output(self):
        pr = _make_pr(title="Big change")
        labels = [
            _make_label(name=f"label-{i}", description=f"Label {i}")
            for i in range(10)
        ]
        # All same embedding -> all above threshold
        emb = [1.0, 0.0]
        label_embs = [emb] * 10
        report = classify_item(
            pr, emb, labels, label_embs,
            threshold=0.1, keyword_weight=0.0, max_suggestions=3,
        )
        assert len(report.suggestions) <= 3

    def test_sorted_by_confidence(self):
        pr = _make_pr(title="Test")
        labels = [
            _make_label(name="low"),
            _make_label(name="high"),
        ]
        # Different similarity levels; use threshold=0.01 to avoid config default
        report = classify_item(
            pr, [1.0, 0.0], labels, [[0.3, 0.95], [0.99, 0.1]],
            threshold=0.01, keyword_weight=0.01,
        )
        assert len(report.suggestions) == 2
        assert report.suggestions[0].confidence >= report.suggestions[1].confidence

    def test_keyword_boost(self):
        """Keywords should boost confidence beyond pure embedding similarity."""
        pr = _make_pr(title="Fix a crash bug in auth")
        labels = [
            _make_label(name="bug", description="Bugs", keywords=["bug", "crash"]),
            _make_label(name="docs", description="Documentation"),
        ]
        # Both labels have same embedding similarity (identical embeddings)
        emb = [1.0, 0.0]
        report = classify_item(
            pr, emb, labels, [emb, emb],
            threshold=0.0, keyword_weight=0.3,
        )
        # bug should score higher due to keyword matches
        assert report.suggestions[0].label == "bug"
        assert len(report.suggestions[0].keyword_matches) >= 1

    def test_report_metadata(self):
        issue = _make_issue(number=42, title="Test issue", labels=["existing"])
        labels = [_make_label()]
        emb = [1.0, 0.0]
        report = classify_item(
            issue, emb, labels, [emb], threshold=0.3,
        )
        assert report.owner == "owner"
        assert report.repo == "repo"
        assert report.item_type == "issue"
        assert report.item_number == 42
        assert report.existing_labels == ["existing"]


# --- Taxonomy merging ---

class TestTaxonomyMerging:
    def test_vision_overrides_github(self):
        vision = [_make_label(name="bug", description="Vision bug def", source="vision")]
        github = [_make_label(name="bug", description="GitHub bug def", source="github")]
        merged = merge_taxonomies(vision, github)
        assert len(merged) == 1
        assert merged[0].source == "vision"
        assert merged[0].description == "Vision bug def"

    def test_github_supplements_vision(self):
        vision = [_make_label(name="bug", source="vision")]
        github = [_make_label(name="enhancement", source="github")]
        merged = merge_taxonomies(vision, github)
        assert len(merged) == 2
        names = {lb.name for lb in merged}
        assert names == {"bug", "enhancement"}


# --- Vision document extension ---

class TestVisionDocLabelTaxonomy:
    def test_parse_label_taxonomy_from_yaml(self, tmp_path):
        doc_path = tmp_path / "vision.yaml"
        doc_path.write_text(
            "project: TestProject\n"
            "principles: []\n"
            "label_taxonomy:\n"
            "  - name: bug\n"
            "    description: Something is broken\n"
            "    keywords: [error, crash, fix]\n"
            "  - name: feature\n"
            "    description: New functionality\n"
            "    keywords: [add, new]\n"
        )
        doc = load_vision_document(str(doc_path))
        assert len(doc.label_taxonomy) == 2
        assert doc.label_taxonomy[0].name == "bug"
        assert doc.label_taxonomy[0].source == "vision"
        assert "error" in doc.label_taxonomy[0].keywords
        assert doc.label_taxonomy[1].name == "feature"

    def test_missing_label_taxonomy_defaults_empty(self):
        doc = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        assert doc.label_taxonomy == []


# --- Scorecard ---

class TestLabelingScorecard:
    def test_json_serialization(self):
        report = LabelingReport(
            owner="owner", repo="repo", item_type="pr", item_number=1,
            item_title="Fix bug",
            existing_labels=["bug"],
            suggestions=[
                LabelSuggestion(
                    label="security", confidence=0.72,
                    embedding_similarity=0.65, keyword_matches=["auth"],
                    source="vision",
                ),
            ],
            taxonomy_source="vision", taxonomy_size=5, threshold=0.35,
        )
        json_str = labeling_report_to_json(report)
        data = json.loads(json_str)
        assert data["owner"] == "owner"
        assert data["item_type"] == "pr"
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["label"] == "security"

    def test_rich_rendering(self):
        console = Console(record=True, width=120)
        report = LabelingReport(
            owner="owner", repo="repo", item_type="issue", item_number=42,
            item_title="Login broken",
            existing_labels=[],
            suggestions=[
                LabelSuggestion(
                    label="bug", confidence=0.8,
                    embedding_similarity=0.7, keyword_matches=["bug"],
                    source="vision",
                ),
            ],
            taxonomy_source="merged", taxonomy_size=10, threshold=0.35,
        )
        render_labeling_report(report, console)
        output = console.export_text()
        assert "Label Automation" in output
        assert "owner/repo" in output
        assert "#42" in output
        assert "bug" in output
        assert "Suggested Labels" in output

    def test_rich_rendering_no_suggestions(self):
        console = Console(record=True, width=120)
        report = LabelingReport(
            owner="o", repo="r", item_type="pr", item_number=1,
        )
        render_labeling_report(report, console)
        output = console.export_text()
        assert "No labels above threshold" in output
