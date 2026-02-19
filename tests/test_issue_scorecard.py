"""Tests for the Gatekeeper issue scorecard rendering."""

import json

import pytest
from rich.console import Console

from oss_maintainer_toolkit.gatekeeper.models import (
    DimensionScore,
    FlagSeverity,
    IssueScorecard,
    SuspicionFlag,
    Verdict,
)
from oss_maintainer_toolkit.gatekeeper.issue_scorecard import (
    issue_scorecard_to_json,
    render_issue_scorecard,
)


def _make_scorecard(**kwargs) -> IssueScorecard:
    defaults = {
        "owner": "owner",
        "repo": "repo",
        "issue_number": 101,
        "verdict": Verdict.FAST_TRACK,
        "confidence": 0.8,
        "summary": "Issue passed all tiers.",
    }
    defaults.update(kwargs)
    return IssueScorecard(**defaults)


class TestIssueScorecardJson:
    def test_json_serialization(self):
        scorecard = _make_scorecard()
        json_str = issue_scorecard_to_json(scorecard)
        data = json.loads(json_str)
        assert data["owner"] == "owner"
        assert data["issue_number"] == 101
        assert data["verdict"] == "fast_track"

    def test_json_with_dimensions(self):
        scorecard = _make_scorecard(
            dimensions=[
                DimensionScore(dimension="Issue Dedup", score=1.0, summary="No duplicates"),
                DimensionScore(dimension="Issue Quality", score=0.8, summary="Good quality"),
            ]
        )
        data = json.loads(issue_scorecard_to_json(scorecard))
        assert len(data["dimensions"]) == 2


class TestRenderIssueScorecard:
    def test_render_fast_track(self):
        console = Console(record=True, width=120)
        scorecard = _make_scorecard(verdict=Verdict.FAST_TRACK)
        render_issue_scorecard(scorecard, console)
        output = console.export_text()
        assert "FAST TRACK" in output
        assert "Issue: owner/repo#101" in output

    def test_render_with_flags(self):
        console = Console(record=True, width=120)
        scorecard = _make_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=[
                SuspicionFlag(
                    rule_id="vague_description",
                    severity=FlagSeverity.MEDIUM,
                    title="Vague description",
                    explanation="Body is too short",
                )
            ],
        )
        render_issue_scorecard(scorecard, console)
        output = console.export_text()
        assert "REVIEW REQUIRED" in output
        assert "Vague description" in output
