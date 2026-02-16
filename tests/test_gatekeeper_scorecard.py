"""Tests for the Gatekeeper scorecard formatting."""

import json
import os

import pytest
from rich.console import Console

from src.gatekeeper.models import (
    AssessmentScorecard,
    DedupResult,
    DimensionScore,
    FlagSeverity,
    HeuristicsResult,
    SuspicionFlag,
    TierOutcome,
    Verdict,
)
from src.gatekeeper.scorecard import render_scorecard, scorecard_to_json


def _make_scorecard(verdict: Verdict = Verdict.FAST_TRACK, **kwargs) -> AssessmentScorecard:
    defaults = dict(
        owner="owner",
        repo="repo",
        pr_number=42,
        verdict=verdict,
        confidence=0.8,
        dimensions=[
            DimensionScore(dimension="Hygiene & Dedup", score=1.0, summary="No duplicates"),
            DimensionScore(dimension="Supply Chain Suspicion", score=0.9, summary="Low suspicion"),
        ],
        dedup_result=DedupResult(outcome=TierOutcome.PASS),
        heuristics_result=HeuristicsResult(outcome=TierOutcome.PASS, suspicion_score=0.1),
        flags=[],
        summary="PR passed all tiers.",
    )
    defaults.update(kwargs)
    return AssessmentScorecard(**defaults)


class TestScorecardToJson:
    def test_valid_json_output(self):
        scorecard = _make_scorecard()
        json_str = scorecard_to_json(scorecard)
        data = json.loads(json_str)

        assert data["verdict"] == "fast_track"
        assert data["owner"] == "owner"
        assert data["repo"] == "repo"
        assert data["pr_number"] == 42
        assert len(data["dimensions"]) == 2

    def test_review_required_json(self):
        scorecard = _make_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=[
                SuspicionFlag(
                    rule_id="new_account",
                    severity=FlagSeverity.MEDIUM,
                    title="New account",
                    explanation="Account is 5 days old",
                ),
            ],
        )
        data = json.loads(scorecard_to_json(scorecard))
        assert data["verdict"] == "review_required"
        assert len(data["flags"]) == 1
        assert data["flags"][0]["rule_id"] == "new_account"

    def test_recommend_close_json(self):
        scorecard = _make_scorecard(verdict=Verdict.RECOMMEND_CLOSE)
        data = json.loads(scorecard_to_json(scorecard))
        assert data["verdict"] == "recommend_close"


class TestRenderScorecard:
    def test_render_fast_track(self):
        """Rendering FAST_TRACK scorecard doesn't crash."""
        scorecard = _make_scorecard()
        console = Console(file=open(os.devnull, "w"), force_terminal=True)
        render_scorecard(scorecard, console)

    def test_render_review_required_with_flags(self):
        """Rendering scorecard with flags doesn't crash."""
        scorecard = _make_scorecard(
            verdict=Verdict.REVIEW_REQUIRED,
            flags=[
                SuspicionFlag(
                    rule_id="new_account",
                    severity=FlagSeverity.MEDIUM,
                    title="New account",
                    explanation="Account is 5 days old",
                ),
                SuspicionFlag(
                    rule_id="sensitive_paths",
                    severity=FlagSeverity.HIGH,
                    title="Sensitive paths",
                    explanation="Touches auth code",
                ),
            ],
        )
        console = Console(file=open(os.devnull, "w"), force_terminal=True)
        render_scorecard(scorecard, console)

    def test_render_recommend_close(self):
        """Rendering RECOMMEND_CLOSE scorecard doesn't crash."""
        scorecard = _make_scorecard(verdict=Verdict.RECOMMEND_CLOSE)
        console = Console(file=open(os.devnull, "w"), force_terminal=True)
        render_scorecard(scorecard, console)

    def test_render_no_dimensions(self):
        """Rendering scorecard with no dimensions doesn't crash."""
        scorecard = _make_scorecard(dimensions=[])
        console = Console(file=open(os.devnull, "w"), force_terminal=True)
        render_scorecard(scorecard, console)

    def test_render_default_console(self):
        """Rendering with no explicit console uses default."""
        scorecard = _make_scorecard()
        # Should not raise
        render_scorecard(scorecard)
