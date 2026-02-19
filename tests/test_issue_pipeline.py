"""Tests for the Gatekeeper issue pipeline orchestrator."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from oss_maintainer_toolkit.gatekeeper.models import (
    DedupResult,
    IssueAuthor,
    IssueMetadata,
    IssueScorecard,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)
from oss_maintainer_toolkit.gatekeeper.issue_pipeline import run_issue_pipeline


def _make_issue(
    number: int = 1,
    title: str = "Clear and detailed bug report about login",
    body: str = "This is a detailed description with steps to reproduce the problem.",
    account_age_days: int = 200,
    contributions: int = 5,
) -> IssueMetadata:
    return IssueMetadata(
        owner="owner",
        repo="repo",
        number=number,
        title=title,
        body=body,
        author=IssueAuthor(
            login="testuser",
            account_created_at=datetime.now(timezone.utc) - timedelta(days=account_age_days),
            contributions_to_repo=contributions,
        ),
        created_at=datetime.now(timezone.utc),
    )


class TestIssuePipeline:
    @pytest.mark.asyncio
    async def test_clean_issue_fast_tracks_no_tier3(self):
        """Clean issue with no embedding and no vision → FAST_TRACK."""
        issue = _make_issue()
        scorecard = await run_issue_pipeline(issue, enable_tier3=False)
        assert scorecard.verdict == Verdict.FAST_TRACK
        assert scorecard.issue_number == 1

    @pytest.mark.asyncio
    async def test_duplicate_issue_recommend_close(self):
        """Duplicate detected in Tier 1 → RECOMMEND_CLOSE, pipeline stops."""
        issue = _make_issue(number=1)
        existing = _make_issue(number=2, title="Clear and detailed bug report about login")

        # Use identical embeddings to trigger duplicate
        scorecard = await run_issue_pipeline(
            issue,
            issue_embedding=[1.0, 0.0],
            existing_issues=[existing],
            existing_embeddings=[[1.0, 0.0]],
            enable_tier3=False,
        )
        assert scorecard.verdict == Verdict.RECOMMEND_CLOSE
        assert scorecard.dedup_result is not None
        assert scorecard.dedup_result.is_duplicate

    @pytest.mark.asyncio
    async def test_suspicious_issue_review_required(self):
        """Issue with many flags → REVIEW_REQUIRED at Tier 2."""
        issue = IssueMetadata(
            owner="owner",
            repo="repo",
            number=1,
            title="BUG HELP",
            body="bug",
            author=IssueAuthor(
                login="testuser",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=5),
                contributions_to_repo=0,
            ),
            labels=["bug"],
            created_at=datetime.now(timezone.utc),
        )
        scorecard = await run_issue_pipeline(issue, enable_tier3=False)
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        assert len(scorecard.flags) > 0

    @pytest.mark.asyncio
    async def test_skipped_dedup_when_no_embedding(self):
        """Tier 1 skipped when no embedding provided."""
        issue = _make_issue()
        scorecard = await run_issue_pipeline(issue, enable_tier3=False)
        assert scorecard.dedup_result is not None
        assert scorecard.dedup_result.outcome == TierOutcome.SKIPPED

    @pytest.mark.asyncio
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.run_issue_vision_alignment")
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.load_vision_document")
    async def test_tier3_low_alignment_review_required(self, mock_load, mock_vision):
        """Low vision alignment → REVIEW_REQUIRED."""
        mock_load.return_value = type("V", (), {
            "project": "test",
            "principles": [],
            "anti_patterns": [],
            "focus_areas": [],
        })()
        mock_vision.return_value = VisionAlignmentResult(
            outcome=TierOutcome.PASS,
            alignment_score=0.2,
            violated_principles=["scope"],
        )

        issue = _make_issue()
        scorecard = await run_issue_pipeline(
            issue,
            vision_document_path="/fake/vision.yaml",
            enable_tier3=True,
        )
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED

    @pytest.mark.asyncio
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.run_issue_vision_alignment")
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.load_vision_document")
    async def test_tier3_high_alignment_fast_track(self, mock_load, mock_vision):
        """High vision alignment + clean issue → FAST_TRACK."""
        mock_load.return_value = type("V", (), {
            "project": "test",
            "principles": [],
            "anti_patterns": [],
            "focus_areas": [],
        })()
        mock_vision.return_value = VisionAlignmentResult(
            outcome=TierOutcome.PASS,
            alignment_score=0.9,
            strengths=["well-aligned"],
        )

        issue = _make_issue()
        scorecard = await run_issue_pipeline(
            issue,
            vision_document_path="/fake/vision.yaml",
            enable_tier3=True,
        )
        assert scorecard.verdict == Verdict.FAST_TRACK
        assert scorecard.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.run_issue_vision_alignment")
    @patch("oss_maintainer_toolkit.gatekeeper.issue_pipeline.load_vision_document")
    async def test_tier3_error_review_required(self, mock_load, mock_vision):
        """Vision error → REVIEW_REQUIRED."""
        mock_load.return_value = type("V", (), {
            "project": "test",
            "principles": [],
            "anti_patterns": [],
            "focus_areas": [],
        })()
        mock_vision.return_value = VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=["Provider error"],
        )

        issue = _make_issue()
        scorecard = await run_issue_pipeline(
            issue,
            vision_document_path="/fake/vision.yaml",
            enable_tier3=True,
        )
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED

    @pytest.mark.asyncio
    async def test_scorecard_dimensions_present(self):
        """Scorecard should include dimension scores."""
        issue = _make_issue()
        scorecard = await run_issue_pipeline(issue, enable_tier3=False)
        dimension_names = [d.dimension for d in scorecard.dimensions]
        assert "Issue Dedup" in dimension_names
        assert "Issue Quality" in dimension_names
