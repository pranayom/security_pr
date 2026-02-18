"""Tests for the Gatekeeper three-tier pipeline orchestrator."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcp_ai_auditor.gatekeeper.models import (
    AssessmentScorecard,
    DedupResult,
    HeuristicsResult,
    PRAuthor,
    PRFileChange,
    PRMetadata,
    SuspicionFlag,
    FlagSeverity,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)
from mcp_ai_auditor.gatekeeper.pipeline import run_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _make_pr(
    number: int = 42,
    files: list[PRFileChange] | None = None,
    **kwargs,
) -> PRMetadata:
    defaults = dict(
        owner="owner",
        repo="repo",
        number=number,
        title="Test PR",
        body="Test body",
        author=PRAuthor(login="user", contributions_to_repo=5),
        files=files or [],
        diff_text="",
    )
    defaults.update(kwargs)
    return PRMetadata(**defaults)


class TestPipelineGating:
    @pytest.mark.asyncio
    async def test_tier1_gates_on_duplicate(self):
        """When Tier 1 finds a duplicate, pipeline returns RECOMMEND_CLOSE immediately."""
        pr = _make_pr()
        existing = _make_pr(number=10)
        emb = [1.0, 0.0, 0.0]  # identical embedding = cosine sim 1.0

        scorecard = await run_pipeline(
            pr,
            pr_embedding=emb,
            existing_prs=[existing],
            existing_embeddings=[emb],
            enable_tier3=False,
        )

        assert scorecard.verdict == Verdict.RECOMMEND_CLOSE
        assert scorecard.dedup_result is not None
        assert scorecard.dedup_result.is_duplicate is True
        # Heuristics should NOT have run
        assert scorecard.heuristics_result is None

    @pytest.mark.asyncio
    async def test_tier2_gates_on_suspicion(self):
        """When Tier 2 suspicion score exceeds threshold, pipeline returns REVIEW_REQUIRED."""
        from datetime import datetime, timedelta, timezone

        files = [
            PRFileChange(filename="src/auth/login.py", additions=50),
            PRFileChange(filename="requirements.txt", additions=3),
        ]
        pr = _make_pr(
            files=files,
            body="Fixed stuff",
            author=PRAuthor(
                login="newuser",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=5),
                contributions_to_repo=0,
            ),
            total_additions=53,
            total_deletions=0,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)

        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        assert scorecard.heuristics_result is not None
        assert scorecard.heuristics_result.outcome == TierOutcome.GATED
        assert len(scorecard.flags) > 0
        # Vision should NOT have run
        assert scorecard.vision_result is None

    @pytest.mark.asyncio
    async def test_clean_pr_fast_tracks(self):
        """A clean PR with no flags passes all tiers and gets FAST_TRACK."""
        from datetime import datetime, timedelta, timezone

        files = [
            PRFileChange(filename="src/utils.py", additions=15),
            PRFileChange(filename="tests/test_utils.py", additions=10),
        ]
        pr = _make_pr(
            files=files,
            body="Refactored helpers",
            author=PRAuthor(
                login="trusteduser",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=365),
                contributions_to_repo=50,
            ),
            total_additions=25,
            total_deletions=0,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)

        assert scorecard.verdict == Verdict.FAST_TRACK
        assert "passed all tiers" in scorecard.summary

    @pytest.mark.asyncio
    async def test_tier1_skipped_without_embedding(self):
        """When no embedding is provided, Tier 1 is SKIPPED."""
        from datetime import datetime, timedelta, timezone

        pr = _make_pr(
            author=PRAuthor(
                login="user",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=365),
                contributions_to_repo=10,
            ),
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)

        assert scorecard.dedup_result is not None
        assert scorecard.dedup_result.outcome == TierOutcome.SKIPPED

    @pytest.mark.asyncio
    async def test_tier3_vision_alignment(self):
        """Tier 3 runs when vision_document_path is provided and Tiers 1+2 pass."""
        from datetime import datetime, timedelta, timezone

        files = [
            PRFileChange(filename="src/utils.py", additions=15),
            PRFileChange(filename="tests/test_utils.py", additions=10),
        ]
        pr = _make_pr(
            files=files,
            body="Refactored helpers",
            author=PRAuthor(
                login="trusteduser",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=365),
                contributions_to_repo=50,
            ),
            total_additions=25,
            total_deletions=0,
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.PASS,
            alignment_score=0.9,
            strengths=["Good tests"],
            concerns=[],
        )

        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            with patch("mcp_ai_auditor.gatekeeper.pipeline.load_vision_document"):
                scorecard = await run_pipeline(
                    pr,
                    vision_document_path=str(FIXTURES / "sample_vision_document.yaml"),
                    enable_tier3=True,
                )

        assert scorecard.verdict == Verdict.FAST_TRACK
        assert scorecard.vision_result is not None
        assert scorecard.vision_result.alignment_score == 0.9

    @pytest.mark.asyncio
    async def test_tier3_low_alignment_review_required(self):
        """Low vision alignment triggers REVIEW_REQUIRED."""
        from datetime import datetime, timedelta, timezone

        pr = _make_pr(
            author=PRAuthor(
                login="user",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=365),
                contributions_to_repo=10,
            ),
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.GATED,
            alignment_score=0.2,
            violated_principles=["Security First"],
            concerns=["Bypasses auth"],
        )

        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            with patch("mcp_ai_auditor.gatekeeper.pipeline.load_vision_document"):
                scorecard = await run_pipeline(
                    pr,
                    vision_document_path="vision.yaml",
                    enable_tier3=True,
                )

        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        assert "Low vision alignment" in scorecard.summary

    @pytest.mark.asyncio
    async def test_tier3_error_returns_review_required(self):
        """Vision tier error results in REVIEW_REQUIRED."""
        from datetime import datetime, timedelta, timezone

        pr = _make_pr(
            author=PRAuthor(
                login="user",
                account_created_at=datetime.now(timezone.utc) - timedelta(days=365),
                contributions_to_repo=10,
            ),
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=["claude CLI not found"],
        )

        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            with patch("mcp_ai_auditor.gatekeeper.pipeline.load_vision_document"):
                scorecard = await run_pipeline(
                    pr,
                    vision_document_path="vision.yaml",
                    enable_tier3=True,
                )

        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        assert "errored" in scorecard.summary

    @pytest.mark.asyncio
    async def test_scorecard_has_all_fields(self):
        """Scorecard always has owner, repo, pr_number, verdict, dimensions."""
        pr = _make_pr()
        scorecard = await run_pipeline(pr, enable_tier3=False)

        assert scorecard.owner == "owner"
        assert scorecard.repo == "repo"
        assert scorecard.pr_number == 42
        assert scorecard.verdict in list(Verdict)
        assert len(scorecard.dimensions) >= 1
        assert scorecard.summary != ""
