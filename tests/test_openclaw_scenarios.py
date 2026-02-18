"""Integration tests: OpenClaw-specific PR scenarios against the full pipeline.

Uses the real vision document (vision_documents/openclaw.yaml) with Tier 3 mocked.
Tests realistic attack patterns and legitimate contribution scenarios.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mcp_ai_auditor.gatekeeper.models import (
    PRAuthor,
    PRFileChange,
    PRMetadata,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)
from mcp_ai_auditor.gatekeeper.pipeline import run_pipeline
from mcp_ai_auditor.gatekeeper.vision import load_vision_document

VISION_DOC = str(Path(__file__).parent.parent / "vision_documents" / "openclaw.yaml")


def _openclaw_pr(
    number: int = 100,
    title: str = "Fix bug",
    body: str = "Fixed a bug",
    login: str = "contributor",
    account_age_days: int = 365,
    contributions: int = 10,
    files: list[PRFileChange] | None = None,
    total_additions: int = 0,
    total_deletions: int = 0,
) -> PRMetadata:
    return PRMetadata(
        owner="openclaw",
        repo="openclaw",
        number=number,
        title=title,
        body=body,
        author=PRAuthor(
            login=login,
            account_created_at=datetime.now(timezone.utc) - timedelta(days=account_age_days),
            contributions_to_repo=contributions,
        ),
        files=files or [],
        diff_text="",
        created_at=datetime.now(timezone.utc),
        total_additions=total_additions,
        total_deletions=total_deletions,
    )


class TestOpenClawVisionDocument:
    """Verify the vision document loads and has expected structure."""

    def test_loads_successfully(self):
        vision = load_vision_document(VISION_DOC)
        assert vision.project == "OpenClaw"

    def test_has_required_principles(self):
        vision = load_vision_document(VISION_DOC)
        names = {p.name for p in vision.principles}
        assert "Local-First Privacy" in names
        assert "Skills Ecosystem Integrity" in names
        assert "Architecture-First Contribution" in names

    def test_has_anti_patterns(self):
        vision = load_vision_document(VISION_DOC)
        assert len(vision.anti_patterns) >= 10

    def test_has_focus_areas(self):
        vision = load_vision_document(VISION_DOC)
        assert len(vision.focus_areas) >= 10
        # Should flag skill-related and credential paths
        assert "extensions/" in vision.focus_areas
        assert "credentials" in vision.focus_areas


class TestOpenClawLegitimateContributions:
    """Legitimate contributions should FAST_TRACK through Tiers 1+2."""

    @pytest.mark.asyncio
    async def test_bugfix_from_trusted_contributor(self):
        pr = _openclaw_pr(
            title="Fix WhatsApp reconnection race condition",
            body="Fixes #1234. The reconnection handler was not awaiting the session lock.",
            login="shadow",
            account_age_days=500,
            contributions=120,
            files=[
                PRFileChange(filename="src/channels/whatsapp/connection.ts", additions=8, deletions=3),
                PRFileChange(filename="src/channels/whatsapp/connection.test.ts", additions=15, deletions=0),
            ],
            total_additions=23,
            total_deletions=3,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.FAST_TRACK

    @pytest.mark.asyncio
    async def test_docs_improvement(self):
        pr = _openclaw_pr(
            title="Improve onboarding docs for Telegram setup",
            body="Clarifies the Telegram BotFather token configuration steps",
            contributions=5,
            files=[
                PRFileChange(filename="docs/channels/telegram.md", additions=25, deletions=10),
            ],
            total_additions=25,
            total_deletions=10,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.FAST_TRACK

    @pytest.mark.asyncio
    async def test_test_only_pr(self):
        pr = _openclaw_pr(
            title="Add missing tests for compaction logic",
            body="Increases coverage for token compaction from 45% to 72%",
            contributions=20,
            files=[
                PRFileChange(filename="src/core/compaction.test.ts", additions=120, deletions=0),
            ],
            total_additions=120,
            total_deletions=0,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.FAST_TRACK


class TestOpenClawSuspiciousContributions:
    """Supply chain attack patterns should be flagged REVIEW_REQUIRED."""

    @pytest.mark.asyncio
    async def test_new_account_touches_auth(self):
        """Classic supply chain: new account modifying auth paths without tests."""
        pr = _openclaw_pr(
            title="Refactor OAuth token refresh logic",
            body="Small cleanup",
            login="helpful-contributor-2026",
            account_age_days=7,
            contributions=0,
            files=[
                PRFileChange(filename="src/auth/oauth-profiles.ts", additions=45, deletions=12),
                PRFileChange(filename="src/auth/token-refresh.ts", additions=30, deletions=8),
            ],
            total_additions=75,
            total_deletions=20,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        rule_ids = {f.rule_id for f in scorecard.flags}
        assert "new_account" in rule_ids
        assert "first_contribution" in rule_ids
        assert "sensitive_paths" in rule_ids

    @pytest.mark.asyncio
    async def test_skill_loader_modification_no_tests(self):
        """Modifying skill execution paths without tests + sneaking in deps is high-risk."""
        pr = _openclaw_pr(
            title="Optimize skill loading performance",
            body="Cached skill manifests for faster startup",  # no mention of deps
            login="perf-optimizer",
            account_age_days=30,
            contributions=0,
            files=[
                PRFileChange(filename="extensions/skill-loader/src/index.ts", additions=80, deletions=15),
                PRFileChange(filename="extensions/skill-loader/src/cache.ts", status="added", additions=120, deletions=0),
                # Sneaked-in dependency change without justification
                PRFileChange(filename="package.json", additions=3, deletions=1),
            ],
            total_additions=203,
            total_deletions=16,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        rule_ids = {f.rule_id for f in scorecard.flags}
        assert "new_account" in rule_ids
        assert "low_test_ratio" in rule_ids
        assert "unjustified_deps" in rule_ids

    @pytest.mark.asyncio
    async def test_dependency_change_without_justification(self):
        """Unjustified dependency bumps are a classic supply chain vector."""
        pr = _openclaw_pr(
            title="Minor fixes and cleanup",
            body="Small fixes across the codebase",
            login="cleanup-bot",
            account_age_days=15,
            contributions=0,
            files=[
                PRFileChange(filename="src/utils/formatting.ts", additions=5, deletions=3),
                PRFileChange(filename="package.json", additions=2, deletions=1),
                PRFileChange(filename="pnpm-lock.yaml", additions=500, deletions=200),
            ],
            total_additions=507,
            total_deletions=204,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        rule_ids = {f.rule_id for f in scorecard.flags}
        assert "unjustified_deps" in rule_ids

    @pytest.mark.asyncio
    async def test_large_diff_hiding_credential_change(self):
        """Large refactor that hides a small change to credential handling."""
        pr = _openclaw_pr(
            title="Major refactor: modernize channel handlers",
            body="Modernized all channel handlers to use new base class pattern. Updated dependencies.",
            login="refactor-king",
            account_age_days=60,
            contributions=0,
            files=[
                PRFileChange(filename="src/channels/whatsapp/handler.ts", additions=200, deletions=180),
                PRFileChange(filename="src/channels/telegram/handler.ts", additions=180, deletions=160),
                PRFileChange(filename="src/channels/discord/handler.ts", additions=170, deletions=150),
                PRFileChange(filename="src/channels/slack/handler.ts", additions=160, deletions=140),
                # The suspicious change buried in the bulk
                PRFileChange(filename="src/auth/credentials.ts", additions=8, deletions=3),
            ],
            total_additions=718,
            total_deletions=633,
        )

        scorecard = await run_pipeline(pr, enable_tier3=False)
        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        rule_ids = {f.rule_id for f in scorecard.flags}
        assert "large_diff_hiding" in rule_ids


class TestOpenClawDuplicateDetection:
    """Tier 1 dedup should catch copy-paste PRs."""

    @pytest.mark.asyncio
    async def test_identical_embedding_flagged(self):
        pr = _openclaw_pr(number=200, title="Fix typo in README")
        existing = _openclaw_pr(number=150, title="Fix typo in README")
        emb = [0.5, 0.8, 0.2]

        scorecard = await run_pipeline(
            pr,
            pr_embedding=emb,
            existing_prs=[existing],
            existing_embeddings=[emb],
            enable_tier3=False,
        )

        assert scorecard.verdict == Verdict.RECOMMEND_CLOSE
        assert scorecard.dedup_result is not None
        assert scorecard.dedup_result.is_duplicate

    @pytest.mark.asyncio
    async def test_different_embeddings_pass(self):
        pr = _openclaw_pr(number=200, title="Add Matrix channel support")
        existing = _openclaw_pr(number=150, title="Fix WhatsApp reconnection")
        # Orthogonal embeddings → cosine similarity ≈ 0
        pr_emb = [1.0, 0.0, 0.0]
        existing_emb = [0.0, 1.0, 0.0]

        scorecard = await run_pipeline(
            pr,
            pr_embedding=pr_emb,
            existing_prs=[existing],
            existing_embeddings=[existing_emb],
            enable_tier3=False,
        )

        assert scorecard.dedup_result is not None
        assert not scorecard.dedup_result.is_duplicate


class TestOpenClawFocusAreasIntegration:
    """Vision doc focus_areas flow into heuristics as extra sensitive paths."""

    @pytest.mark.asyncio
    async def test_extensions_path_flagged_with_vision_doc(self):
        """extensions/* is not in default sensitive paths, but IS in OpenClaw focus_areas."""
        pr = _openclaw_pr(
            title="Update skill loader",
            body="Refactored skill loading",
            login="newdev",
            account_age_days=10,
            contributions=0,
            files=[
                # extensions/ is only sensitive when vision doc focus_areas are active
                PRFileChange(filename="extensions/skill-loader/src/index.ts", additions=30, deletions=5),
                PRFileChange(filename="extensions/skill-loader/src/index.test.ts", additions=20, deletions=0),
            ],
            total_additions=50,
            total_deletions=5,
        )

        # Without vision doc: extensions/ not flagged as sensitive
        scorecard_no_vision = await run_pipeline(pr, enable_tier3=False)
        flags_no_vision = {f.rule_id for f in scorecard_no_vision.flags}
        assert "sensitive_paths" not in flags_no_vision

        # With vision doc: extensions/ IS flagged via focus_areas
        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.PASS, alignment_score=0.8,
        )
        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            scorecard_with_vision = await run_pipeline(
                pr, vision_document_path=VISION_DOC, enable_tier3=True,
            )
        flags_with_vision = {f.rule_id for f in scorecard_with_vision.flags}
        assert "sensitive_paths" in flags_with_vision

    @pytest.mark.asyncio
    async def test_credential_path_flagged_with_vision_doc(self):
        """Credential paths from focus_areas trigger sensitive path detection."""
        pr = _openclaw_pr(
            title="Fix credential refresh",
            body="Token refresh was broken",
            contributions=20,
            files=[
                PRFileChange(filename="src/infrastructure/credentials/store.ts", additions=10, deletions=5),
                PRFileChange(filename="src/infrastructure/credentials/store.test.ts", additions=8, deletions=0),
            ],
            total_additions=18,
            total_deletions=5,
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.PASS, alignment_score=0.9,
        )
        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            scorecard = await run_pipeline(
                pr, vision_document_path=VISION_DOC, enable_tier3=True,
            )
        flags = {f.rule_id for f in scorecard.flags}
        assert "sensitive_paths" in flags


class TestOpenClawVisionTier3:
    """Tier 3 with real vision doc loading + mocked claude --print."""

    @pytest.mark.asyncio
    async def test_high_alignment_fast_tracks(self):
        pr = _openclaw_pr(
            title="Improve Telegram error handling",
            body="Better error messages when Telegram bot token is invalid",
            contributions=30,
            files=[
                PRFileChange(filename="src/channels/telegram/client.ts", additions=20, deletions=5),
                PRFileChange(filename="src/channels/telegram/client.test.ts", additions=15, deletions=0),
            ],
            total_additions=35,
            total_deletions=5,
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.PASS,
            alignment_score=0.85,
            strengths=["Improves stability", "Includes tests"],
            concerns=[],
        )

        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            scorecard = await run_pipeline(
                pr,
                vision_document_path=VISION_DOC,
                enable_tier3=True,
            )

        assert scorecard.verdict == Verdict.FAST_TRACK
        assert scorecard.vision_result is not None
        assert scorecard.vision_result.alignment_score == 0.85

    @pytest.mark.asyncio
    async def test_low_alignment_crypto_feature(self):
        """A crypto/DeFi feature should have low vision alignment."""
        pr = _openclaw_pr(
            title="Add DeFi wallet skill",
            body="New skill for managing crypto wallets through OpenClaw",
            contributions=5,
            files=[
                PRFileChange(filename="extensions/defi-wallet/src/index.ts", additions=300, deletions=0),
            ],
            total_additions=300,
            total_deletions=0,
        )

        mock_vision_result = VisionAlignmentResult(
            outcome=TierOutcome.GATED,
            alignment_score=0.15,
            violated_principles=["Skills Ecosystem Integrity"],
            concerns=["Crypto/DeFi features are explicitly unwelcome"],
        )

        with patch("mcp_ai_auditor.gatekeeper.pipeline.run_vision_alignment", new_callable=AsyncMock, return_value=mock_vision_result):
            scorecard = await run_pipeline(
                pr,
                vision_document_path=VISION_DOC,
                enable_tier3=True,
            )

        assert scorecard.verdict == Verdict.REVIEW_REQUIRED
        assert "Low vision alignment" in scorecard.summary
