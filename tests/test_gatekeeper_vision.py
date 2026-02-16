"""Tests for the Gatekeeper vision alignment module (Tier 3)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.gatekeeper.models import PRAuthor, PRMetadata, TierOutcome
from src.gatekeeper.vision import (
    _build_prompt,
    load_vision_document,
    run_vision_alignment,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadVisionDocument:
    def test_load_valid_document(self):
        doc = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        assert doc.project == "OpenClaw"
        assert len(doc.principles) == 3
        assert doc.principles[0].name == "Security First"
        assert len(doc.anti_patterns) == 3
        assert len(doc.focus_areas) == 3

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_vision_document("/nonexistent/vision.yaml")


class TestBuildPrompt:
    def test_prompt_contains_key_elements(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="nicoseng",
            repo="OpenClaw",
            number=42,
            title="Add validation",
            body="This adds input validation",
            author=PRAuthor(login="contributor"),
            diff_text="diff text here",
        )

        prompt = _build_prompt(pr, vision)

        assert "OpenClaw" in prompt
        assert "Security First" in prompt
        assert "Add validation" in prompt
        assert "contributor" in prompt
        assert "alignment_score" in prompt
        assert "violated_principles" in prompt

    def test_prompt_truncates_long_diff(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=1, title="t",
            author=PRAuthor(login="u"),
            diff_text="x" * 10000,
        )
        prompt = _build_prompt(pr, vision)
        # Diff should be truncated to 5000 chars
        assert len(prompt) < 15000


class TestRunVisionAlignment:
    @pytest.mark.asyncio
    async def test_successful_alignment(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        response = json.loads((FIXTURES / "sample_claude_response.json").read_text())

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(response).encode(), b"")
        )
        mock_process.returncode = 0

        with patch("src.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision)

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.85
        assert len(result.strengths) == 2
        assert result.violated_principles == []

    @pytest.mark.asyncio
    async def test_low_alignment_gated(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        response = {
            "alignment_score": 0.2,
            "violated_principles": ["Security First"],
            "strengths": [],
            "concerns": ["Bypasses authentication checks"],
        }

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(json.dumps(response).encode(), b"")
        )
        mock_process.returncode = 0

        with patch("src.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision)

        assert result.outcome == TierOutcome.GATED
        assert result.alignment_score == 0.2
        assert "Security First" in result.violated_principles

    @pytest.mark.asyncio
    async def test_cli_not_found(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        with patch(
            "src.gatekeeper.vision.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            result = await run_vision_alignment(pr, vision, claude_command="nonexistent")

        assert result.outcome == TierOutcome.ERROR
        assert any("not found" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_cli_timeout(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("src.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, timeout_seconds=1)

        assert result.outcome == TierOutcome.ERROR
        assert any("timed out" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_cli_nonzero_exit(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Error: something went wrong")
        )
        mock_process.returncode = 1

        with patch("src.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision)

        assert result.outcome == TierOutcome.ERROR

    @pytest.mark.asyncio
    async def test_invalid_json_response(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(
            return_value=(b"not valid json{{{", b"")
        )
        mock_process.returncode = 0

        with patch("src.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision)

        assert result.outcome == TierOutcome.ERROR
        assert any("JSON" in c for c in result.concerns)
