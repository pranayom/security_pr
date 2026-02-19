"""Tests for the Gatekeeper vision alignment module (Tier 3)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from oss_maintainer_toolkit.gatekeeper.models import PRAuthor, PRMetadata, TierOutcome
from oss_maintainer_toolkit.gatekeeper.vision import (
    SCORECARD_SCHEMA,
    _build_prompt,
    _build_schema_instruction,
    _parse_response,
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
        assert "violated" in prompt

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


class TestParseResponse:
    def test_high_alignment_passes(self):
        result = _parse_response({
            "alignment_score": 0.85,
            "violated_principles": [],
            "strengths": ["Good tests"],
            "concerns": [],
        })
        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.85

    def test_low_alignment_gated(self):
        result = _parse_response({
            "alignment_score": 0.2,
            "violated_principles": ["Security First"],
            "strengths": [],
            "concerns": ["Bypasses auth"],
        })
        assert result.outcome == TierOutcome.GATED
        assert result.alignment_score == 0.2
        assert "Security First" in result.violated_principles

    def test_missing_fields_default_to_empty(self):
        result = _parse_response({"alignment_score": 0.5})
        assert result.violated_principles == []
        assert result.strengths == []
        assert result.concerns == []


class TestScorecardSchema:
    def test_schema_has_required_fields(self):
        schema = SCORECARD_SCHEMA["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {
            "alignment_score", "violated_principles", "strengths", "concerns"
        }

    def test_schema_is_strict(self):
        assert SCORECARD_SCHEMA["strict"] is True


class TestOpenRouterProvider:
    @pytest.mark.asyncio
    async def test_successful_alignment(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {
            "alignment_score": 0.85,
            "violated_principles": [],
            "strengths": ["Good tests", "Clean code"],
            "concerns": [],
        }

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.85
        assert len(result.strengths) == 2
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_alignment_gated(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {
            "alignment_score": 0.2,
            "violated_principles": ["Security First"],
            "strengths": [],
            "concerns": ["Bypasses authentication checks"],
        }

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.GATED
        assert result.alignment_score == 0.2
        assert "Security First" in result.violated_principles

    @pytest.mark.asyncio
    async def test_missing_api_key_errors(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        with patch("oss_maintainer_toolkit.gatekeeper.vision.gatekeeper_settings") as mock_settings:
            mock_settings.llm_provider = "openrouter"
            mock_settings.openrouter_api_key = ""
            mock_settings.openrouter_model = "openai/gpt-oss-120b:free"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.openrouter_timeout_seconds = 60
            mock_settings.llm_api_key = ""
            mock_settings.llm_timeout_seconds = 60

            result = await run_vision_alignment(pr, vision, provider="openrouter")

        assert result.outcome == TierOutcome.ERROR
        assert any("API key" in c or "No API key" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_provider_error_returns_error_result(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ProviderError("API returned 429: Rate limited")

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.ERROR
        assert any("429" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_timeout_errors(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ProviderError("Request timed out after 60s")

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.ERROR
        assert any("timed out" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_invalid_json_in_content(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ProviderError("Failed to parse response as JSON: ...")

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.ERROR
        assert any("JSON" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_unexpected_response_structure(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ProviderError("Unexpected response structure: 'choices'")

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.ERROR
        assert any("response structure" in c for c in result.concerns)


class TestClaudeCliProvider:
    """Existing claude --print tests, now explicitly using provider='claude_cli'."""

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

        with patch("oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, provider="claude_cli")

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

        with patch("oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, provider="claude_cli")

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
            "oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            result = await run_vision_alignment(
                pr, vision, provider="claude_cli", claude_command="nonexistent",
            )

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

        with patch("oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(
                pr, vision, provider="claude_cli", timeout_seconds=1,
            )

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

        with patch("oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, provider="claude_cli")

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

        with patch("oss_maintainer_toolkit.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, provider="claude_cli")

        assert result.outcome == TierOutcome.ERROR
        assert any("JSON" in c for c in result.concerns)


class TestSchemaInstruction:
    def test_schema_instruction_contains_required_fields(self):
        instruction = _build_schema_instruction()
        assert "alignment_score" in instruction
        assert "violated_principles" in instruction
        assert "strengths" in instruction
        assert "concerns" in instruction
        assert "JSON" in instruction


class TestProviderDispatch:
    @pytest.mark.asyncio
    async def test_unknown_provider_errors(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        result = await run_vision_alignment(pr, vision, provider="unknown_provider")

        assert result.outcome == TierOutcome.ERROR
        assert any("Unknown LLM provider" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_default_provider_is_auto(self):
        """Verify the default provider from settings is auto."""
        from oss_maintainer_toolkit.gatekeeper.config import GatekeeperSettings
        settings = GatekeeperSettings()
        assert settings.llm_provider == "auto"

    @pytest.mark.asyncio
    async def test_anthropic_dispatch(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {
            "alignment_score": 0.75,
            "violated_principles": [],
            "strengths": ["Good"],
            "concerns": [],
        }

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_anthropic", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, provider="anthropic", api_key="sk-ant-test123",
            )

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.75
        mock_call.assert_called_once()
        # Verify schema instruction was appended
        call_prompt = mock_call.call_args.kwargs["prompt"]
        assert "alignment_score" in call_prompt
        assert "JSON" in call_prompt

    @pytest.mark.asyncio
    async def test_gemini_dispatch(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {
            "alignment_score": 0.6,
            "violated_principles": [],
            "strengths": [],
            "concerns": ["Minor issue"],
        }

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_gemini", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, provider="gemini", api_key="AIzaTest",
            )

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.6
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_openai_dispatch(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {
            "alignment_score": 0.9,
            "violated_principles": [],
            "strengths": ["Excellent"],
            "concerns": [],
        }

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_openai_compatible", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, provider="openai", api_key="sk-test123",
            )

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.9
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_detect_from_api_key(self):
        """Verify auto-detection dispatches to the right provider based on key prefix."""
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_data = {"alignment_score": 0.8, "violated_principles": [], "strengths": [], "concerns": []}

        with patch("oss_maintainer_toolkit.gatekeeper.vision.call_anthropic", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_data

            result = await run_vision_alignment(
                pr, vision, api_key="sk-ant-autodetect",
            )

        assert result.outcome == TierOutcome.PASS
        mock_call.assert_called_once()
