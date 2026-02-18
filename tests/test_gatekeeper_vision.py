"""Tests for the Gatekeeper vision alignment module (Tier 3)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mcp_ai_auditor.gatekeeper.models import PRAuthor, PRMetadata, TierOutcome
from mcp_ai_auditor.gatekeeper.vision import (
    SCORECARD_SCHEMA,
    _build_prompt,
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

        response_body = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "alignment_score": 0.85,
                        "violated_principles": [],
                        "strengths": ["Good tests", "Clean code"],
                        "concerns": [],
                    })
                }
            }]
        }

        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await run_vision_alignment(
                pr, vision, provider="openrouter", openrouter_api_key="test-key",
            )

        assert result.outcome == TierOutcome.PASS
        assert result.alignment_score == 0.85
        assert len(result.strengths) == 2

        # Verify the request payload structure
        call_args = mock_client.post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0
        assert payload["seed"] == 42
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True

    @pytest.mark.asyncio
    async def test_low_alignment_gated(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        response_body = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "alignment_score": 0.2,
                        "violated_principles": ["Security First"],
                        "strengths": [],
                        "concerns": ["Bypasses authentication checks"],
                    })
                }
            }]
        }

        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

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

        with patch("mcp_ai_auditor.gatekeeper.vision.gatekeeper_settings") as mock_settings:
            mock_settings.llm_provider = "openrouter"
            mock_settings.openrouter_api_key = ""
            mock_settings.openrouter_model = "openai/gpt-oss-120b:free"
            mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
            mock_settings.openrouter_timeout_seconds = 60

            result = await run_vision_alignment(pr, vision, provider="openrouter")

        assert result.outcome == TierOutcome.ERROR
        assert any("API key" in c for c in result.concerns)

    @pytest.mark.asyncio
    async def test_api_error_status(self):
        vision = load_vision_document(str(FIXTURES / "sample_vision_document.yaml"))
        pr = PRMetadata(
            owner="o", repo="r", number=42, title="Test",
            author=PRAuthor(login="u"),
        )

        mock_response = httpx.Response(429, text="Rate limited")

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

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

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

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

        response_body = {
            "choices": [{
                "message": {"content": "not valid json{{{"}
            }]
        }

        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

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

        # Missing choices key
        response_body = {"error": "something went wrong"}
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.vision.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

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

        with patch("mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
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

        with patch("mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
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
            "mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec",
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

        with patch("mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
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

        with patch("mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
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

        with patch("mcp_ai_auditor.gatekeeper.vision.asyncio.create_subprocess_exec", return_value=mock_process):
            result = await run_vision_alignment(pr, vision, provider="claude_cli")

        assert result.outcome == TierOutcome.ERROR
        assert any("JSON" in c for c in result.concerns)


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
    async def test_default_provider_is_openrouter(self):
        """Verify the default provider from settings is openrouter."""
        from mcp_ai_auditor.gatekeeper.config import GatekeeperSettings
        settings = GatekeeperSettings()
        assert settings.llm_provider == "openrouter"
