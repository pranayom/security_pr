"""Tests for the Gatekeeper multi-provider LLM module."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mcp_ai_auditor.gatekeeper.providers import (
    ProviderError,
    call_anthropic,
    call_gemini,
    call_openai_compatible,
    detect_provider_from_key,
    resolve_provider_and_key,
)


class TestDetectProviderFromKey:
    def test_anthropic_key(self):
        assert detect_provider_from_key("sk-ant-abc123") == "anthropic"

    def test_openrouter_key(self):
        assert detect_provider_from_key("sk-or-abc123") == "openrouter"

    def test_openai_key(self):
        assert detect_provider_from_key("sk-abc123") == "openai"

    def test_gemini_key(self):
        assert detect_provider_from_key("AIzaSyAbc123") == "gemini"

    def test_empty_key(self):
        assert detect_provider_from_key("") is None

    def test_unrecognized_key(self):
        assert detect_provider_from_key("some-random-key") is None

    def test_anthropic_prefix_exact(self):
        assert detect_provider_from_key("sk-ant-") == "anthropic"

    def test_openrouter_prefix_exact(self):
        assert detect_provider_from_key("sk-or-") == "openrouter"


class TestResolveProviderAndKey:
    def _settings(self, **kwargs):
        defaults = {
            "llm_provider": "auto",
            "llm_api_key": "",
            "openrouter_api_key": "",
            "anthropic_api_key": "",
            "openai_api_key": "",
            "gemini_api_key": "",
            "generic_api_key": "",
            "generic_base_url": "",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_explicit_provider_openrouter(self):
        s = self._settings(llm_provider="openrouter", openrouter_api_key="sk-or-test")
        provider, key = resolve_provider_and_key(s)
        assert provider == "openrouter"
        assert key == "sk-or-test"

    def test_explicit_provider_anthropic(self):
        s = self._settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test")
        provider, key = resolve_provider_and_key(s)
        assert provider == "anthropic"
        assert key == "sk-ant-test"

    def test_explicit_provider_uses_llm_api_key_fallback(self):
        s = self._settings(llm_provider="openai", llm_api_key="sk-unified")
        provider, key = resolve_provider_and_key(s)
        assert provider == "openai"
        assert key == "sk-unified"

    def test_auto_detects_from_llm_api_key(self):
        s = self._settings(llm_api_key="sk-ant-fromkey")
        provider, key = resolve_provider_and_key(s)
        assert provider == "anthropic"
        assert key == "sk-ant-fromkey"

    def test_auto_detects_openai_from_llm_api_key(self):
        s = self._settings(llm_api_key="sk-test123")
        provider, key = resolve_provider_and_key(s)
        assert provider == "openai"
        assert key == "sk-test123"

    def test_auto_falls_back_to_openrouter_key(self):
        s = self._settings(openrouter_api_key="sk-or-fallback")
        provider, key = resolve_provider_and_key(s)
        assert provider == "openrouter"
        assert key == "sk-or-fallback"

    def test_auto_falls_back_to_anthropic_key(self):
        s = self._settings(anthropic_api_key="sk-ant-fallback")
        provider, key = resolve_provider_and_key(s)
        assert provider == "anthropic"
        assert key == "sk-ant-fallback"

    def test_auto_falls_back_to_gemini_key(self):
        s = self._settings(gemini_api_key="AIzaTest")
        provider, key = resolve_provider_and_key(s)
        assert provider == "gemini"
        assert key == "AIzaTest"

    def test_auto_falls_back_to_generic(self):
        s = self._settings(generic_api_key="custom-key", generic_base_url="https://local.ai/v1")
        provider, key = resolve_provider_and_key(s)
        assert provider == "generic"
        assert key == "custom-key"

    def test_auto_no_keys_falls_back_to_claude_cli(self):
        s = self._settings()
        provider, key = resolve_provider_and_key(s)
        assert provider == "claude_cli"
        assert key == ""

    def test_auto_priority_openrouter_before_anthropic(self):
        s = self._settings(openrouter_api_key="sk-or-x", anthropic_api_key="sk-ant-x")
        provider, key = resolve_provider_and_key(s)
        assert provider == "openrouter"

    def test_auto_llm_api_key_takes_priority(self):
        s = self._settings(llm_api_key="AIzaGemini", openrouter_api_key="sk-or-x")
        provider, key = resolve_provider_and_key(s)
        assert provider == "gemini"
        assert key == "AIzaGemini"

    def test_generic_without_base_url_skipped(self):
        s = self._settings(generic_api_key="key-only")
        provider, key = resolve_provider_and_key(s)
        assert provider == "claude_cli"  # generic needs base_url


class TestCallOpenaiCompatible:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        response_body = {
            "choices": [{
                "message": {
                    "content": json.dumps({"alignment_score": 0.85, "strengths": ["good"]})
                }
            }]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await call_openai_compatible(
                prompt="test prompt",
                system_prompt="test system",
                api_key="sk-test",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
            )

        assert result["alignment_score"] == 0.85

    @pytest.mark.asyncio
    async def test_no_api_key_raises(self):
        with pytest.raises(ProviderError, match="No API key"):
            await call_openai_compatible(
                prompt="test", system_prompt="test",
                api_key="", model="m", base_url="https://x.com",
            )

    @pytest.mark.asyncio
    async def test_no_base_url_raises(self):
        with pytest.raises(ProviderError, match="No base URL"):
            await call_openai_compatible(
                prompt="test", system_prompt="test",
                api_key="key", model="m", base_url="",
            )

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        mock_response = httpx.Response(429, text="Rate limited")

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="429"):
                await call_openai_compatible(
                    prompt="test", system_prompt="test",
                    api_key="key", model="m", base_url="https://x.com/v1",
                )

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="timed out"):
                await call_openai_compatible(
                    prompt="test", system_prompt="test",
                    api_key="key", model="m", base_url="https://x.com/v1",
                )

    @pytest.mark.asyncio
    async def test_invalid_json_raises(self):
        response_body = {
            "choices": [{"message": {"content": "not json{{"}}]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="JSON"):
                await call_openai_compatible(
                    prompt="test", system_prompt="test",
                    api_key="key", model="m", base_url="https://x.com/v1",
                )

    @pytest.mark.asyncio
    async def test_json_schema_included_in_payload(self):
        response_body = {
            "choices": [{"message": {"content": json.dumps({"ok": True})}}]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            schema = {"name": "test", "strict": True, "schema": {"type": "object"}}
            await call_openai_compatible(
                prompt="test", system_prompt="test", json_schema=schema,
                api_key="key", model="m", base_url="https://x.com/v1",
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["response_format"]["type"] == "json_schema"
            assert payload["response_format"]["json_schema"]["strict"] is True


class TestCallAnthropic:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        response_body = {
            "content": [{"text": json.dumps({"alignment_score": 0.9})}]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await call_anthropic(
                prompt="test", system_prompt="system",
                api_key="sk-ant-test", model="claude-sonnet-4-20250514",
            )

        assert result["alignment_score"] == 0.9

    @pytest.mark.asyncio
    async def test_uses_correct_headers(self):
        response_body = {"content": [{"text": json.dumps({"ok": True})}]}
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await call_anthropic(
                prompt="test", system_prompt="system",
                api_key="sk-ant-test",
            )

            headers = mock_client.post.call_args.kwargs["headers"]
            assert headers["x-api-key"] == "sk-ant-test"
            assert "anthropic-version" in headers

    @pytest.mark.asyncio
    async def test_system_prompt_is_top_level(self):
        response_body = {"content": [{"text": json.dumps({"ok": True})}]}
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await call_anthropic(
                prompt="test", system_prompt="be helpful",
                api_key="sk-ant-test",
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["system"] == "be helpful"
            # system should NOT be in messages
            for msg in payload["messages"]:
                assert msg["role"] != "system"

    @pytest.mark.asyncio
    async def test_no_api_key_raises(self):
        with pytest.raises(ProviderError, match="No API key"):
            await call_anthropic(prompt="test", system_prompt="test", api_key="")

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        mock_response = httpx.Response(401, text="Unauthorized")

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="401"):
                await call_anthropic(
                    prompt="test", system_prompt="test",
                    api_key="sk-ant-test",
                )


class TestCallGemini:
    @pytest.mark.asyncio
    async def test_successful_call(self):
        response_body = {
            "candidates": [{
                "content": {"parts": [{"text": json.dumps({"alignment_score": 0.7})}]}
            }]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await call_gemini(
                prompt="test", system_prompt="system",
                api_key="AIzaTest123",
            )

        assert result["alignment_score"] == 0.7

    @pytest.mark.asyncio
    async def test_api_key_in_url(self):
        response_body = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"ok": True})}]}}]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await call_gemini(
                prompt="test", system_prompt="system",
                api_key="AIzaTest123", model="gemini-2.0-flash",
            )

            url = mock_client.post.call_args.args[0]
            assert "key=AIzaTest123" in url
            assert "gemini-2.0-flash" in url

    @pytest.mark.asyncio
    async def test_response_mime_type_set(self):
        response_body = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({"ok": True})}]}}]
        }
        mock_response = httpx.Response(200, json=response_body)

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            await call_gemini(
                prompt="test", system_prompt="system",
                api_key="AIzaTest",
            )

            payload = mock_client.post.call_args.kwargs["json"]
            assert payload["generationConfig"]["responseMimeType"] == "application/json"

    @pytest.mark.asyncio
    async def test_no_api_key_raises(self):
        with pytest.raises(ProviderError, match="No API key"):
            await call_gemini(prompt="test", system_prompt="test", api_key="")

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        mock_response = httpx.Response(403, text="Forbidden")

        with patch("mcp_ai_auditor.gatekeeper.providers.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            with pytest.raises(ProviderError, match="403"):
                await call_gemini(
                    prompt="test", system_prompt="test",
                    api_key="AIzaTest",
                )
