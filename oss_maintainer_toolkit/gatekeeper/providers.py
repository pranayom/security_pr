"""Multi-provider LLM transport for Gatekeeper Tier 3.

Supports: OpenRouter, OpenAI, Anthropic, Google Gemini, Generic OpenAI-compatible, Claude CLI.
Auto-detects provider from API key prefix when llm_provider is set to "auto".
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""


def detect_provider_from_key(api_key: str) -> str | None:
    """Auto-detect LLM provider from API key prefix.

    Returns provider name or None if unrecognized.
    """
    if not api_key:
        return None
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("sk-or-"):
        return "openrouter"
    if api_key.startswith("sk-"):
        return "openai"
    if api_key.startswith("AIza"):
        return "gemini"
    return None


def resolve_provider_and_key(settings: Any) -> tuple[str, str]:
    """Resolve effective (provider, api_key) from GatekeeperSettings.

    Logic:
    - Explicit provider (non-"auto") → use that + matching key
    - Auto mode → detect from llm_api_key, then check provider-specific keys,
      then fall back to claude_cli
    """
    provider = settings.llm_provider

    if provider != "auto":
        # Explicit provider — find the matching key
        key = _get_key_for_provider(provider, settings)
        return (provider, key)

    # Auto mode: try llm_api_key first
    if settings.llm_api_key:
        detected = detect_provider_from_key(settings.llm_api_key)
        if detected:
            return (detected, settings.llm_api_key)

    # Check provider-specific keys in priority order
    if settings.openrouter_api_key:
        return ("openrouter", settings.openrouter_api_key)
    if settings.anthropic_api_key:
        return ("anthropic", settings.anthropic_api_key)
    if settings.openai_api_key:
        return ("openai", settings.openai_api_key)
    if settings.gemini_api_key:
        return ("gemini", settings.gemini_api_key)
    if settings.generic_api_key and settings.generic_base_url:
        return ("generic", settings.generic_api_key)

    # No keys found — fall back to claude_cli
    return ("claude_cli", "")


def _get_key_for_provider(provider: str, settings: Any) -> str:
    """Get the API key for a specific provider from settings."""
    key_map = {
        "openrouter": lambda s: s.llm_api_key or s.openrouter_api_key,
        "openai": lambda s: s.llm_api_key or s.openai_api_key,
        "anthropic": lambda s: s.llm_api_key or s.anthropic_api_key,
        "gemini": lambda s: s.llm_api_key or s.gemini_api_key,
        "generic": lambda s: s.llm_api_key or s.generic_api_key,
        "claude_cli": lambda s: "",
    }
    getter = key_map.get(provider)
    if getter:
        return getter(settings)
    return settings.llm_api_key


async def call_openai_compatible(
    prompt: str,
    system_prompt: str,
    json_schema: dict | None = None,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout_seconds: int = 60,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    """Call an OpenAI-compatible chat completions endpoint.

    Used for OpenRouter, OpenAI direct, and generic providers.
    Returns the parsed JSON response content.

    Raises ProviderError on failure.
    """
    if not api_key:
        raise ProviderError("No API key provided for OpenAI-compatible provider.")
    if not base_url:
        raise ProviderError("No base URL provided for OpenAI-compatible provider.")

    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "seed": 42,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        if resp.status_code != 200:
            raise ProviderError(f"API returned {resp.status_code}: {resp.text[:500]}")

        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    except httpx.TimeoutException:
        raise ProviderError(f"Request timed out after {timeout_seconds}s")
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected response structure: {e}")
    except json.JSONDecodeError as e:
        raise ProviderError(f"Failed to parse response as JSON: {e}")


async def call_anthropic(
    prompt: str,
    system_prompt: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-20250514",
    base_url: str = "https://api.anthropic.com",
    timeout_seconds: int = 60,
) -> dict:
    """Call the Anthropic Messages API.

    JSON output is enforced via prompt instruction (no structured output mode).
    Returns the parsed JSON response content.

    Raises ProviderError on failure.
    """
    if not api_key:
        raise ProviderError("No API key provided for Anthropic provider.")

    payload = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(
                f"{base_url}/v1/messages",
                headers=headers,
                json=payload,
            )

        if resp.status_code != 200:
            raise ProviderError(f"Anthropic API returned {resp.status_code}: {resp.text[:500]}")

        body = resp.json()
        # Anthropic returns content as a list of blocks
        text = body["content"][0]["text"]
        return json.loads(text)

    except httpx.TimeoutException:
        raise ProviderError(f"Anthropic request timed out after {timeout_seconds}s")
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected Anthropic response structure: {e}")
    except json.JSONDecodeError as e:
        raise ProviderError(f"Failed to parse Anthropic response as JSON: {e}")


async def call_gemini(
    prompt: str,
    system_prompt: str,
    api_key: str = "",
    model: str = "gemini-2.0-flash",
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    timeout_seconds: int = 60,
) -> dict:
    """Call the Google Gemini generateContent API.

    JSON output is enforced via responseMimeType.
    Returns the parsed JSON response content.

    Raises ProviderError on failure.
    """
    if not api_key:
        raise ProviderError("No API key provided for Gemini provider.")

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {"parts": [{"text": prompt}]},
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(
                f"{base_url}/models/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )

        if resp.status_code != 200:
            raise ProviderError(f"Gemini API returned {resp.status_code}: {resp.text[:500]}")

        body = resp.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)

    except httpx.TimeoutException:
        raise ProviderError(f"Gemini request timed out after {timeout_seconds}s")
    except (KeyError, IndexError) as e:
        raise ProviderError(f"Unexpected Gemini response structure: {e}")
    except json.JSONDecodeError as e:
        raise ProviderError(f"Failed to parse Gemini response as JSON: {e}")
