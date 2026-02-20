"""Tier 3: Vision alignment assessment via multiple LLM providers.

Supports: OpenRouter, OpenAI, Anthropic, Gemini, Generic OpenAI-compatible, Claude CLI.
Provider auto-detection from API key prefix when llm_provider is "auto".
"""

from __future__ import annotations

import asyncio
import json

import yaml

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import (
    IssueMetadata,
    LabelDefinition,
    PRMetadata,
    TierOutcome,
    VisionAlignmentResult,
    VisionDocument,
    VisionPrinciple,
)
from oss_maintainer_toolkit.gatekeeper.providers import (
    ProviderError,
    call_anthropic,
    call_gemini,
    call_openai_compatible,
    resolve_provider_and_key,
)

# JSON Schema for structured outputs — enforced by OpenAI-compatible providers
SCORECARD_SCHEMA = {
    "name": "pr_vision_scorecard",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["alignment_score", "violated_principles", "strengths", "concerns"],
        "properties": {
            "alignment_score": {
                "type": "number",
                "description": "How well the PR aligns with the project vision (0.0 = no alignment, 1.0 = perfect alignment)",
            },
            "violated_principles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of vision principles that this PR violates",
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Positive aspects of this PR's alignment with the vision",
            },
            "concerns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concerns about this PR's alignment with the vision",
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You are a code reviewer assessing pull requests against a project's vision document. "
    "Return ONLY valid JSON matching the provided schema. No markdown, no extra keys, no extra text."
)


def _build_schema_instruction() -> str:
    """Build a text instruction describing the required JSON schema.

    Used for providers that don't support structured output mode (Anthropic, Gemini).
    """
    return (
        "\n\nYou MUST respond with ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "alignment_score": <number 0.0-1.0>,\n'
        '  "violated_principles": [<string>, ...],\n'
        '  "strengths": [<string>, ...],\n'
        '  "concerns": [<string>, ...]\n'
        "}\n"
        "No other keys, no markdown fences, no extra text."
    )


def load_vision_document(path: str) -> VisionDocument:
    """Load a YAML vision document from disk."""
    with open(path) as f:
        data = yaml.safe_load(f)

    principles = [
        VisionPrinciple(name=p["name"], description=p["description"])
        for p in data.get("principles", [])
    ]

    label_taxonomy = [
        LabelDefinition(
            name=lb["name"],
            description=lb.get("description", ""),
            keywords=lb.get("keywords", []),
            color=lb.get("color", ""),
            source="vision",
        )
        for lb in data.get("label_taxonomy", [])
        if lb.get("name")
    ]

    return VisionDocument(
        project=data.get("project", ""),
        principles=principles,
        anti_patterns=data.get("anti_patterns", []),
        focus_areas=data.get("focus_areas", []),
        label_taxonomy=label_taxonomy,
    )


def _build_prompt(pr: PRMetadata, vision: VisionDocument) -> str:
    """Build the user prompt for vision alignment assessment."""
    principles_text = "\n".join(
        f"- {p.name}: {p.description}" for p in vision.principles
    )
    anti_patterns_text = "\n".join(f"- {ap}" for ap in vision.anti_patterns)
    focus_areas_text = "\n".join(f"- {fa}" for fa in vision.focus_areas)

    # Truncate diff to avoid overwhelming the prompt
    diff_truncated = pr.diff_text[:5000] if pr.diff_text else "(no diff available)"

    files_list = "\n".join(
        f"- {f.filename} (+{f.additions}/-{f.deletions})" for f in pr.files
    )

    return f"""Assess this pull request for alignment with the project's vision document.

## Project: {vision.project}

### Vision Principles
{principles_text}

### Anti-Patterns to Watch For
{anti_patterns_text}

### Focus Areas
{focus_areas_text}

## Pull Request #{pr.number}: {pr.title}

**Author:** {pr.author.login}
**Description:** {pr.body[:2000] if pr.body else '(no description)'}

### Changed Files
{files_list}

### Diff (truncated)
```
{diff_truncated}
```

Evaluate alignment_score from 0.0 (violates vision) to 1.0 (perfect fit). List any violated principle names exactly as shown above."""


def _parse_response(data: dict) -> VisionAlignmentResult:
    """Parse a JSON response dict into a VisionAlignmentResult."""
    alignment_score = float(data.get("alignment_score", 0.0))
    outcome = TierOutcome.PASS if alignment_score >= 0.4 else TierOutcome.GATED

    return VisionAlignmentResult(
        outcome=outcome,
        alignment_score=alignment_score,
        violated_principles=data.get("violated_principles", []),
        strengths=data.get("strengths", []),
        concerns=data.get("concerns", []),
    )


# --- Claude CLI provider (stays here — subprocess, not HTTP) ---

async def _run_claude_cli(
    prompt: str,
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment via claude --print (requires local Max subscription)."""
    cmd = claude_command or gatekeeper_settings.claude_command
    timeout = timeout_seconds or gatekeeper_settings.claude_timeout_seconds

    try:
        process = await asyncio.create_subprocess_exec(
            cmd, "--print", "--output-format", "json", "-p", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

        if process.returncode != 0:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"claude CLI exited with code {process.returncode}: {stderr.decode()[:500]}"],
            )

        response_text = stdout.decode().strip()
        data = json.loads(response_text)
        return _parse_response(data)

    except asyncio.TimeoutError:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"claude CLI timed out after {timeout}s"],
        )
    except FileNotFoundError:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"claude CLI not found at '{cmd}'. Install Claude Code or check AUDITOR_GK_CLAUDE_COMMAND."],
        )
    except json.JSONDecodeError as e:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"Failed to parse claude response as JSON: {e}"],
        )


# --- Provider dispatch helpers ---

def _get_provider_config(provider: str, api_key: str, settings=None):
    """Get model, base_url, and timeout for a given provider."""
    s = settings or gatekeeper_settings
    configs = {
        "openrouter": (s.openrouter_model, s.openrouter_base_url, s.openrouter_timeout_seconds),
        "openai": (s.openai_model, s.openai_base_url, s.llm_timeout_seconds),
        "anthropic": (s.anthropic_model, s.anthropic_base_url, s.llm_timeout_seconds),
        "gemini": (s.gemini_model, s.gemini_base_url, s.llm_timeout_seconds),
        "generic": (s.generic_model, s.generic_base_url, s.llm_timeout_seconds),
    }
    return configs.get(provider, ("", "", s.llm_timeout_seconds))


# --- Shared provider dispatch ---

def _resolve_effective_provider(
    provider: str = "",
    api_key: str = "",
    openrouter_api_key: str = "",
) -> tuple[str, str]:
    """Resolve effective provider and key from explicit params + config.

    Returns (effective_provider, effective_key).
    """
    if provider and provider != "auto":
        effective_provider = provider
        effective_key = api_key or openrouter_api_key
    elif api_key:
        from oss_maintainer_toolkit.gatekeeper.providers import detect_provider_from_key
        detected = detect_provider_from_key(api_key)
        effective_provider = detected or (provider if provider else gatekeeper_settings.llm_provider)
        effective_key = api_key
        if effective_provider == "auto":
            effective_provider, effective_key = resolve_provider_and_key(gatekeeper_settings)
    else:
        resolved_provider, resolved_key = resolve_provider_and_key(gatekeeper_settings)
        effective_provider = provider if (provider and provider != "auto") else resolved_provider
        effective_key = resolved_key
        if effective_provider == "openrouter" and openrouter_api_key:
            effective_key = openrouter_api_key

    return effective_provider, effective_key


async def _dispatch_to_provider(
    prompt: str,
    system_prompt: str,
    provider: str = "",
    api_key: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "",
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Dispatch a prompt to the resolved LLM provider and return a VisionAlignmentResult.

    This is the shared provider dispatch used by both PR and issue vision alignment.
    """
    effective_provider, effective_key = _resolve_effective_provider(
        provider, api_key, openrouter_api_key,
    )

    if effective_provider == "claude_cli":
        return await _run_claude_cli(
            prompt,
            claude_command=claude_command,
            timeout_seconds=timeout_seconds,
        )

    if effective_provider in ("openrouter", "openai", "generic"):
        model, base_url, default_timeout = _get_provider_config(effective_provider, effective_key)
        if not effective_key:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"No API key for {effective_provider}. Set AUDITOR_GK_LLM_API_KEY or the provider-specific key."],
            )
        try:
            data = await call_openai_compatible(
                prompt=prompt,
                system_prompt=system_prompt,
                json_schema=SCORECARD_SCHEMA,
                api_key=effective_key,
                model=openrouter_model or model,
                base_url=base_url,
                timeout_seconds=timeout_seconds or default_timeout,
            )
            return _parse_response(data)
        except ProviderError as e:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"{effective_provider}: {e}"],
            )

    if effective_provider == "anthropic":
        model, base_url, default_timeout = _get_provider_config("anthropic", effective_key)
        if not effective_key:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=["No API key for Anthropic. Set AUDITOR_GK_LLM_API_KEY or AUDITOR_GK_ANTHROPIC_API_KEY."],
            )
        prompt_with_schema = prompt + _build_schema_instruction()
        try:
            data = await call_anthropic(
                prompt=prompt_with_schema,
                system_prompt=system_prompt,
                api_key=effective_key,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds or default_timeout,
            )
            return _parse_response(data)
        except ProviderError as e:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"anthropic: {e}"],
            )

    if effective_provider == "gemini":
        model, base_url, default_timeout = _get_provider_config("gemini", effective_key)
        if not effective_key:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=["No API key for Gemini. Set AUDITOR_GK_LLM_API_KEY or AUDITOR_GK_GEMINI_API_KEY."],
            )
        prompt_with_schema = prompt + _build_schema_instruction()
        try:
            data = await call_gemini(
                prompt=prompt_with_schema,
                system_prompt=system_prompt,
                api_key=effective_key,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds or default_timeout,
            )
            return _parse_response(data)
        except ProviderError as e:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"gemini: {e}"],
            )

    return VisionAlignmentResult(
        outcome=TierOutcome.ERROR,
        concerns=[f"Unknown LLM provider '{effective_provider}'. Use 'auto', 'openrouter', 'openai', 'anthropic', 'gemini', 'generic', or 'claude_cli'."],
    )


# --- PR vision alignment (unified entry point) ---

async def run_vision_alignment(
    pr: PRMetadata,
    vision: VisionDocument,
    provider: str = "",
    api_key: str = "",
    # OpenRouter overrides (backward compat)
    openrouter_api_key: str = "",
    openrouter_model: str = "",
    # Claude CLI overrides
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment assessment for a PR using the configured LLM provider.

    Provider selection (in order):
    1. Explicit `provider` param
    2. Auto-detect from `api_key` prefix
    3. Config-level auto-detection via AUDITOR_GK_LLM_PROVIDER / keys

    Supported providers: auto, openrouter, openai, anthropic, gemini, generic, claude_cli.
    """
    prompt = _build_prompt(pr, vision)
    return await _dispatch_to_provider(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        api_key=api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        claude_command=claude_command,
        timeout_seconds=timeout_seconds,
    )


# --- Issue vision alignment ---

ISSUE_SYSTEM_PROMPT = (
    "You are a project maintainer triaging GitHub issues against a project's vision document. "
    "Return ONLY valid JSON matching the provided schema. No markdown, no extra keys, no extra text."
)


def _build_issue_prompt(issue: IssueMetadata, vision: VisionDocument) -> str:
    """Build the user prompt for issue vision alignment assessment."""
    principles_text = "\n".join(
        f"- {p.name}: {p.description}" for p in vision.principles
    )
    anti_patterns_text = "\n".join(f"- {ap}" for ap in vision.anti_patterns)
    focus_areas_text = "\n".join(f"- {fa}" for fa in vision.focus_areas)

    labels_text = ", ".join(issue.labels) if issue.labels else "(none)"

    return f"""Assess this GitHub issue for alignment with the project's vision document.

## Project: {vision.project}

### Vision Principles
{principles_text}

### Anti-Patterns to Watch For
{anti_patterns_text}

### Focus Areas
{focus_areas_text}

## Issue #{issue.number}: {issue.title}

**Author:** {issue.author.login}
**State:** {issue.state}
**Labels:** {labels_text}
**Comments:** {issue.comment_count}
**Description:** {issue.body[:2000] if issue.body else '(no description)'}

Evaluate alignment_score from 0.0 (off-topic / violates vision) to 1.0 (perfect fit). List any violated principle names exactly as shown above."""


async def run_issue_vision_alignment(
    issue: IssueMetadata,
    vision: VisionDocument,
    provider: str = "",
    api_key: str = "",
    openrouter_api_key: str = "",
    openrouter_model: str = "",
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment assessment for an issue using the configured LLM provider."""
    prompt = _build_issue_prompt(issue, vision)
    return await _dispatch_to_provider(
        prompt=prompt,
        system_prompt=ISSUE_SYSTEM_PROMPT,
        provider=provider,
        api_key=api_key,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
        claude_command=claude_command,
        timeout_seconds=timeout_seconds,
    )
