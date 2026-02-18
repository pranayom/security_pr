"""Tier 3: Vision alignment assessment via OpenRouter (primary) or claude --print (fallback)."""

from __future__ import annotations

import asyncio
import json

import httpx
import yaml

from mcp_ai_auditor.gatekeeper.config import gatekeeper_settings
from mcp_ai_auditor.gatekeeper.models import (
    PRMetadata,
    TierOutcome,
    VisionAlignmentResult,
    VisionDocument,
    VisionPrinciple,
)

# JSON Schema for structured outputs — enforced by OpenRouter's strict mode
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


def load_vision_document(path: str) -> VisionDocument:
    """Load a YAML vision document from disk."""
    with open(path) as f:
        data = yaml.safe_load(f)

    principles = [
        VisionPrinciple(name=p["name"], description=p["description"])
        for p in data.get("principles", [])
    ]

    return VisionDocument(
        project=data.get("project", ""),
        principles=principles,
        anti_patterns=data.get("anti_patterns", []),
        focus_areas=data.get("focus_areas", []),
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


# --- OpenRouter provider ---

async def _run_openrouter(
    prompt: str,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment via OpenRouter's chat completions API with structured outputs."""
    api_key = api_key or gatekeeper_settings.openrouter_api_key
    model = model or gatekeeper_settings.openrouter_model
    base_url = base_url or gatekeeper_settings.openrouter_base_url
    timeout = timeout_seconds or gatekeeper_settings.openrouter_timeout_seconds

    if not api_key:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=["No OpenRouter API key. Set AUDITOR_GK_OPENROUTER_API_KEY or switch to claude_cli provider."],
        )

    payload = {
        "model": model,
        "temperature": 0,
        "seed": 42,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": SCORECARD_SCHEMA,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if resp.status_code != 200:
            return VisionAlignmentResult(
                outcome=TierOutcome.ERROR,
                concerns=[f"OpenRouter API returned {resp.status_code}: {resp.text[:500]}"],
            )

        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        data = json.loads(content)
        return _parse_response(data)

    except httpx.TimeoutException:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"OpenRouter request timed out after {timeout}s"],
        )
    except (KeyError, IndexError) as e:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"Unexpected OpenRouter response structure: {e}"],
        )
    except json.JSONDecodeError as e:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"Failed to parse OpenRouter response as JSON: {e}"],
        )


# --- Claude CLI provider ---

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


# --- Unified entry point ---

async def run_vision_alignment(
    pr: PRMetadata,
    vision: VisionDocument,
    provider: str = "",
    # OpenRouter overrides
    openrouter_api_key: str = "",
    openrouter_model: str = "",
    # Claude CLI overrides
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment assessment using the configured LLM provider.

    Provider selection: openrouter (default) or claude_cli.
    Override via provider param or AUDITOR_GK_LLM_PROVIDER env var.
    """
    provider = provider or gatekeeper_settings.llm_provider
    prompt = _build_prompt(pr, vision)

    if provider == "openrouter":
        return await _run_openrouter(
            prompt,
            api_key=openrouter_api_key,
            timeout_seconds=timeout_seconds,
        )
    elif provider == "claude_cli":
        return await _run_claude_cli(
            prompt,
            claude_command=claude_command,
            timeout_seconds=timeout_seconds,
        )
    else:
        return VisionAlignmentResult(
            outcome=TierOutcome.ERROR,
            concerns=[f"Unknown LLM provider '{provider}'. Use 'openrouter' or 'claude_cli'."],
        )
