"""Generate a Vision Document for a GitHub repository using LLM analysis.

Fetches repo context (README, CONTRIBUTING.md, merged PRs, rejected PRs),
passes to Tier 3 LLM provider, and outputs a standard vision.yaml.
"""

from __future__ import annotations

import yaml

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import (
    LabelDefinition,
    VisionDocument,
    VisionPrinciple,
)
from oss_maintainer_toolkit.gatekeeper.providers import (
    ProviderError,
    call_anthropic,
    call_gemini,
    call_openai_compatible,
)
from oss_maintainer_toolkit.gatekeeper.vision import (
    _get_provider_config,
    _resolve_effective_provider,
)

# JSON Schema for structured outputs — enforced by OpenAI-compatible providers
VISION_DOC_SCHEMA = {
    "name": "vision_document",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "project",
            "principles",
            "anti_patterns",
            "focus_areas",
            "label_taxonomy",
        ],
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name",
            },
            "principles": {
                "type": "array",
                "description": "Core governance principles (5-10 items)",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "description"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short principle name (2-5 words)",
                        },
                        "description": {
                            "type": "string",
                            "description": "Detailed explanation of the principle",
                        },
                    },
                },
            },
            "anti_patterns": {
                "type": "array",
                "description": "Contribution anti-patterns to reject (5-15 items)",
                "items": {"type": "string"},
            },
            "focus_areas": {
                "type": "array",
                "description": "Security-sensitive file paths/directories to watch (path substrings)",
                "items": {"type": "string"},
            },
            "label_taxonomy": {
                "type": "array",
                "description": "Label categories for PR/issue classification (5-15 items)",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "description", "keywords"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Label name (e.g. 'bug', 'feature', 'docs')",
                        },
                        "description": {
                            "type": "string",
                            "description": "What this label means",
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords that indicate this label applies",
                        },
                    },
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You are an expert open-source maintainer generating a Vision Document for a GitHub project. "
    "A Vision Document captures the project's unwritten governance rules: what it is, what it is NOT, "
    "architectural principles, and contribution anti-patterns. "
    "Return ONLY valid JSON matching the provided schema. No markdown, no extra keys, no extra text."
)


def _build_schema_instruction() -> str:
    """Build a text instruction describing the required JSON schema.

    Used for providers that don't support structured output mode (Anthropic, Gemini).
    """
    return (
        "\n\nYou MUST respond with ONLY valid JSON matching this exact schema:\n"
        "{\n"
        '  "project": "<string>",\n'
        '  "principles": [{"name": "<string>", "description": "<string>"}, ...],\n'
        '  "anti_patterns": ["<string>", ...],\n'
        '  "focus_areas": ["<string>", ...],\n'
        '  "label_taxonomy": [{"name": "<string>", "description": "<string>", '
        '"keywords": ["<string>", ...]}, ...]\n'
        "}\n"
        "No other keys, no markdown fences, no extra text."
    )


async def fetch_repo_context(
    owner: str,
    repo: str,
    client,
    max_merged: int = 10,
    max_rejected: int = 10,
) -> dict:
    """Fetch repo context for vision document generation.

    Returns a dict with keys: readme, contributing, merged_prs, rejected_prs.
    """
    readme = await client.get_file_content(owner, repo, "README.md")
    if readme is None:
        readme = await client.get_file_content(owner, repo, "readme.md")

    contributing = await client.get_file_content(owner, repo, "CONTRIBUTING.md")
    if contributing is None:
        contributing = await client.get_file_content(owner, repo, "contributing.md")

    merged_prs = await client.list_recently_merged_prs(owner, repo, since_days=90)
    merged_prs = merged_prs[:max_merged]

    rejected_prs = await client.list_closed_unmerged_prs(owner, repo, max_results=max_rejected)

    # Fetch diffs for merged and rejected PRs
    merged_with_diffs = []
    for pr in merged_prs:
        diff = await client.get_pr_diff(owner, repo, pr["number"])
        merged_with_diffs.append({
            "number": pr["number"],
            "title": pr.get("title", ""),
            "body": (pr.get("body") or "")[:500],
            "diff_summary": diff[:2000] if diff else "(no diff)",
        })

    rejected_with_diffs = []
    for pr in rejected_prs:
        diff = await client.get_pr_diff(owner, repo, pr["number"])
        rejected_with_diffs.append({
            "number": pr["number"],
            "title": pr.get("title", ""),
            "body": (pr.get("body") or "")[:500],
            "diff_summary": diff[:2000] if diff else "(no diff)",
        })

    return {
        "readme": (readme or "")[:5000],
        "contributing": (contributing or "")[:3000],
        "merged_prs": merged_with_diffs,
        "rejected_prs": rejected_with_diffs,
    }


def build_generation_prompt(owner: str, repo: str, context: dict) -> str:
    """Build the LLM prompt for vision document generation."""
    merged_text = ""
    for pr in context["merged_prs"]:
        merged_text += f"\n### Merged PR #{pr['number']}: {pr['title']}\n"
        merged_text += f"Description: {pr['body']}\n"
        merged_text += f"Diff:\n```\n{pr['diff_summary']}\n```\n"

    rejected_text = ""
    for pr in context["rejected_prs"]:
        rejected_text += f"\n### Rejected PR #{pr['number']}: {pr['title']}\n"
        rejected_text += f"Description: {pr['body']}\n"
        rejected_text += f"Diff:\n```\n{pr['diff_summary']}\n```\n"

    return f"""Analyze this GitHub repository and generate a Vision Document that captures its governance rules.

## Repository: {owner}/{repo}

### README
{context['readme'] or '(no README found)'}

### CONTRIBUTING.md
{context['contributing'] or '(no CONTRIBUTING.md found)'}

## Recently Merged PRs (what the project ACCEPTS)
{merged_text or '(no recent merged PRs found)'}

## Recently Rejected PRs (closed without merge — what the project REJECTS)
{rejected_text or '(no recent rejected PRs found)'}

## Instructions

Based on the above evidence, generate a Vision Document with:

1. **principles** (5-10): Core governance rules. Deduce from the README, contributing guide, and patterns in merged/rejected PRs. Each principle should have a concise name and a detailed description.

2. **anti_patterns** (5-15): Specific things contributors should NOT do. Infer from rejected PRs, contributing guidelines, and README warnings.

3. **focus_areas** (5-15): Security-sensitive file paths or directories that deserve extra review. Use path substrings like "auth/", "config/", ".github/workflows". Infer from the codebase structure visible in PR diffs.

4. **label_taxonomy** (5-15): Labels for classifying PRs and issues. Include standard categories (bug, feature, docs, etc.) plus project-specific ones inferred from the PRs.

Be specific to THIS project. Do not generate generic platitudes. Every principle should be evidenced by the repository data above."""


def _parse_vision_response(data: dict) -> VisionDocument:
    """Parse a JSON response dict into a VisionDocument."""
    principles = [
        VisionPrinciple(name=p["name"], description=p["description"])
        for p in data.get("principles", [])
    ]
    label_taxonomy = [
        LabelDefinition(
            name=lb["name"],
            description=lb.get("description", ""),
            keywords=lb.get("keywords", []),
            source="generated",
        )
        for lb in data.get("label_taxonomy", [])
    ]
    return VisionDocument(
        project=data.get("project", ""),
        principles=principles,
        anti_patterns=data.get("anti_patterns", []),
        focus_areas=data.get("focus_areas", []),
        label_taxonomy=label_taxonomy,
    )


def vision_document_to_yaml(doc: VisionDocument, owner: str, repo: str) -> str:
    """Serialize a VisionDocument to YAML string."""
    data = {
        "project": doc.project,
        "principles": [
            {"name": p.name, "description": p.description}
            for p in doc.principles
        ],
        "anti_patterns": doc.anti_patterns,
        "focus_areas": doc.focus_areas,
    }
    if doc.label_taxonomy:
        data["label_taxonomy"] = [
            {
                "name": lb.name,
                "description": lb.description,
                "keywords": lb.keywords,
            }
            for lb in doc.label_taxonomy
        ]

    header = (
        f"# Vision Document: {doc.project}\n"
        f"# Generated by OSS Maintainer Toolkit\n"
        f"# Repo: {owner}/{repo}\n"
        f"# Status: draft — maintainer review required before enforcement\n"
        f"\n"
    )
    return header + yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


async def _dispatch_for_generation(
    prompt: str,
    provider: str = "",
    api_key: str = "",
) -> dict:
    """Dispatch a vision generation prompt to the resolved LLM provider.

    Returns the raw JSON dict from the provider.
    Raises ProviderError on failure.
    """
    effective_provider, effective_key = _resolve_effective_provider(provider, api_key)

    if effective_provider == "claude_cli":
        raise ProviderError(
            "Vision generation requires an LLM API provider. "
            "claude_cli is not supported for generation. "
            "Set AUDITOR_GK_LLM_API_KEY or AUDITOR_GK_OPENROUTER_API_KEY."
        )

    if effective_provider in ("openrouter", "openai", "generic"):
        model, base_url, default_timeout = _get_provider_config(effective_provider, effective_key)
        if not effective_key:
            raise ProviderError(
                f"No API key for {effective_provider}. "
                "Set AUDITOR_GK_LLM_API_KEY or the provider-specific key."
            )
        return await call_openai_compatible(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            json_schema=VISION_DOC_SCHEMA,
            api_key=effective_key,
            model=model,
            base_url=base_url,
            timeout_seconds=default_timeout,
        )

    if effective_provider == "anthropic":
        model, base_url, default_timeout = _get_provider_config("anthropic", effective_key)
        if not effective_key:
            raise ProviderError(
                "No API key for Anthropic. "
                "Set AUDITOR_GK_LLM_API_KEY or AUDITOR_GK_ANTHROPIC_API_KEY."
            )
        return await call_anthropic(
            prompt=prompt + _build_schema_instruction(),
            system_prompt=SYSTEM_PROMPT,
            api_key=effective_key,
            model=model,
            base_url=base_url,
            timeout_seconds=default_timeout,
        )

    if effective_provider == "gemini":
        model, base_url, default_timeout = _get_provider_config("gemini", effective_key)
        if not effective_key:
            raise ProviderError(
                "No API key for Gemini. "
                "Set AUDITOR_GK_LLM_API_KEY or AUDITOR_GK_GEMINI_API_KEY."
            )
        return await call_gemini(
            prompt=prompt + _build_schema_instruction(),
            system_prompt=SYSTEM_PROMPT,
            api_key=effective_key,
            model=model,
            base_url=base_url,
            timeout_seconds=default_timeout,
        )

    raise ProviderError(
        f"Unknown LLM provider '{effective_provider}'. "
        "Use 'openrouter', 'openai', 'anthropic', 'gemini', or 'generic'."
    )


async def generate_vision_document(
    owner: str,
    repo: str,
    provider: str = "",
    api_key: str = "",
    max_merged: int = 10,
    max_rejected: int = 10,
) -> VisionDocument:
    """Generate a Vision Document for a GitHub repository.

    Fetches repo context (README, CONTRIBUTING.md, merged/rejected PRs),
    passes to a Tier 3 LLM provider, and returns a VisionDocument.

    Requires an LLM API key (OpenRouter, OpenAI, Anthropic, or Gemini).
    """
    from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient

    async with GitHubClient() as client:
        context = await fetch_repo_context(
            owner, repo, client,
            max_merged=max_merged,
            max_rejected=max_rejected,
        )

    prompt = build_generation_prompt(owner, repo, context)
    data = await _dispatch_for_generation(prompt, provider=provider, api_key=api_key)
    return _parse_vision_response(data)
