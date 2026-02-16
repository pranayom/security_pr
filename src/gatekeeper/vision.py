"""Tier 3: Vision alignment assessment via claude --print."""

from __future__ import annotations

import asyncio
import json

import yaml

from src.gatekeeper.config import gatekeeper_settings
from src.gatekeeper.models import (
    PRMetadata,
    TierOutcome,
    VisionAlignmentResult,
    VisionDocument,
    VisionPrinciple,
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
    """Build the structured prompt for claude --print."""
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

    return f"""You are assessing a pull request for alignment with the project's vision document.

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

## Instructions
Assess how well this PR aligns with the project vision. Respond with ONLY valid JSON matching this schema:
{{
  "alignment_score": <float 0.0-1.0>,
  "violated_principles": [<list of principle names that are violated>],
  "strengths": [<list of strengths>],
  "concerns": [<list of concerns>]
}}
"""


async def run_vision_alignment(
    pr: PRMetadata,
    vision: VisionDocument,
    claude_command: str = "",
    timeout_seconds: int = 0,
) -> VisionAlignmentResult:
    """Run vision alignment assessment via claude --print.

    Shells out to the claude CLI with --print and --output-format json flags.
    """
    cmd = claude_command or gatekeeper_settings.claude_command
    timeout = timeout_seconds or gatekeeper_settings.claude_timeout_seconds
    prompt = _build_prompt(pr, vision)

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

        alignment_score = float(data.get("alignment_score", 0.0))
        outcome = TierOutcome.PASS if alignment_score >= 0.4 else TierOutcome.GATED

        return VisionAlignmentResult(
            outcome=outcome,
            alignment_score=alignment_score,
            violated_principles=data.get("violated_principles", []),
            strengths=data.get("strengths", []),
            concerns=data.get("concerns", []),
        )

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
