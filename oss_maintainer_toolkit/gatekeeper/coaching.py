"""PR coaching bot — actionable remediation advice for triage flags."""

from __future__ import annotations

from oss_maintainer_toolkit.gatekeeper.models import SuspicionFlag, VisionAlignmentResult


# Maps rule_id -> coaching advice template.
# {evidence} is substituted from the flag's evidence field.
COACHING_ADVICE: dict[str, str] = {
    "new_account": (
        "Consider starting with smaller PRs (docs, tests) to build trust with maintainers."
    ),
    "first_contribution": (
        "Welcome! Check CONTRIBUTING.md for project-specific guidelines before your next PR."
    ),
    "sensitive_paths": (
        "Changes to security-sensitive paths ({evidence}) require extra justification. "
        "Add a rationale in the PR description explaining why these files need modification."
    ),
    "low_test_ratio": (
        "This PR adds code without proportional tests ({evidence}). "
        "Add tests covering the new behavior to help get this fast-tracked."
    ),
    "unjustified_deps": (
        "Dependency changes ({evidence}) need justification. "
        "Explain in the PR body: version bump, CVE fix, or new feature requirement."
    ),
    "large_diff_hiding": (
        "Large PRs risk burying important changes ({evidence}). "
        "Consider splitting into smaller, focused PRs for easier review."
    ),
    "temporal_clustering": (
        "This PR was submitted alongside several others from new accounts. "
        "If this is legitimate, introduce yourself in the project's community channels."
    ),
    # Issue-specific rules
    "vague_description": (
        "Add more detail to the issue body: what you expected, what happened, "
        "and steps to reproduce. This helps maintainers triage faster."
    ),
    "missing_reproduction": (
        "Bug reports without reproduction steps are hard to act on. "
        "Add step-by-step instructions and include code snippets or error output."
    ),
    "short_title": (
        "Use a descriptive title that summarizes the issue. "
        "Good: 'Login fails with 403 after password reset'. Bad: 'Bug'."
    ),
    "all_caps_title": (
        "Rewrite the title in normal case. ALL CAPS titles are harder to scan "
        "and may be deprioritized by maintainers."
    ),
}


def build_flag_coaching(flags: list[SuspicionFlag]) -> list[str]:
    """Generate coaching bullets for each flag that has advice.

    Returns markdown lines (one bullet per flag with coaching).
    """
    lines: list[str] = []
    for flag in flags:
        template = COACHING_ADVICE.get(flag.rule_id)
        if not template:
            continue
        advice = template.replace("{evidence}", flag.evidence) if flag.evidence else template.replace("{evidence}", "")
        lines.append(f"- **{flag.title}**: {advice}")
    return lines


def build_vision_coaching(vision_result: VisionAlignmentResult) -> list[str]:
    """Generate coaching bullets from violated vision principles and concerns.

    Returns markdown lines with improvement suggestions.
    """
    lines: list[str] = []
    for principle in vision_result.violated_principles:
        lines.append(f"- **{principle}** was flagged as violated. "
                     "Review the project's vision document and align your contribution accordingly.")
    for concern in vision_result.concerns:
        lines.append(f"- The reviewer noted: \"{concern}\"")
    return lines
