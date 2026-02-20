"""Review Routing: suggest reviewers based on file ownership and history.

Analyzes CODEOWNERS, past review patterns, and file paths to rank
potential reviewers for a PR. Tier 2 only (heuristic rules), no LLM, $0 cost.
"""

from __future__ import annotations

import fnmatch
from collections import Counter

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.models import (
    CodeOwnerRule,
    PRMetadata,
    ReviewerSuggestion,
    ReviewRoutingReport,
)


def parse_codeowners(content: str) -> list[CodeOwnerRule]:
    """Parse a CODEOWNERS file into a list of rules.

    Handles standard CODEOWNERS format: pattern followed by @owner(s).
    Lines starting with # are comments. Empty lines are skipped.
    """
    rules: list[CodeOwnerRule] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pattern = parts[0]
        owners = [p.lstrip("@") for p in parts[1:] if p.startswith("@")]
        if owners:
            rules.append(CodeOwnerRule(pattern=pattern, owners=owners))
    return rules


def _match_codeowners(
    changed_files: list[str],
    rules: list[CodeOwnerRule],
) -> dict[str, list[str]]:
    """Match changed files against CODEOWNERS rules.

    Returns dict mapping username -> list of reasons (matched patterns).
    Later rules override earlier ones (CODEOWNERS convention: last match wins).
    """
    owner_reasons: dict[str, list[str]] = {}

    for filepath in changed_files:
        matched_owners: list[str] = []
        matched_pattern = ""
        # Last matching rule wins
        for rule in rules:
            if fnmatch.fnmatch(filepath, rule.pattern) or fnmatch.fnmatch(filepath, f"**/{rule.pattern}"):
                matched_owners = rule.owners
                matched_pattern = rule.pattern
        for owner in matched_owners:
            reason = f"CODEOWNERS: {matched_pattern}"
            owner_reasons.setdefault(owner, [])
            if reason not in owner_reasons[owner]:
                owner_reasons[owner].append(reason)

    return owner_reasons


def _score_past_reviewers(
    changed_files: list[str],
    recent_prs: list[PRMetadata],
    reviews_by_pr: dict[int, list[str]],
) -> dict[str, list[str]]:
    """Score reviewers based on past review patterns on similar files.

    Returns dict mapping username -> list of reasons.
    """
    changed_set = set(changed_files)
    reviewer_reasons: dict[str, list[str]] = {}
    reviewer_counts: Counter[str] = Counter()

    for pr in recent_prs:
        pr_files = {f.filename for f in pr.files}
        overlap = changed_set & pr_files
        if not overlap:
            continue
        reviewers = reviews_by_pr.get(pr.number, [])
        for reviewer in reviewers:
            reviewer_counts[reviewer] += 1

    for reviewer, count in reviewer_counts.items():
        reviewer_reasons[reviewer] = [f"Reviewed {count} recent PR(s) touching similar files"]

    return reviewer_reasons


def suggest_reviewers(
    pr: PRMetadata,
    codeowners_rules: list[CodeOwnerRule] | None = None,
    recent_prs: list[PRMetadata] | None = None,
    reviews_by_pr: dict[int, list[str]] | None = None,
    max_suggestions: int = 0,
) -> ReviewRoutingReport:
    """Suggest reviewers for a PR.

    Combines CODEOWNERS matches and past review patterns.
    Excludes the PR author from suggestions.

    Args:
        pr: The PR needing reviewers.
        codeowners_rules: Parsed CODEOWNERS rules (None = no CODEOWNERS).
        recent_prs: Recently merged PRs for review history analysis.
        reviews_by_pr: Mapping of PR number -> list of reviewer usernames.
        max_suggestions: Max reviewers to suggest (0 = config default).

    Returns:
        ReviewRoutingReport with ranked reviewer suggestions.
    """
    if max_suggestions <= 0:
        max_suggestions = gatekeeper_settings.review_max_suggestions

    changed_files = [f.filename for f in pr.files]

    report = ReviewRoutingReport(
        owner=pr.owner,
        repo=pr.repo,
        pr_number=pr.number,
        pr_title=pr.title,
        changed_files=changed_files,
        codeowners_found=codeowners_rules is not None and len(codeowners_rules) > 0,
        recent_reviewers_checked=len(recent_prs) if recent_prs else 0,
    )

    # Collect scores from both sources
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    # CODEOWNERS (higher weight: 2.0 per match)
    if codeowners_rules:
        co_reasons = _match_codeowners(changed_files, codeowners_rules)
        for user, user_reasons in co_reasons.items():
            scores[user] = scores.get(user, 0.0) + 2.0 * len(user_reasons)
            reasons.setdefault(user, []).extend(user_reasons)

    # Past reviews (1.0 per reviewed PR)
    if recent_prs and reviews_by_pr:
        pr_reasons = _score_past_reviewers(changed_files, recent_prs, reviews_by_pr)
        for user, user_reasons in pr_reasons.items():
            scores[user] = scores.get(user, 0.0) + 1.0
            reasons.setdefault(user, []).extend(user_reasons)

    # Exclude PR author
    author = pr.author.login
    scores.pop(author, None)
    reasons.pop(author, None)

    # Normalize scores to 0-1 range
    max_score = max(scores.values()) if scores else 1.0
    suggestions = []
    for user, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        normalized = score / max_score if max_score > 0 else 0.0
        suggestions.append(ReviewerSuggestion(
            username=user,
            score=round(normalized, 4),
            reasons=reasons.get(user, []),
        ))

    report.suggestions = suggestions[:max_suggestions]
    return report
