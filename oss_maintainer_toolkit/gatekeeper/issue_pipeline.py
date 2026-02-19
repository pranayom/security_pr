"""Three-tier gated pipeline orchestrator for issue triage."""

from __future__ import annotations

from oss_maintainer_toolkit.gatekeeper.models import (
    DedupResult,
    DimensionScore,
    HeuristicsResult,
    IssueMetadata,
    IssueScorecard,
    SuspicionFlag,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)
from oss_maintainer_toolkit.gatekeeper.issue_dedup import check_issue_duplicates
from oss_maintainer_toolkit.gatekeeper.issue_heuristics import run_issue_heuristics
from oss_maintainer_toolkit.gatekeeper.vision import load_vision_document, run_issue_vision_alignment


async def run_issue_pipeline(
    issue: IssueMetadata,
    issue_embedding: list[float] | None = None,
    existing_issues: list[IssueMetadata] | None = None,
    existing_embeddings: list[list[float]] | None = None,
    recent_issues: list[IssueMetadata] | None = None,
    vision_document_path: str = "",
    enable_tier3: bool = True,
    llm_provider: str = "",
    llm_api_key: str = "",
) -> IssueScorecard:
    """Run the three-tier gated pipeline for issue triage.

    Tier 1 (Dedup) -> if GATED -> RECOMMEND_CLOSE, stop
    Tier 2 (Heuristics) -> if GATED -> REVIEW_REQUIRED, stop
    Tier 3 (Vision) -> scoring-based verdict

    Args:
        issue: The issue to assess.
        issue_embedding: Pre-computed embedding (required for Tier 1).
        existing_issues: Other issues for dedup comparison.
        existing_embeddings: Embeddings for existing issues.
        recent_issues: Recent issues for temporal clustering.
        vision_document_path: Path to YAML vision document.
        enable_tier3: Whether to run Tier 3.
        llm_provider: LLM provider for Tier 3. Defaults to config (auto-detect).
        llm_api_key: Unified API key (auto-detects provider from prefix).
    """
    all_flags: list[SuspicionFlag] = []
    dimensions: list[DimensionScore] = []

    # Load vision document early
    vision = None
    if vision_document_path:
        try:
            vision = load_vision_document(vision_document_path)
        except FileNotFoundError:
            print(f"Warning: Vision document not found at '{vision_document_path}' — skipping")

    # --- Tier 1: Dedup ---
    dedup_result: DedupResult
    if issue_embedding is not None:
        dedup_result = check_issue_duplicates(
            issue, issue_embedding,
            existing_issues or [], existing_embeddings or [],
        )
    else:
        dedup_result = DedupResult(outcome=TierOutcome.SKIPPED)

    dimensions.append(DimensionScore(
        dimension="Issue Dedup",
        score=0.0 if dedup_result.is_duplicate else 1.0,
        summary=f"Duplicate of Issue#{dedup_result.duplicate_of} (similarity: {dedup_result.max_similarity:.2f})"
        if dedup_result.is_duplicate
        else "No duplicates found",
    ))

    if dedup_result.outcome == TierOutcome.GATED:
        return IssueScorecard(
            owner=issue.owner,
            repo=issue.repo,
            issue_number=issue.number,
            verdict=Verdict.RECOMMEND_CLOSE,
            confidence=dedup_result.max_similarity,
            dimensions=dimensions,
            dedup_result=dedup_result,
            flags=all_flags,
            summary=f"Issue is a duplicate of Issue#{dedup_result.duplicate_of} "
                    f"(similarity: {dedup_result.max_similarity:.2f}). Recommend closing.",
        )

    # --- Tier 2: Heuristics ---
    heuristics_result = run_issue_heuristics(issue, recent_issues)
    all_flags.extend(heuristics_result.flags)

    dimensions.append(DimensionScore(
        dimension="Issue Quality",
        score=1.0 - heuristics_result.suspicion_score,
        flags=heuristics_result.flags,
        summary=f"Suspicion score: {heuristics_result.suspicion_score:.2f} "
                f"({len(heuristics_result.flags)} flag(s))",
    ))

    if heuristics_result.outcome == TierOutcome.GATED:
        return IssueScorecard(
            owner=issue.owner,
            repo=issue.repo,
            issue_number=issue.number,
            verdict=Verdict.REVIEW_REQUIRED,
            confidence=heuristics_result.suspicion_score,
            dimensions=dimensions,
            dedup_result=dedup_result,
            heuristics_result=heuristics_result,
            flags=all_flags,
            summary=f"Suspicion score {heuristics_result.suspicion_score:.2f} exceeds threshold. "
                    f"Flagged: {', '.join(f.title for f in heuristics_result.flags)}.",
        )

    # --- Tier 3: Vision Alignment ---
    vision_result: VisionAlignmentResult | None = None
    if enable_tier3 and vision is not None:
        vision_result = await run_issue_vision_alignment(
            issue, vision, provider=llm_provider, api_key=llm_api_key,
        )

        dimensions.append(DimensionScore(
            dimension="Vision Alignment",
            score=vision_result.alignment_score,
            summary=f"Alignment: {vision_result.alignment_score:.2f}",
        ))

        if vision_result.outcome == TierOutcome.ERROR:
            return IssueScorecard(
                owner=issue.owner,
                repo=issue.repo,
                issue_number=issue.number,
                verdict=Verdict.REVIEW_REQUIRED,
                confidence=0.5,
                dimensions=dimensions,
                dedup_result=dedup_result,
                heuristics_result=heuristics_result,
                vision_result=vision_result,
                flags=all_flags,
                summary=f"Vision assessment errored: {vision_result.concerns[0] if vision_result.concerns else 'unknown error'}",
            )

        # Vision-based verdict logic
        if vision_result.alignment_score < 0.4:
            return IssueScorecard(
                owner=issue.owner,
                repo=issue.repo,
                issue_number=issue.number,
                verdict=Verdict.REVIEW_REQUIRED,
                confidence=1.0 - vision_result.alignment_score,
                dimensions=dimensions,
                dedup_result=dedup_result,
                heuristics_result=heuristics_result,
                vision_result=vision_result,
                flags=all_flags,
                summary=f"Low vision alignment ({vision_result.alignment_score:.2f}). "
                        f"Violated: {', '.join(vision_result.violated_principles) or 'none'}.",
            )

        if all_flags and vision_result.alignment_score < 0.6:
            return IssueScorecard(
                owner=issue.owner,
                repo=issue.repo,
                issue_number=issue.number,
                verdict=Verdict.REVIEW_REQUIRED,
                confidence=0.6,
                dimensions=dimensions,
                dedup_result=dedup_result,
                heuristics_result=heuristics_result,
                vision_result=vision_result,
                flags=all_flags,
                summary=f"Moderate alignment ({vision_result.alignment_score:.2f}) combined with "
                        f"{len(all_flags)} suspicion flag(s) warrants review.",
            )

    # --- All clear -> FAST_TRACK ---
    confidence = 0.8
    if vision_result and vision_result.alignment_score > 0:
        confidence = vision_result.alignment_score

    return IssueScorecard(
        owner=issue.owner,
        repo=issue.repo,
        issue_number=issue.number,
        verdict=Verdict.FAST_TRACK,
        confidence=confidence,
        dimensions=dimensions,
        dedup_result=dedup_result,
        heuristics_result=heuristics_result,
        vision_result=vision_result,
        flags=all_flags,
        summary="Issue passed all tiers. Safe to fast-track.",
    )
