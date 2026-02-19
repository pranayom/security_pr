"""Three-tier gated pipeline orchestrator for PR risk assessment."""

from __future__ import annotations

from mcp_ai_auditor.gatekeeper.models import (
    AssessmentScorecard,
    DedupResult,
    DimensionScore,
    HeuristicsResult,
    PRMetadata,
    SuspicionFlag,
    TierOutcome,
    Verdict,
    VisionAlignmentResult,
)
from mcp_ai_auditor.gatekeeper.dedup import check_duplicates
from mcp_ai_auditor.gatekeeper.heuristics import run_heuristics
from mcp_ai_auditor.gatekeeper.vision import load_vision_document, run_vision_alignment


async def run_pipeline(
    pr: PRMetadata,
    pr_embedding: list[float] | None = None,
    existing_prs: list[PRMetadata] | None = None,
    existing_embeddings: list[list[float]] | None = None,
    recent_prs: list[PRMetadata] | None = None,
    vision_document_path: str = "",
    enable_tier3: bool = True,
    llm_provider: str = "",
    llm_api_key: str = "",
) -> AssessmentScorecard:
    """Run the three-tier gated pipeline.

    Tier 1 (Dedup) → if GATED → RECOMMEND_CLOSE, stop
    Tier 2 (Heuristics) → if GATED → REVIEW_REQUIRED, stop
    Tier 3 (Vision) → scoring-based verdict

    Args:
        pr: The PR to assess.
        pr_embedding: Pre-computed embedding (required for Tier 1).
        existing_prs: Other PRs for dedup comparison.
        existing_embeddings: Embeddings for existing PRs.
        recent_prs: Recent PRs for temporal clustering.
        vision_document_path: Path to YAML vision document.
        enable_tier3: Whether to run Tier 3.
        llm_provider: LLM provider for Tier 3. Defaults to config (auto-detect).
        llm_api_key: Unified API key (auto-detects provider from prefix).
    """
    all_flags: list[SuspicionFlag] = []
    dimensions: list[DimensionScore] = []

    # Load vision document early so focus_areas can feed into heuristics
    vision = None
    if vision_document_path:
        try:
            vision = load_vision_document(vision_document_path)
        except FileNotFoundError:
            print(f"Warning: Vision document not found at '{vision_document_path}' — skipping")

    # --- Tier 1: Dedup ---
    dedup_result: DedupResult
    if pr_embedding is not None:
        dedup_result = check_duplicates(
            pr, pr_embedding,
            existing_prs or [], existing_embeddings or [],
        )
    else:
        dedup_result = DedupResult(outcome=TierOutcome.SKIPPED)

    dimensions.append(DimensionScore(
        dimension="Hygiene & Dedup",
        score=0.0 if dedup_result.is_duplicate else 1.0,
        summary=f"Duplicate of PR#{dedup_result.duplicate_of} (similarity: {dedup_result.max_similarity:.2f})"
        if dedup_result.is_duplicate
        else "No duplicates found",
    ))

    if dedup_result.outcome == TierOutcome.GATED:
        return AssessmentScorecard(
            owner=pr.owner,
            repo=pr.repo,
            pr_number=pr.number,
            verdict=Verdict.RECOMMEND_CLOSE,
            confidence=dedup_result.max_similarity,
            dimensions=dimensions,
            dedup_result=dedup_result,
            flags=all_flags,
            summary=f"PR is a duplicate of PR#{dedup_result.duplicate_of} "
                    f"(similarity: {dedup_result.max_similarity:.2f}). Recommend closing.",
        )

    # --- Tier 2: Heuristics ---
    extra_sensitive = vision.focus_areas if vision else None
    heuristics_result = run_heuristics(pr, recent_prs, extra_sensitive_paths=extra_sensitive)
    all_flags.extend(heuristics_result.flags)

    dimensions.append(DimensionScore(
        dimension="Supply Chain Suspicion",
        score=1.0 - heuristics_result.suspicion_score,
        flags=heuristics_result.flags,
        summary=f"Suspicion score: {heuristics_result.suspicion_score:.2f} "
                f"({len(heuristics_result.flags)} flag(s))",
    ))

    if heuristics_result.outcome == TierOutcome.GATED:
        return AssessmentScorecard(
            owner=pr.owner,
            repo=pr.repo,
            pr_number=pr.number,
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
        vision_result = await run_vision_alignment(pr, vision, provider=llm_provider, api_key=llm_api_key)

        dimensions.append(DimensionScore(
            dimension="Vision Alignment",
            score=vision_result.alignment_score,
            summary=f"Alignment: {vision_result.alignment_score:.2f}",
        ))

        if vision_result.outcome == TierOutcome.ERROR:
            return AssessmentScorecard(
                owner=pr.owner,
                repo=pr.repo,
                pr_number=pr.number,
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
            return AssessmentScorecard(
                owner=pr.owner,
                repo=pr.repo,
                pr_number=pr.number,
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
            return AssessmentScorecard(
                owner=pr.owner,
                repo=pr.repo,
                pr_number=pr.number,
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

    # --- All clear → FAST_TRACK ---
    confidence = 0.8
    if vision_result and vision_result.alignment_score > 0:
        confidence = vision_result.alignment_score

    return AssessmentScorecard(
        owner=pr.owner,
        repo=pr.repo,
        pr_number=pr.number,
        verdict=Verdict.FAST_TRACK,
        confidence=confidence,
        dimensions=dimensions,
        dedup_result=dedup_result,
        heuristics_result=heuristics_result,
        vision_result=vision_result,
        flags=all_flags,
        summary="PR passed all tiers. Safe to fast-track.",
    )
