"""Label Automation: classify PRs and issues into project-defined label taxonomies.

Tier 1 (embedding similarity) + Tier 2 (keyword heuristics) only. No LLM, $0 cost.
Dual-source taxonomy: vision document label_taxonomy + GitHub repo labels API.
"""

from __future__ import annotations

import re

import numpy as np

from oss_maintainer_toolkit.gatekeeper.config import gatekeeper_settings
from oss_maintainer_toolkit.gatekeeper.dedup import _get_model
from oss_maintainer_toolkit.gatekeeper.linking import _compute_similarity_matrix
from oss_maintainer_toolkit.gatekeeper.models import (
    IssueMetadata,
    LabelDefinition,
    LabelingReport,
    LabelSuggestion,
    PRMetadata,
)


def _build_label_embedding_text(label: LabelDefinition) -> str:
    """Build embedding text from label name, description, and keywords."""
    parts = [label.name]
    if label.description:
        parts.append(label.description)
    if label.keywords:
        parts.append(" ".join(label.keywords))
    return " ".join(parts)


def _build_item_embedding_text(item: PRMetadata | IssueMetadata) -> str:
    """Build embedding text from a PR or issue for label classification."""
    parts = [item.title]
    if item.body:
        parts.append(item.body[:1000])
    if isinstance(item, PRMetadata):
        if item.files:
            parts.append(" ".join(f.filename for f in item.files))
    if item.labels:
        parts.append(" ".join(item.labels))
    return "\n".join(parts)


def compute_label_embeddings(labels: list[LabelDefinition]) -> list[list[float]]:
    """Compute embedding vectors for a list of label definitions."""
    if not labels:
        return []
    model = _get_model()
    texts = [_build_label_embedding_text(lb) for lb in labels]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def compute_item_embedding(item: PRMetadata | IssueMetadata) -> list[float]:
    """Compute embedding vector for a PR or issue."""
    model = _get_model()
    text = _build_item_embedding_text(item)
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def _compute_keyword_scores(
    item_text: str,
    labels: list[LabelDefinition],
) -> list[tuple[float, list[str]]]:
    """Compute keyword match scores for each label against item text.

    Returns list of (score, matched_keywords) tuples, one per label.
    Score is fraction of keywords that matched (0.0 if no keywords defined).
    """
    text_lower = item_text.lower()
    results: list[tuple[float, list[str]]] = []

    for label in labels:
        if not label.keywords:
            results.append((0.0, []))
            continue

        matched = []
        for kw in label.keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(kw)

        score = len(matched) / len(label.keywords)
        results.append((score, matched))

    return results


def github_labels_to_taxonomy(raw_labels: list[dict]) -> list[LabelDefinition]:
    """Convert GitHub API label dicts to LabelDefinition objects."""
    taxonomy: list[LabelDefinition] = []
    for lb in raw_labels:
        name = lb.get("name", "").strip()
        if not name:
            continue
        taxonomy.append(LabelDefinition(
            name=name,
            description=lb.get("description", "") or "",
            color=lb.get("color", "") or "",
            source="github",
        ))
    return taxonomy


def merge_taxonomies(
    vision_labels: list[LabelDefinition],
    github_labels: list[LabelDefinition],
) -> list[LabelDefinition]:
    """Merge vision doc labels with GitHub labels. Vision overrides same-name GitHub labels."""
    vision_names = {lb.name.lower() for lb in vision_labels}
    merged = list(vision_labels)
    for lb in github_labels:
        if lb.name.lower() not in vision_names:
            merged.append(lb)
    return merged


def classify_item(
    item: PRMetadata | IssueMetadata,
    item_embedding: list[float],
    taxonomy: list[LabelDefinition],
    label_embeddings: list[list[float]],
    threshold: float = 0.0,
    keyword_weight: float = 0.0,
    max_suggestions: int = 0,
) -> LabelingReport:
    """Classify a PR or issue against a label taxonomy.

    Blended score: (1 - keyword_weight) * embedding_similarity + keyword_weight * keyword_score.

    Args:
        item: PR or issue metadata.
        item_embedding: Pre-computed embedding for the item.
        taxonomy: List of label definitions to classify against.
        label_embeddings: Pre-computed embeddings for each label.
        threshold: Minimum confidence to include (0 = config default).
        keyword_weight: Weight for keyword score vs embedding (0 = config default).
        max_suggestions: Max labels to suggest (0 = config default).

    Returns:
        LabelingReport with suggested labels sorted by confidence.
    """
    if threshold <= 0:
        threshold = gatekeeper_settings.label_similarity_threshold
    if keyword_weight <= 0:
        keyword_weight = gatekeeper_settings.label_keyword_weight
    if max_suggestions <= 0:
        max_suggestions = gatekeeper_settings.label_max_suggestions

    is_pr = isinstance(item, PRMetadata)
    item_type = "pr" if is_pr else "issue"

    report = LabelingReport(
        owner=item.owner,
        repo=item.repo,
        item_type=item_type,
        item_number=item.number,
        item_title=item.title,
        existing_labels=list(item.labels),
        taxonomy_size=len(taxonomy),
        threshold=threshold,
    )

    if not taxonomy or not label_embeddings:
        return report

    # Compute embedding similarities
    sim_matrix = _compute_similarity_matrix([item_embedding], label_embeddings)

    # Compute keyword scores
    item_text = _build_item_embedding_text(item)
    kw_scores = _compute_keyword_scores(item_text, taxonomy)

    # Blend and collect suggestions
    suggestions: list[LabelSuggestion] = []
    embed_weight = 1.0 - keyword_weight

    for j, label in enumerate(taxonomy):
        emb_sim = float(sim_matrix[0, j]) if sim_matrix.size > 0 else 0.0
        kw_score, kw_matches = kw_scores[j]
        confidence = embed_weight * emb_sim + keyword_weight * kw_score

        if confidence >= threshold:
            suggestions.append(LabelSuggestion(
                label=label.name,
                confidence=round(confidence, 4),
                embedding_similarity=round(emb_sim, 4),
                keyword_matches=kw_matches,
                source=label.source,
            ))

    # Sort by confidence descending, limit
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    report.suggestions = suggestions[:max_suggestions]

    return report
