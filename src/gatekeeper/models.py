"""Pydantic models for the Gatekeeper PR risk assessment pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# --- Enums ---

class Verdict(str, Enum):
    FAST_TRACK = "fast_track"
    REVIEW_REQUIRED = "review_required"
    RECOMMEND_CLOSE = "recommend_close"


class TierOutcome(str, Enum):
    PASS = "pass"
    GATED = "gated"
    SKIPPED = "skipped"
    ERROR = "error"


class FlagSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- GitHub PR Models ---

class PRAuthor(BaseModel):
    login: str
    account_created_at: datetime | None = None
    contributions_to_repo: int = 0


class PRFileChange(BaseModel):
    filename: str
    status: str = "modified"  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: str = ""


class PRMetadata(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str = ""
    author: PRAuthor
    files: list[PRFileChange] = []
    diff_text: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    linked_issues: list[int] = []
    labels: list[str] = []
    total_additions: int = 0
    total_deletions: int = 0


# --- Tier 1: Dedup ---

class DedupResult(BaseModel):
    outcome: TierOutcome
    is_duplicate: bool = False
    duplicate_of: int | None = None
    max_similarity: float = 0.0


# --- Tier 2: Heuristics ---

class SuspicionFlag(BaseModel):
    rule_id: str
    severity: FlagSeverity
    title: str
    explanation: str
    evidence: str = ""


class HeuristicsResult(BaseModel):
    outcome: TierOutcome
    suspicion_score: float = 0.0
    flags: list[SuspicionFlag] = []


# --- Tier 3: Vision Alignment ---

class VisionPrinciple(BaseModel):
    name: str
    description: str


class VisionDocument(BaseModel):
    project: str
    principles: list[VisionPrinciple] = []
    anti_patterns: list[str] = []
    focus_areas: list[str] = []


class VisionAlignmentResult(BaseModel):
    outcome: TierOutcome
    alignment_score: float = 0.0
    violated_principles: list[str] = []
    strengths: list[str] = []
    concerns: list[str] = []


# --- Scorecard ---

class DimensionScore(BaseModel):
    dimension: str
    score: float = 0.0
    flags: list[SuspicionFlag] = []
    summary: str = ""


class AssessmentScorecard(BaseModel):
    owner: str
    repo: str
    pr_number: int
    verdict: Verdict
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: list[DimensionScore] = []
    dedup_result: DedupResult | None = None
    heuristics_result: HeuristicsResult | None = None
    vision_result: VisionAlignmentResult | None = None
    flags: list[SuspicionFlag] = []
    summary: str = ""
