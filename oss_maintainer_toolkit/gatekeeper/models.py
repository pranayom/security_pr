"""Pydantic models for the Gatekeeper PR and issue triage pipeline."""

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
    state: str = "open"
    merged_at: datetime | None = None


# --- GitHub Issue Models ---

class IssueAuthor(BaseModel):
    login: str
    account_created_at: datetime | None = None
    contributions_to_repo: int = 0


class IssueMetadata(BaseModel):
    owner: str
    repo: str
    number: int
    title: str
    body: str = ""
    author: IssueAuthor
    state: str = "open"
    labels: list[str] = []
    assignees: list[str] = []
    milestone: str = ""
    reactions: dict[str, int] = {}
    comment_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None


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


class LabelDefinition(BaseModel):
    name: str
    description: str = ""
    keywords: list[str] = []
    color: str = ""
    source: str = "vision"  # "vision" or "github"


class VisionDocument(BaseModel):
    project: str
    principles: list[VisionPrinciple] = []
    anti_patterns: list[str] = []
    focus_areas: list[str] = []
    label_taxonomy: list[LabelDefinition] = []


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


class IssueScorecard(BaseModel):
    owner: str
    repo: str
    issue_number: int
    verdict: Verdict
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dimensions: list[DimensionScore] = []
    dedup_result: DedupResult | None = None
    heuristics_result: HeuristicsResult | None = None
    vision_result: VisionAlignmentResult | None = None
    flags: list[SuspicionFlag] = []
    summary: str = ""


# --- Issue-to-PR Linking ---

class LinkSuggestion(BaseModel):
    pr_number: int
    issue_number: int
    similarity: float
    pr_title: str = ""
    issue_title: str = ""
    is_explicit: bool = False  # True if PR body already references this issue


class LinkingReport(BaseModel):
    owner: str
    repo: str
    total_prs: int = 0
    total_issues: int = 0
    suggestions: list[LinkSuggestion] = []
    explicit_links: list[LinkSuggestion] = []
    orphan_issues: list[int] = []
    threshold: float = 0.0


# --- Smart Stale Detection ---

class StaleItem(BaseModel):
    item_type: str          # "pr" or "issue"
    number: int
    title: str = ""
    signal: str             # "superseded", "addressed", "blocked", "inactive"
    related_number: int = 0
    related_title: str = ""
    similarity: float = 0.0
    last_activity: datetime | None = None
    explanation: str = ""


class StalenessReport(BaseModel):
    owner: str
    repo: str
    superseded_prs: list[StaleItem] = []
    addressed_issues: list[StaleItem] = []
    blocked_prs: list[StaleItem] = []
    inactive_prs: list[StaleItem] = []
    inactive_issues: list[StaleItem] = []
    total_open_prs: int = 0
    total_open_issues: int = 0
    total_merged_prs_checked: int = 0
    threshold: float = 0.0
    inactive_days: int = 0


# --- Label Automation ---

class LabelSuggestion(BaseModel):
    label: str
    confidence: float
    embedding_similarity: float = 0.0
    keyword_matches: list[str] = []
    source: str = "vision"  # "vision" or "github"


class LabelingReport(BaseModel):
    owner: str
    repo: str
    item_type: str          # "pr" or "issue"
    item_number: int
    item_title: str = ""
    existing_labels: list[str] = []
    suggestions: list[LabelSuggestion] = []
    taxonomy_source: str = ""  # "vision", "github", "merged"
    taxonomy_size: int = 0
    threshold: float = 0.0


# --- Contributor Profiles ---

class ContributorProfile(BaseModel):
    owner: str
    repo: str
    username: str
    total_prs: int = 0
    merged_prs: int = 0
    open_prs: int = 0
    closed_prs: int = 0          # closed without merge
    merge_rate: float = 0.0
    test_inclusion_rate: float = 0.0
    avg_additions: float = 0.0
    avg_deletions: float = 0.0
    areas_of_expertise: list[str] = []  # top directories
    first_contribution: datetime | None = None
    last_contribution: datetime | None = None
    review_count: int = 0         # PRs reviewed by this user
    prs_analyzed: int = 0         # PRs fetched for detailed analysis


# --- Review Routing ---

class CodeOwnerRule(BaseModel):
    pattern: str
    owners: list[str]


class ReviewerSuggestion(BaseModel):
    username: str
    score: float
    reasons: list[str] = []


class ReviewRoutingReport(BaseModel):
    owner: str
    repo: str
    pr_number: int
    pr_title: str = ""
    changed_files: list[str] = []
    suggestions: list[ReviewerSuggestion] = []
    codeowners_found: bool = False
    recent_reviewers_checked: int = 0


# --- Cross-PR Conflict Detection ---

class ConflictPair(BaseModel):
    pr_a: int
    pr_b: int
    pr_a_title: str = ""
    pr_b_title: str = ""
    overlapping_files: list[str] = []
    semantic_similarity: float = 0.0
    confidence: float = 0.0       # blended file overlap + semantic


class ConflictReport(BaseModel):
    owner: str
    repo: str
    total_open_prs: int = 0
    conflict_pairs: list[ConflictPair] = []
    file_overlap_weight: float = 0.5
    threshold: float = 0.0


# --- Audit Backlog ---

class DuplicateCluster(BaseModel):
    members: list[dict] = []  # [{pr: int, title: str, author: str, similarity: float}]
    threshold: float = 0.0


class AuditRiskEntry(BaseModel):
    pr_number: int
    title: str = ""
    author: str = ""
    score: float = 0.0
    flag_count: int = 0
    high_severity_count: int = 0
    flags: list[str] = []


class AuditReport(BaseModel):
    owner: str
    repo: str
    prs_analyzed: int = 0
    total_open_prs: int = 0
    elapsed_seconds: float = 0.0
    # Verdict distribution
    fast_track_count: int = 0
    review_required_count: int = 0
    recommend_close_count: int = 0
    # Duplicate clusters at multiple thresholds
    clusters_090: list[DuplicateCluster] = []
    clusters_085: list[DuplicateCluster] = []
    clusters_080: list[DuplicateCluster] = []
    # Highest-risk PRs
    highest_risk: list[AuditRiskEntry] = []
    # Flag frequency: {flag_id: count}
    flag_frequency: dict[str, int] = {}
    # Contributor stats
    unique_authors: int = 0
    first_time_contributors: int = 0
    new_accounts: int = 0
    sensitive_path_prs: int = 0
    low_test_prs: int = 0
    # Vision document used
    vision_document: str = ""
