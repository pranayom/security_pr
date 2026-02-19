"""Shared Pydantic models for oss-maintainer-toolkit."""

from enum import Enum
from pydantic import BaseModel


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityFinding(BaseModel):
    """A single vulnerability finding from the scanner."""
    file: str
    line: int
    severity: Severity
    category: str
    pattern_name: str
    matched_text: str
    description: str


class ScanResult(BaseModel):
    """Result of a vulnerability scan across files."""
    files_scanned: int
    total_findings: int
    findings: list[VulnerabilityFinding]
    errors: list[str] = []


class TaintFlow(BaseModel):
    """A single taint flow from source to sink."""
    file: str
    source_line: int
    source_code: str
    sink_line: int
    sink_code: str
    taint_type: str
    variable: str
    description: str


class DataFlowResult(BaseModel):
    """Result of data flow / taint analysis."""
    files_analyzed: int
    total_flows: int
    flows: list[TaintFlow]
    errors: list[str] = []


class Dependency(BaseModel):
    """A parsed dependency with name and version."""
    name: str
    version: str
    source_file: str


class CVERecord(BaseModel):
    """A CVE/vulnerability record from OSV.dev."""
    id: str
    summary: str
    details: str = ""
    severity: Severity
    affected_package: str
    affected_version: str
    fixed_version: str = ""
    references: list[str] = []


class CVECheckResult(BaseModel):
    """Result of CVE checking across dependencies."""
    dependencies_checked: int
    total_vulnerabilities: int
    dependencies: list[Dependency]
    vulnerabilities: list[CVERecord]
    errors: list[str] = []
