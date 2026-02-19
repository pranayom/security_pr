"""CVE checker using the OSV.dev batch API (free, no auth required)."""

import httpx

from oss_maintainer_toolkit.config import settings
from oss_maintainer_toolkit.models import CVECheckResult, CVERecord, Dependency, Severity
from oss_maintainer_toolkit.cve.parsers import find_and_parse_dependencies


# Map ecosystem names for OSV.dev
_ECOSYSTEM_MAP: dict[str, str] = {
    ".txt": "PyPI",
    ".json": "npm",
}


def _get_ecosystem(dep: Dependency) -> str:
    """Determine the OSV ecosystem based on source file extension."""
    for ext, eco in _ECOSYSTEM_MAP.items():
        if dep.source_file.endswith(ext):
            return eco
    return "PyPI"  # default


def _severity_from_osv(vuln: dict) -> Severity:
    """Extract severity from an OSV vulnerability record."""
    for severity_entry in vuln.get("severity", []):
        score_str = severity_entry.get("score", "")
        try:
            score = float(score_str)
            if score >= 9.0:
                return Severity.CRITICAL
            if score >= 7.0:
                return Severity.HIGH
            if score >= 4.0:
                return Severity.MEDIUM
            return Severity.LOW
        except (ValueError, TypeError):
            pass

    # Fall back to database_specific severity or default to MEDIUM
    db_specific = vuln.get("database_specific", {})
    severity_str = db_specific.get("severity", "").upper()
    if severity_str == "CRITICAL":
        return Severity.CRITICAL
    if severity_str == "HIGH":
        return Severity.HIGH
    if severity_str in ("MODERATE", "MEDIUM"):
        return Severity.MEDIUM
    if severity_str == "LOW":
        return Severity.LOW

    return Severity.MEDIUM


def _extract_fixed_version(vuln: dict, pkg_name: str) -> str:
    """Extract the fixed version from an OSV record."""
    for affected in vuln.get("affected", []):
        pkg = affected.get("package", {})
        if pkg.get("name", "").lower() == pkg_name.lower():
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if "fixed" in event:
                        return event["fixed"]
    return ""


async def query_osv_batch(dependencies: list[Dependency]) -> list[CVERecord]:
    """Query OSV.dev batch API for known vulnerabilities.

    Args:
        dependencies: List of dependencies to check.

    Returns:
        List of CVERecord for any known vulnerabilities.
    """
    if not dependencies:
        return []

    # Build batch query
    queries = []
    for dep in dependencies:
        if dep.version == "*":
            # Can't check unversioned deps
            continue
        queries.append({
            "package": {
                "name": dep.name,
                "ecosystem": _get_ecosystem(dep),
            },
            "version": dep.version,
        })

    if not queries:
        return []

    records: list[CVERecord] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.osv_api_url}/querybatch",
            json={"queries": queries},
        )
        resp.raise_for_status()
        batch_results = resp.json().get("results", [])

        for i, result in enumerate(batch_results):
            vulns = result.get("vulns", [])
            if not vulns:
                continue

            dep = dependencies[i]

            for vuln_ref in vulns:
                vuln_id = vuln_ref.get("id", "")
                # Fetch full vulnerability details
                try:
                    detail_resp = await client.get(
                        f"{settings.osv_api_url}/vulns/{vuln_id}"
                    )
                    detail_resp.raise_for_status()
                    vuln = detail_resp.json()
                except httpx.HTTPError:
                    vuln = vuln_ref  # Use the summary if detail fetch fails

                references = [
                    ref.get("url", "")
                    for ref in vuln.get("references", [])
                    if ref.get("url")
                ]

                records.append(CVERecord(
                    id=vuln_id,
                    summary=vuln.get("summary", vuln.get("details", "No summary")[:200]),
                    details=vuln.get("details", ""),
                    severity=_severity_from_osv(vuln),
                    affected_package=dep.name,
                    affected_version=dep.version,
                    fixed_version=_extract_fixed_version(vuln, dep.name),
                    references=references[:5],
                ))

    return records


async def check_cve(target: str) -> CVECheckResult:
    """Check dependencies for known CVEs using OSV.dev.

    Args:
        target: Path to a dependency file or directory containing them.

    Returns:
        CVECheckResult with found vulnerabilities.
    """
    errors: list[str] = []

    try:
        dependencies = find_and_parse_dependencies(target)
    except Exception as e:
        return CVECheckResult(
            dependencies_checked=0,
            total_vulnerabilities=0,
            dependencies=[],
            vulnerabilities=[],
            errors=[f"Error parsing dependencies: {e}"],
        )

    if not dependencies:
        return CVECheckResult(
            dependencies_checked=0,
            total_vulnerabilities=0,
            dependencies=dependencies,
            vulnerabilities=[],
            errors=["No dependency files found in target"],
        )

    try:
        vulnerabilities = await query_osv_batch(dependencies)
    except httpx.HTTPError as e:
        vulnerabilities = []
        errors.append(f"OSV API error: {e}")

    return CVECheckResult(
        dependencies_checked=len(dependencies),
        total_vulnerabilities=len(vulnerabilities),
        dependencies=dependencies,
        vulnerabilities=vulnerabilities,
        errors=errors,
    )
