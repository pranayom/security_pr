"""Dependency file parsers for requirements.txt, package.json, etc."""

import json
import re
from pathlib import Path

from mcp_ai_auditor.models import Dependency


def parse_requirements_txt(file_path: Path) -> list[Dependency]:
    """Parse a requirements.txt file into dependencies."""
    deps: list[Dependency] = []
    content = file_path.read_text(encoding="utf-8", errors="replace")

    for line in content.splitlines():
        line = line.strip()
        # Skip comments, blank lines, options
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Handle ==, >=, <=, ~=, != version specs
        match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([=<>!~]+)\s*([^\s;#,]+)", line)
        if match:
            deps.append(Dependency(
                name=match.group(1).lower(),
                version=match.group(3),
                source_file=str(file_path),
            ))
        else:
            # Package without version pin
            pkg_match = re.match(r"^([a-zA-Z0-9_.-]+)", line)
            if pkg_match:
                deps.append(Dependency(
                    name=pkg_match.group(1).lower(),
                    version="*",
                    source_file=str(file_path),
                ))

    return deps


def parse_package_json(file_path: Path) -> list[Dependency]:
    """Parse a package.json file into dependencies."""
    deps: list[Dependency] = []
    content = json.loads(file_path.read_text(encoding="utf-8"))

    for dep_key in ("dependencies", "devDependencies"):
        for name, version_spec in content.get(dep_key, {}).items():
            # Strip leading ^, ~, = etc. to get base version
            version = re.sub(r"^[^0-9]*", "", version_spec) or version_spec
            deps.append(Dependency(
                name=name,
                version=version,
                source_file=str(file_path),
            ))

    return deps


def find_and_parse_dependencies(target: str) -> list[Dependency]:
    """Find and parse all dependency files in a target path.

    Supports: requirements.txt, requirements*.txt, package.json
    """
    target_path = Path(target)
    deps: list[Dependency] = []

    if target_path.is_file():
        if target_path.name.endswith(".txt") and "requirements" in target_path.name.lower():
            deps.extend(parse_requirements_txt(target_path))
        elif target_path.name == "package.json":
            deps.extend(parse_package_json(target_path))
        return deps

    if not target_path.is_dir():
        return deps

    # Find requirements*.txt files
    for req_file in target_path.rglob("requirements*.txt"):
        try:
            deps.extend(parse_requirements_txt(req_file))
        except Exception:
            pass

    # Find package.json files (skip node_modules)
    for pkg_file in target_path.rglob("package.json"):
        if "node_modules" in pkg_file.parts:
            continue
        try:
            deps.extend(parse_package_json(pkg_file))
        except Exception:
            pass

    return deps
