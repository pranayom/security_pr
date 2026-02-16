"""Vulnerability patterns for regex-based scanning (OWASP Top 10 coverage)."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VulnPattern:
    """A vulnerability detection pattern."""
    name: str
    category: str
    severity: str  # maps to Severity enum value
    pattern: re.Pattern
    description: str
    languages: frozenset[str]  # file extensions this applies to, empty = all


def _pat(name: str, category: str, severity: str, regex: str, desc: str,
         languages: tuple[str, ...] = ()) -> VulnPattern:
    return VulnPattern(
        name=name,
        category=category,
        severity=severity,
        pattern=re.compile(regex, re.IGNORECASE),
        description=desc,
        languages=frozenset(languages),
    )


PATTERNS: list[VulnPattern] = [
    # --- SQL Injection ---
    _pat(
        "sql_injection_format",
        "SQL Injection",
        "critical",
        r"""(?:execute|executemany|raw)\s*\(\s*(?:f["\']|["\'].*\.format\(|["\'].*\+\s*\w)""",
        "SQL query built with string formatting/concatenation — use parameterized queries",
        (".py",),
    ),
    _pat(
        "sql_injection_concat",
        "SQL Injection",
        "critical",
        r"""(?:SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*["\']?\s*\+\s*\w""",
        "SQL query built with string concatenation",
    ),

    # --- Command Injection ---
    _pat(
        "command_injection_os_system",
        "Command Injection",
        "critical",
        r"""os\.system\s*\(""",
        "os.system() is vulnerable to command injection — use subprocess with shell=False",
        (".py",),
    ),
    _pat(
        "command_injection_subprocess_shell",
        "Command Injection",
        "critical",
        r"""subprocess\.(?:call|run|Popen|check_output|check_call)\s*\([^)]*shell\s*=\s*True""",
        "subprocess with shell=True is vulnerable to command injection",
        (".py",),
    ),
    _pat(
        "command_injection_eval",
        "Command Injection",
        "critical",
        r"""\beval\s*\(""",
        "eval() can execute arbitrary code — avoid using with untrusted input",
        (".py", ".js", ".ts", ".jsx", ".tsx"),
    ),
    _pat(
        "command_injection_exec",
        "Command Injection",
        "high",
        r"""\bexec\s*\(""",
        "exec() can execute arbitrary code",
        (".py",),
    ),

    # --- Hardcoded Secrets ---
    _pat(
        "hardcoded_password",
        "Hardcoded Secrets",
        "high",
        r"""(?:password|passwd|pwd|secret)\s*=\s*["\'][^"\']{4,}["\']""",
        "Possible hardcoded password/secret — use environment variables or a secrets manager",
    ),
    _pat(
        "hardcoded_aws_key",
        "Hardcoded Secrets",
        "critical",
        r"""(?:AKIA[0-9A-Z]{16})""",
        "Possible hardcoded AWS access key",
    ),
    _pat(
        "hardcoded_api_key",
        "Hardcoded Secrets",
        "high",
        r"""(?:api[_-]?key|apikey|api[_-]?secret)\s*=\s*["\'][^"\']{8,}["\']""",
        "Possible hardcoded API key — use environment variables",
    ),

    # --- XSS ---
    _pat(
        "xss_innerhtml",
        "XSS",
        "high",
        r"""\.innerHTML\s*=""",
        "Direct innerHTML assignment can lead to XSS — use textContent or sanitize input",
        (".js", ".ts", ".jsx", ".tsx"),
    ),
    _pat(
        "xss_dangerously_set",
        "XSS",
        "high",
        r"""dangerouslySetInnerHTML""",
        "dangerouslySetInnerHTML can lead to XSS — ensure input is sanitized",
        (".js", ".ts", ".jsx", ".tsx"),
    ),
    _pat(
        "xss_mark_safe",
        "XSS",
        "high",
        r"""mark_safe\s*\(""",
        "mark_safe() bypasses Django's auto-escaping — ensure input is sanitized",
        (".py",),
    ),

    # --- Path Traversal ---
    _pat(
        "path_traversal",
        "Path Traversal",
        "high",
        r"""open\s*\([^)]*\+[^)]*\)|open\s*\(.*(?:request|user|input|args|params)""",
        "File open with user-controlled path may allow path traversal",
        (".py",),
    ),

    # --- Insecure Deserialization ---
    _pat(
        "insecure_deserialization_pickle",
        "Insecure Deserialization",
        "critical",
        r"""pickle\.(?:loads?|Unpickler)\s*\(""",
        "pickle deserialization of untrusted data can execute arbitrary code",
        (".py",),
    ),
    _pat(
        "insecure_deserialization_yaml",
        "Insecure Deserialization",
        "high",
        r"""yaml\.(?:load|unsafe_load)\s*\(""",
        "yaml.load() without SafeLoader can execute arbitrary code — use yaml.safe_load()",
        (".py",),
    ),

    # --- Weak Cryptography ---
    _pat(
        "weak_crypto_md5",
        "Weak Cryptography",
        "medium",
        r"""(?:hashlib\.md5|MD5\.new|createHash\s*\(\s*["\']md5["\'])""",
        "MD5 is cryptographically broken — use SHA-256 or stronger",
    ),
    _pat(
        "weak_crypto_sha1",
        "Weak Cryptography",
        "medium",
        r"""(?:hashlib\.sha1|SHA1\.new|createHash\s*\(\s*["\']sha1["\'])""",
        "SHA1 is deprecated for security use — use SHA-256 or stronger",
    ),

    # --- SSL/TLS Issues ---
    _pat(
        "ssl_verify_disabled",
        "SSL/TLS Issues",
        "high",
        r"""verify\s*=\s*False""",
        "SSL verification disabled — vulnerable to man-in-the-middle attacks",
        (".py",),
    ),

    # --- Debug Mode ---
    _pat(
        "debug_mode",
        "Debug Mode",
        "medium",
        r"""(?:DEBUG\s*=\s*True|app\.debug\s*=\s*True|app\.run\s*\([^)]*debug\s*=\s*True)""",
        "Debug mode enabled — disable in production",
        (".py",),
    ),
]
