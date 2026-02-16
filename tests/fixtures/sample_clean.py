"""Sample clean Python code with no vulnerabilities."""

import hashlib
import subprocess
from pathlib import Path


def get_user_safe(cursor, user_id: int):
    """Parameterized query — safe from SQL injection."""
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))


def run_command_safe(args: list[str]):
    """subprocess with shell=False — safe from command injection."""
    subprocess.run(args, shell=False, check=True, capture_output=True)


def hash_password_safe(password: str) -> str:
    """SHA-256 — cryptographically strong hash."""
    return hashlib.sha256(password.encode()).hexdigest()


def read_config(config_path: Path) -> str:
    """Safe file reading with Path validation."""
    resolved = config_path.resolve()
    if not resolved.is_relative_to(Path("/app/configs")):
        raise ValueError("Invalid config path")
    return resolved.read_text()


def calculate_sum(a: int, b: int) -> int:
    """Simple arithmetic — no security concerns."""
    return a + b
