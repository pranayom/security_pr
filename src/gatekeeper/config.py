"""Configuration for the Gatekeeper module."""

from pydantic_settings import BaseSettings


class GatekeeperSettings(BaseSettings):
    """Gatekeeper settings with AUDITOR_GK_ environment variable prefix."""

    # GitHub
    github_token: str = ""
    github_api_url: str = "https://api.github.com"
    rate_limit_buffer: int = 10

    # Cache
    cache_db_path: str = ".gatekeeper_cache.db"
    cache_ttl_hours: int = 24

    # Tier 1: Dedup
    embedding_model: str = "all-MiniLM-L6-v2"
    duplicate_threshold: float = 0.9

    # Tier 2: Heuristics
    suspicion_threshold: float = 0.6
    new_account_days: int = 90
    sensitive_paths: list[str] = [
        "auth", "crypto", "security", "login", "password",
        ".github/workflows", "ci", "cd", "deploy",
        "Dockerfile", "docker-compose",
        "requirements.txt", "package.json", "pyproject.toml",
        "Gemfile", "go.mod", "Cargo.toml",
    ]
    min_test_ratio: float = 0.1

    # Tier 3: Vision — LLM provider
    llm_provider: str = "openrouter"  # "openrouter" or "claude_cli"

    # OpenRouter (primary — free, works in CI)
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: int = 60

    # Claude CLI (fallback — requires local Max subscription)
    claude_command: str = "claude"
    claude_timeout_seconds: int = 120

    vision_document_path: str = ""
    enable_tier3: bool = True

    model_config = {"env_prefix": "AUDITOR_GK_"}


gatekeeper_settings = GatekeeperSettings()
