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

    # Issue triage
    issue_duplicate_threshold: float = 0.85
    issue_suspicion_threshold: float = 0.6
    issue_min_body_length: int = 30

    # Issue-to-PR Linking
    linking_similarity_threshold: float = 0.45

    # Smart Stale Detection
    stale_similarity_threshold: float = 0.75
    stale_inactive_days: int = 90

    # Tier 3: Vision — LLM provider
    llm_provider: str = "auto"  # auto, openrouter, openai, anthropic, gemini, generic, claude_cli
    llm_api_key: str = ""  # unified key (auto-detects provider from prefix)
    llm_timeout_seconds: int = 60  # shared timeout for all HTTP providers

    # OpenRouter (free, works in CI)
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-120b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: int = 60

    # OpenAI direct
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic direct
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_base_url: str = "https://api.anthropic.com"

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Generic OpenAI-compatible
    generic_api_key: str = ""
    generic_model: str = ""
    generic_base_url: str = ""  # required for generic

    # Claude CLI (fallback — requires local Max subscription)
    claude_command: str = "claude"
    claude_timeout_seconds: int = 120

    vision_document_path: str = ""
    enable_tier3: bool = True

    model_config = {"env_prefix": "AUDITOR_GK_"}


gatekeeper_settings = GatekeeperSettings()
