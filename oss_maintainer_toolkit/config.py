"""Configuration for oss-maintainer-toolkit."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    osv_api_url: str = "https://api.osv.dev/v1"
    scan_extensions: list[str] = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb", ".php"]
    max_file_size_kb: int = 500
    max_call_depth: int = 5

    model_config = {"env_prefix": "AUDITOR_"}


settings = Settings()
