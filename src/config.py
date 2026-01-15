"""Configuration management using pydantic-settings + TOML config file."""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import tomli
except ImportError:
    tomli = None


def _load_config_toml() -> dict[str, Any]:
    """Load settings from config.toml.

    Returns:
        Dict of config values
    """
    config_path = Path("config.toml")

    if not config_path.exists():
        return {}

    if tomli is None:
        # tomli not installed, return empty dict
        return {}

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
            return config
    except Exception:
        return {}


# Load config.toml once at module load
_config_toml = _load_config_toml()


class Settings(BaseSettings):
    """Application settings loaded from .env (secrets) and config.toml."""

    # ============ API Keys (from .env) ============
    llamaparse_api_key: str | None = None
    chroma_cloud_api_key: str
    chroma_tenant: str
    chroma_database: str
    openrouter_api_key: str
    cohere_api_key: str

    # ============ ChromaDB Cloud settings ============
    chroma_host: str = "https://api.trychroma.cloud"
    chroma_quota_limit: int = 300  # Max items per request
    chroma_batch_size: int = 250  # Stay under quota limit

    # ============ Model settings (from config.toml) ============
    @property
    def llm_model(self) -> str:
        """Get LLM model from config.toml or default."""
        return self._get_from_toml("models", "llm", "anthropic/claude-3.5-sonnet")

    @property
    def embedding_model(self) -> str:
        """Get embedding model from config.toml or default."""
        return self._get_from_toml("models", "embedding", "BAAI/bge-large-en-v1.5")

    @property
    def embedding_provider(self) -> str:
        """Get embedding provider from config.toml or default."""
        return self._get_from_toml("models", "embedding_provider", "openrouter")

    # ============ Pipeline settings (from config.toml) ============
    @property
    def top_k_retrieve(self) -> int:
        """Get top_k_retrieve from config.toml or default."""
        return int(self._get_from_toml("pipeline", "top_k_retrieve", 50))

    @property
    def top_k_rerank(self) -> int:
        """Get top_k_rerank from config.toml or default."""
        return int(self._get_from_toml("pipeline", "top_k_rerank", 5))

    @property
    def max_papers(self) -> int:
        """Get max_papers from config.toml or default."""
        return int(self._get_from_toml("pipeline", "max_papers", 200))

    @property
    def chunk_size(self) -> int:
        """Get chunk_size from config.toml or default."""
        return int(self._get_from_toml("pipeline", "chunk_size", 800))

    @property
    def chunk_overlap(self) -> int:
        """Get chunk_overlap from config.toml or default."""
        return int(self._get_from_toml("pipeline", "chunk_overlap", 100))

    def _get_from_toml(self, section: str, key: str, default: Any = None) -> Any:
        """Get value from config.toml with fallback to default.

        Args:
            section: TOML section (e.g., "models", "pipeline")
            key: Key within section
            default: Default value if not found

        Returns:
            Value from config.toml or default
        """
        if section in _config_toml and key in _config_toml[section]:
            return _config_toml[section][key]
        return default

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance
settings = Settings()
