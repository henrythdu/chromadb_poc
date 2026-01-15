"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (llamaparse_api_key is optional - uses Docling by default)
    llamaparse_api_key: str | None = None
    chroma_cloud_api_key: str
    chroma_tenant: str
    chroma_database: str
    openrouter_api_key: str
    cohere_api_key: str

    # ChromaDB Cloud settings
    chroma_host: str = "https://api.trychroma.cloud"
    chroma_quota_limit: int = 300  # Max items per request
    chroma_batch_size: int = 250  # Stay under quota limit

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance
settings = Settings()
