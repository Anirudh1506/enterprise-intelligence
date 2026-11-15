from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App-wide settings from environment / backend/.env.

    Gemini-specific settings (GEMINI_API_KEY, GEMINI_EMBEDDING_*) are read by
    EmbeddingService itself so it stays independently configurable and testable.
    """

    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://localhost:5432/enterprise_intelligence"
    storage_dir: Path = BACKEND_DIR / "storage" / "uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:5173"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # auto = gemini when GEMINI_API_KEY is set, otherwise mock
    embedding_provider: str = "auto"
    llm_provider: str = "auto"
    gemini_llm_model: str = "gemini-3.5-flash"
    llm_timeout_seconds: int = 60

    search_default_top_k: int = 5
    search_max_top_k: int = 20

    @field_validator("storage_dir")
    @classmethod
    def _resolve_storage_dir(cls, v: Path) -> Path:
        # relative paths are relative to backend/, not to wherever the server was started
        return v if v.is_absolute() else BACKEND_DIR / v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
