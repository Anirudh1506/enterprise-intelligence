import logging
import os
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal, get_db
from app.models.document import VECTOR_DIMENSIONS
from app.repositories.document_repository import DocumentRepository
from app.services.ask_service import AskService
from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentService
from app.services.embedding_service import (
    EmbeddingConfigError,
    EmbeddingService,
    MockEmbeddingService,
)
from app.services.ingestion_service import IngestionService
from app.services.llm_service import GeminiLLMService, MockLLMService
from app.services.search_service import SearchService
from app.services.storage_service import FileStorage

logger = logging.getLogger(__name__)


def _resolve_provider(configured: str) -> str:
    configured = configured.strip().lower()
    if configured == "auto":
        return "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "mock"
    if configured not in ("gemini", "mock"):
        raise EmbeddingConfigError("Provider must be one of: auto, gemini, mock")
    return configured


@lru_cache
def get_embedding_service():
    # cached: one client per process; config errors are not cached and surface as 500
    if _resolve_provider(get_settings().embedding_provider) == "mock":
        logger.warning("Using MOCK embedding provider (lexical hashing, not semantic)")
        service = MockEmbeddingService()
    else:
        service = EmbeddingService()
    if service.dimensions != VECTOR_DIMENSIONS:
        raise EmbeddingConfigError(
            f"Embedding dimensions must be {VECTOR_DIMENSIONS} to match the database schema"
        )
    return service


@lru_cache
def get_llm_service():
    settings = get_settings()
    if _resolve_provider(settings.llm_provider) == "mock":
        logger.warning("Using MOCK LLM provider: answers will not be generated")
        return MockLLMService()
    return GeminiLLMService(
        os.getenv("GEMINI_API_KEY", "").strip(),
        settings.gemini_llm_model,
        settings.llm_timeout_seconds,
    )


def get_storage(settings: Settings = Depends(get_settings)) -> FileStorage:
    return FileStorage(settings.storage_dir)


def get_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    return DocumentRepository(db)


def get_document_service(
    settings: Settings = Depends(get_settings),
    embedding_service=Depends(get_embedding_service),
) -> DocumentService:
    return DocumentService(
        ChunkingService(settings.chunk_size, settings.chunk_overlap), embedding_service
    )


def get_ingestion_service(
    document_service: DocumentService = Depends(get_document_service),
) -> IngestionService:
    return IngestionService(SessionLocal, document_service)


def get_search_service(
    repository: DocumentRepository = Depends(get_repository),
    embedding_service=Depends(get_embedding_service),
    settings: Settings = Depends(get_settings),
) -> SearchService:
    return SearchService(
        repository, embedding_service, settings.search_default_top_k, settings.search_max_top_k
    )


def get_ask_service(
    search_service: SearchService = Depends(get_search_service),
    llm_service=Depends(get_llm_service),
) -> AskService:
    return AskService(search_service, llm_service)
