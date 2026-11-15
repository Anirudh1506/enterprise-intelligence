import asyncio
import logging
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.repositories.document_repository import DocumentRepository

from .document_service import DocumentProcessingError, DocumentService
from .embedding_service import EmbeddingError
from .extraction_service import ExtractionError, extract_pages

logger = logging.getLogger(__name__)

# errors whose messages are written to be shown to users
SAFE_ERRORS = (ExtractionError, EmbeddingError, DocumentProcessingError, ValueError)


class IngestionService:
    """Runs the pipeline for a stored document: extract -> clean -> chunk -> embed -> store.

    Status: UPLOADED -> PROCESSING -> COMPLETED | FAILED. Chunks are only written in the
    same transaction that marks COMPLETED, so a failure never leaves partial chunks.
    """

    def __init__(self, session_factory: sessionmaker, document_service: DocumentService):
        self.session_factory = session_factory
        self.document_service = document_service

    async def process(self, document_id: str) -> None:
        db: Session = self.session_factory()
        repo = DocumentRepository(db)
        try:
            if not repo.claim_for_processing(document_id):
                logger.info("Skipping document_id=%s (not in UPLOADED state)", document_id)
                return

            document = repo.get(document_id)
            logger.info("Processing document_id=%s file_type=%s", document_id, document.file_type)

            # PDF parsing is CPU-bound; keep it off the event loop
            pages = await asyncio.to_thread(
                extract_pages, Path(document.storage_path), document.file_type
            )
            processed = await self.document_service.process_document(
                document_id=document.id,
                filename=document.filename,
                content_type=document.content_type,
                pages=pages,
            )
            repo.complete(
                document,
                processed,
                extracted_text="\n\n".join(text for _, text in pages),
                page_count=len(pages) if document.file_type == "pdf" else None,
                embedding_model=self.document_service.embedding_service.model_id,
            )
            logger.info("Completed document_id=%s chunks=%s", document_id, processed.chunk_count)
        except SAFE_ERRORS as e:
            logger.warning("Failed document_id=%s error=%s", document_id, e)
            self._fail(repo, document_id, str(e))
        except Exception:
            logger.exception("Unexpected failure document_id=%s", document_id)
            self._fail(repo, document_id, "Internal error while processing the document")
        finally:
            db.close()

    @staticmethod
    def _fail(repo: DocumentRepository, document_id: str, message: str) -> None:
        try:
            repo.fail(document_id, message)
        except Exception:
            logger.exception("Could not mark document_id=%s as FAILED", document_id)
