from datetime import datetime, timezone

from app.schemas.document import DocumentUploadResponse, EmbeddedChunk, EmbeddingMetadata

from .chunking_service import ChunkingService
from .embedding_service import EmbeddingService


class DocumentProcessingError(Exception):
    """Raised when the ingestion workflow cannot complete. Message is safe to show to clients."""


class DocumentService:
    """Orchestrates ingestion. Chunking and embedding logic live in their own services."""

    def __init__(
        self,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService,
    ):
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service

    async def process_document(
        self,
        document_id: str,
        filename: str,
        content_type: str,
        text: str,
    ) -> DocumentUploadResponse:
        if type(text) is not str or not text.strip():
            raise ValueError("Document text must not be empty")

        chunks = self.chunking_service.chunk(text, document_id)
        if not chunks:
            raise DocumentProcessingError("No chunks were generated for the document")

        # EmbeddingError propagates as-is; its message is already safe
        vectors = await self.embedding_service.embed_texts([c.content for c in chunks])
        if len(vectors) != len(chunks):
            raise DocumentProcessingError(
                f"Embedding count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )

        # order is preserved by the embedding service, so position i <-> chunk i
        embedded_chunks = [
            EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            for chunk, vector in zip(chunks, vectors)
        ]

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            content_type=content_type or "application/octet-stream",
            status="processed",
            created_at=datetime.now(timezone.utc),
            chunk_count=len(embedded_chunks),
            embedding_metadata=EmbeddingMetadata(
                provider=self.embedding_service.provider,
                model=self.embedding_service.model,
                dimensions=self.embedding_service.dimensions,
            ),
            chunks=embedded_chunks,
        )
