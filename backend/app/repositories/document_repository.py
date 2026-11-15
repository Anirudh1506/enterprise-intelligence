from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, selectinload

from app.models.document import Chunk, Document, DocumentStatus
from app.schemas.document import ProcessedDocument


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_all(self, documents: list[Document]) -> None:
        self.db.add_all(documents)
        self.db.commit()

    def get(self, document_id: str, with_chunks: bool = False) -> Optional[Document]:
        stmt = select(Document).where(Document.id == document_id)
        if with_chunks:
            stmt = stmt.options(selectinload(Document.chunks))
        return self.db.scalars(stmt).first()

    def list_all(self, limit: int = 500) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit)
        return list(self.db.scalars(stmt))

    def delete(self, document: Document) -> None:
        # chunks are removed by ON DELETE CASCADE
        self.db.delete(document)
        self.db.commit()

    def claim_for_processing(self, document_id: str) -> bool:
        """Atomically move UPLOADED -> PROCESSING. False if already claimed/finished."""
        result = self.db.execute(
            update(Document)
            .where(Document.id == document_id, Document.status == DocumentStatus.UPLOADED.value)
            .values(status=DocumentStatus.PROCESSING.value, error=None)
        )
        self.db.commit()
        return result.rowcount == 1

    def reset_for_reprocessing(self, document: Document) -> None:
        document.status = DocumentStatus.UPLOADED.value
        document.error = None
        self.db.commit()

    def complete(
        self,
        document: Document,
        processed: ProcessedDocument,
        extracted_text: str,
        page_count: Optional[int],
        embedding_model: str,
    ) -> None:
        """Replace all chunks and mark COMPLETED in one transaction (idempotent re-runs)."""
        self.db.execute(delete(Chunk).where(Chunk.document_id == document.id))
        self.db.add_all(
            Chunk(
                id=c.chunk_id,
                document_id=document.id,
                chunk_index=c.chunk_index,
                content=c.content,
                page_number=c.page_number,
                start_offset=c.start_offset,
                end_offset=c.end_offset,
                embedding=c.embedding,
                embedding_model=embedding_model,
            )
            for c in processed.chunks
        )
        document.status = DocumentStatus.COMPLETED.value
        document.error = None
        document.extracted_text = extracted_text
        document.page_count = page_count
        document.char_count = len(extracted_text)
        document.chunk_count = processed.chunk_count
        document.embedding_model = embedding_model
        document.processed_at = datetime.now(timezone.utc)
        self.db.commit()

    def fail(self, document_id: str, error: str) -> None:
        """Mark FAILED and drop any chunks so a failed document is never searchable."""
        self.db.rollback()
        self.db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.FAILED.value, error=error[:1000], chunk_count=0)
        )
        self.db.commit()

    def fail_interrupted(self) -> int:
        """On startup: anything left mid-pipeline by a crash/restart is marked FAILED."""
        result = self.db.execute(
            update(Document)
            .where(Document.status.in_([DocumentStatus.UPLOADED.value, DocumentStatus.PROCESSING.value]))
            .values(
                status=DocumentStatus.FAILED.value,
                error="Processing was interrupted by a server restart. Use Reprocess to retry.",
            )
        )
        self.db.commit()
        return result.rowcount

    def search(
        self,
        query_vector: list[float],
        embedding_model: str,
        top_k: int,
        document_ids: Optional[list[str]] = None,
    ) -> list[tuple[Chunk, str, float]]:
        """Top-k chunks by cosine similarity. Returns (chunk, filename, score)."""
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(Chunk, Document.filename, (1 - distance).label("score"))
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Document.status == DocumentStatus.COMPLETED.value,
                # vectors from different models/dimensions are not comparable
                Chunk.embedding_model == embedding_model,
            )
            .order_by(distance)
            .limit(top_k)
        )
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return [(row[0], row[1], float(row[2])) for row in self.db.execute(stmt)]
