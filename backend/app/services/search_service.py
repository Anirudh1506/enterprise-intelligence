from typing import Optional

from app.repositories.document_repository import DocumentRepository
from app.schemas.search import SearchResult


class SearchService:
    def __init__(self, repository: DocumentRepository, embedding_service, default_top_k: int, max_top_k: int):
        self.repository = repository
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k

    async def search(
        self, query: str, top_k: Optional[int] = None, document_ids: Optional[list[str]] = None
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("Query must not be empty")
        k = min(top_k or self.default_top_k, self.max_top_k)

        query_vector = await self.embedding_service.embed_text(query, task_type="RETRIEVAL_QUERY")
        rows = self.repository.search(query_vector, self.embedding_service.model_id, k, document_ids)

        return [
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                content=chunk.content,
                score=round(score, 4),
            )
            for chunk, filename, score in rows
        ]
