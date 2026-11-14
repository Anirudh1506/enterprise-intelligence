from pydantic import BaseModel, ConfigDict
from datetime import datetime


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int


class EmbeddedChunk(DocumentChunk):
    embedding: list[float]


class EmbeddingMetadata(BaseModel):
    provider: str
    model: str
    dimensions: int


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    content_type: str
    status: str
    created_at: datetime
    chunk_count: int
    embedding_metadata: EmbeddingMetadata
    # dev only: vectors must not be exposed in the public upload response in production
    chunks: list[EmbeddedChunk]
