from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# --- internal pipeline types ---

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    page_number: Optional[int] = None


class EmbeddedChunk(DocumentChunk):
    embedding: list[float]


class EmbeddingMetadata(BaseModel):
    provider: str
    model: str
    dimensions: int


class ProcessedDocument(BaseModel):
    """Result of DocumentService: chunks paired with their vectors, not yet persisted."""

    document_id: str
    filename: str
    content_type: str
    status: str
    created_at: datetime
    chunk_count: int
    embedding_metadata: EmbeddingMetadata
    chunks: list[EmbeddedChunk]


# --- API responses ---

class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    content_type: str
    file_size: int
    status: str
    error: Optional[str]
    page_count: Optional[int]
    char_count: Optional[int]
    chunk_count: int
    embedding_model: Optional[str]
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]


class ChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    embedding_dimensions: int
    # first few values only, so stored vectors can be inspected without shipping 768 floats
    embedding_preview: list[float]


class DocumentDetail(DocumentRead):
    chunks: list[ChunkRead]
