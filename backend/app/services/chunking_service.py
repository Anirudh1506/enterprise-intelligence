from uuid import uuid4

from app.schemas.document import DocumentChunk

# fixed size character chunking
def chunk_text(text: str, document_id:str, chunkSize: int = 500, overlap: int = 100) -> list[DocumentChunk]:
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id is required")
    if type(chunkSize) is not int or chunkSize <= 0:
        raise ValueError("Chunk Size should be a positive non-zero integer")
    if type(overlap) is not int or overlap < 0:
        raise ValueError("Overlap should not be negative")
    if overlap >= chunkSize:
        raise ValueError("Overlap must be smaller than the chunk size")

    result = []
    start = 0
    step = chunkSize - overlap

    while True:
        end = min(start + chunkSize, len(text))
        result.append(DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            chunk_index=len(result),
            content=text[start:end],
            start_offset=start,
            end_offset=end,
        ))
        if end >= len(text):
            break
        start += step

    return result


class ChunkingService:
    """Owns text segmentation. Thin wrapper so DocumentService can receive it via DI."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_id: str) -> list[DocumentChunk]:
        return chunk_text(text, document_id, self.chunk_size, self.overlap)
