from datetime import datetime, timezone
from .chunking_service import chunk_text


class DocumentService:

    async def process_document(
        self,
        document_id: str,
        filename: str,
        content_type: str,
        text: str
    ):
        # Document processing logic will go here.
        # Later: chunking, embeddings, vector storage, etc.
        
        chunkedTextData = chunk_text(text, document_id, 100, 50)
        print(chunkedTextData)
        return {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type or "application/octet-stream",
            "character_count": len(text),
            "status": "processed",
            "created_at": datetime.now(timezone.utc),
            "chunks": chunkedTextData
        }
