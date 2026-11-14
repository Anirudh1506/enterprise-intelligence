
from functools import lru_cache
from uuid import uuid4

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentProcessingError, DocumentService
from app.services.embedding_service import EmbeddingError, EmbeddingService
from app.schemas.document import DocumentUploadResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".txt", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
}

@lru_cache
def get_embedding_service() -> EmbeddingService:
    # one Gemini client per process; config errors are not cached and surface as 500
    return EmbeddingService()


def get_document_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> DocumentService:
    return DocumentService(ChunkingService(), embedding_service)


@router.post("/", 
    response_model=DocumentUploadResponse,
    status_code=201)
async def upload_documents(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
):

    # 1. Validate filename and extension
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = "." + file.filename.rsplit(".", 1)[-1].lower() \
        if "." in file.filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file extension. Allowed: .txt, .md"
        )

    # 2. Read uploaded file
    contents = await file.read(MAX_FILE_SIZE+1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit"
        )

    # 3. Validate empty file
    if not contents or not contents.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # 4. Decode UTF-8
    try:
        extracted_text = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )
    
    detected_content_type = MIME_TYPES[extension]

    # 5. Chunk + embed via the document service
    try:
        return await document_service.process_document(
            document_id=str(uuid4()),
            filename=file.filename,
            content_type=detected_content_type,
            text=extracted_text
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    except DocumentProcessingError as e:
        raise HTTPException(status_code=500, detail=str(e)) from None
