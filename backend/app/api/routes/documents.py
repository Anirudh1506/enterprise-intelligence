import hashlib
import logging
import re
from pathlib import PurePath
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile

from app.api.deps import get_ingestion_service, get_repository, get_storage
from app.core.config import Settings, get_settings
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import ChunkRead, DocumentDetail, DocumentRead
from app.services.ingestion_service import IngestionService
from app.services.storage_service import FileStorage

logger = logging.getLogger(__name__)

router = APIRouter()

MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
}
ALLOWED_EXTENSIONS = set(MIME_TYPES)
MAX_FILES_PER_REQUEST = 20


def _safe_filename(name: str) -> str:
    # keep only the basename and drop control characters; storage never uses this name
    name = re.sub(r"[\x00-\x1f\x7f]", "", PurePath(name.replace("\\", "/")).name).strip()
    return name[:255]


async def _validate_upload(file: UploadFile, settings: Settings) -> tuple[str, str, bytes]:
    """Returns (filename, extension, contents) or raises HTTPException."""
    # 1. Validate filename and extension
    filename = _safe_filename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type for {filename}. Allowed: .pdf, .txt, .md",
        )

    # 2. Read with a hard cap so oversized uploads are never fully buffered
    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413, detail=f"{filename} exceeds the {settings.max_upload_mb} MB limit"
        )

    # 3. Validate empty file
    if not contents or not contents.strip():
        raise HTTPException(status_code=400, detail=f"{filename} is empty")

    # 4. Validate content matches the declared type
    if extension == ".pdf":
        if not contents.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail=f"{filename} is not a valid PDF")
    else:
        try:
            contents.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail=f"{filename} must be UTF-8 encoded") from None

    return filename, extension, contents


@router.post("", response_model=list[DocumentRead], status_code=202)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more .pdf, .txt or .md files"),
    settings: Settings = Depends(get_settings),
    repository: DocumentRepository = Depends(get_repository),
    storage: FileStorage = Depends(get_storage),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    """Upload documents. Processing (extract, chunk, embed) runs in the background;
    poll GET /documents/{id} for status."""
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400, detail=f"At most {MAX_FILES_PER_REQUEST} files per request"
        )

    # validate everything first so a bad file rejects the batch before anything is stored
    validated = [await _validate_upload(f, settings) for f in files]

    documents, saved_paths = [], []
    try:
        for filename, extension, contents in validated:
            document_id = str(uuid4())
            path = storage.save(document_id, extension, contents)
            saved_paths.append(str(path))
            documents.append(
                Document(
                    id=document_id,
                    filename=filename,
                    file_type=extension.lstrip("."),
                    content_type=MIME_TYPES[extension],
                    file_size=len(contents),
                    sha256=hashlib.sha256(contents).hexdigest(),
                    storage_path=str(path),
                    status=DocumentStatus.UPLOADED.value,
                )
            )
        repository.add_all(documents)
    except Exception:
        # don't leave orphaned files behind if storing or the DB insert fails
        for p in saved_paths:
            storage.delete(p)
        raise

    for document in documents:
        background_tasks.add_task(ingestion.process, document.id)
    logger.info("Accepted %s document(s) for processing", len(documents))
    return documents


@router.get("", response_model=list[DocumentRead])
def list_documents(repository: DocumentRepository = Depends(get_repository)):
    return repository.list_all()


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, repository: DocumentRepository = Depends(get_repository)):
    document = repository.get(document_id, with_chunks=True)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = [
        ChunkRead(
            id=c.id,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            content=c.content,
            embedding_dimensions=len(c.embedding),
            embedding_preview=[round(float(v), 5) for v in c.embedding[:5]],
        )
        for c in document.chunks
    ]
    return DocumentDetail(**DocumentRead.model_validate(document).model_dump(), chunks=chunks)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    repository: DocumentRepository = Depends(get_repository),
    storage: FileStorage = Depends(get_storage),
):
    document = repository.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status in (DocumentStatus.UPLOADED.value, DocumentStatus.PROCESSING.value):
        raise HTTPException(status_code=409, detail="Document is still being processed")

    storage_path = document.storage_path
    repository.delete(document)  # chunks + embeddings removed via ON DELETE CASCADE
    storage.delete(storage_path)
    return Response(status_code=204)


@router.post("/{document_id}/reprocess", response_model=DocumentRead, status_code=202)
def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    repository: DocumentRepository = Depends(get_repository),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    document = repository.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status in (DocumentStatus.UPLOADED.value, DocumentStatus.PROCESSING.value):
        raise HTTPException(status_code=409, detail="Document is already being processed")

    repository.reset_for_reprocessing(document)
    background_tasks.add_task(ingestion.process, document.id)
    return document
