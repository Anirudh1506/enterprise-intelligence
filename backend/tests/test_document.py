"""API integration tests: real Postgres + pgvector, mock embedding/LLM providers."""
import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api.deps import get_embedding_service
from app.core.config import Settings, get_settings
from app.main import app
from app.models.document import Chunk, Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingError, MockEmbeddingService

URL = "/api/v1/documents"


@pytest.fixture
def client(db):
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def upload(client, *files):
    return client.post(URL, files=[("files", f) for f in files])


def make_pdf(*pages: str) -> bytes:
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def chunk_count(db, document_id):
    return db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id))


# --- upload + processing ---

def test_upload_txt_is_processed(client, db):
    response = upload(client, ("notes.txt", b"Hello Enterprise Intelligence", "text/plain"))

    assert response.status_code == 202
    [doc] = response.json()
    assert doc["filename"] == "notes.txt"
    assert doc["file_type"] == "txt"
    assert doc["file_size"] == len(b"Hello Enterprise Intelligence")

    # TestClient runs background tasks before returning, so processing is done here
    detail = client.get(f"{URL}/{doc['id']}").json()
    assert detail["status"] == "COMPLETED"
    assert detail["error"] is None
    assert detail["chunk_count"] == 1
    assert detail["embedding_model"] == "mock:hashed-bow-v1:768"
    assert detail["chunks"][0]["content"] == "Hello Enterprise Intelligence"
    assert detail["chunks"][0]["embedding_dimensions"] == 768
    assert len(detail["chunks"][0]["embedding_preview"]) == 5
    assert chunk_count(db, doc["id"]) == 1


def test_upload_pdf_keeps_page_numbers(client):
    pdf = make_pdf("Revenue grew in the first quarter.", "Hiring is paused until March.")
    [doc] = upload(client, ("report.pdf", pdf, "application/pdf")).json()

    detail = client.get(f"{URL}/{doc['id']}").json()
    assert detail["status"] == "COMPLETED"
    assert detail["page_count"] == 2
    assert [c["page_number"] for c in detail["chunks"]] == [1, 2]
    assert "Hiring is paused" in detail["chunks"][1]["content"]


def test_upload_multiple_documents(client):
    response = upload(
        client,
        ("a.txt", b"alpha document", "text/plain"),
        ("b.md", b"# beta document", "text/markdown"),
    )
    assert response.status_code == 202
    assert [d["filename"] for d in response.json()] == ["a.txt", "b.md"]
    statuses = {d["filename"]: d["status"] for d in client.get(URL).json()}
    assert statuses == {"a.txt": "COMPLETED", "b.md": "COMPLETED"}


def test_markdown_content_type_detected(client):
    [doc] = upload(client, ("notes.md", b"# Hello", "text/plain")).json()
    assert doc["content_type"] == "text/markdown"


def test_filename_path_is_stripped(client):
    [doc] = upload(client, ("../../etc/passwd.txt", b"content", "text/plain")).json()
    assert doc["filename"] == "passwd.txt"


# --- validation ---

@pytest.mark.parametrize(
    "file, status",
    [
        (("test.exe", b"Some content", "application/octet-stream"), 415),
        (("noext", b"Some content", "text/plain"), 415),
        (("empty.txt", b"", "text/plain"), 400),
        (("whitespace.txt", b"   \n\t   ", "text/plain"), 400),
        (("invalid.txt", b"\xff\xfe\xfa", "text/plain"), 400),
        (("fake.pdf", b"not really a pdf", "application/pdf"), 400),
    ],
    ids=["bad-ext", "no-ext", "empty", "whitespace", "bad-utf8", "fake-pdf"],
)
def test_invalid_uploads_rejected(client, db, file, status):
    assert upload(client, file).status_code == status
    assert db.scalar(select(func.count()).select_from(Document)) == 0


def test_oversized_upload(client):
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=1)
    response = upload(client, ("large.txt", b"a" * (1024 * 1024 + 1), "text/plain"))
    assert response.status_code == 413


def test_one_bad_file_rejects_whole_batch(client, db):
    response = upload(
        client, ("ok.txt", b"fine", "text/plain"), ("bad.exe", b"x", "application/octet-stream")
    )
    assert response.status_code == 415
    assert db.scalar(select(func.count()).select_from(Document)) == 0


# --- failure handling ---

class FailingEmbeddings(MockEmbeddingService):
    async def embed_texts(self, texts, task_type=None):
        raise EmbeddingError("Embedding provider error (status 503)")


def test_embedding_failure_marks_failed_without_chunks(client, db):
    app.dependency_overrides[get_embedding_service] = FailingEmbeddings
    [doc] = upload(client, ("a.txt", b"some text", "text/plain")).json()

    detail = client.get(f"{URL}/{doc['id']}").json()
    assert detail["status"] == "FAILED"
    assert detail["error"] == "Embedding provider error (status 503)"
    assert detail["chunks"] == []


def test_pdf_without_text_fails_cleanly(client):
    [doc] = upload(client, ("blank.pdf", make_pdf(""), "application/pdf")).json()
    detail = client.get(f"{URL}/{doc['id']}").json()
    assert detail["status"] == "FAILED"
    assert "No extractable text" in detail["error"]


def test_reprocess_failed_document(client):
    app.dependency_overrides[get_embedding_service] = FailingEmbeddings
    [doc] = upload(client, ("a.txt", b"some text", "text/plain")).json()
    app.dependency_overrides.pop(get_embedding_service)

    response = client.post(f"{URL}/{doc['id']}/reprocess")
    assert response.status_code == 202
    detail = client.get(f"{URL}/{doc['id']}").json()
    assert detail["status"] == "COMPLETED"
    assert detail["error"] is None


def test_interrupted_documents_marked_failed_on_startup(db):
    db.add(Document(
        id="stuck", filename="s.txt", file_type="txt", content_type="text/plain", file_size=1,
        sha256="x", storage_path="/nonexistent", status=DocumentStatus.PROCESSING.value,
    ))
    db.commit()

    with TestClient(app) as c:
        detail = c.get(f"{URL}/stuck").json()
    assert detail["status"] == "FAILED"
    assert "interrupted" in detail["error"]


# --- read + delete ---

def test_get_missing_document_404(client):
    assert client.get(f"{URL}/nope").status_code == 404


def test_delete_removes_document_chunks_and_file(client, db):
    [doc] = upload(client, ("a.txt", b"delete me " * 300, "text/plain")).json()
    stored = db.get(Document, doc["id"]).storage_path
    assert chunk_count(db, doc["id"]) > 1

    assert client.delete(f"{URL}/{doc['id']}").status_code == 204

    assert client.get(f"{URL}/{doc['id']}").status_code == 404
    assert chunk_count(db, doc["id"]) == 0
    import os
    assert not os.path.exists(stored)


def test_cannot_delete_while_processing(client, db):
    db.add(Document(
        id="busy", filename="b.txt", file_type="txt", content_type="text/plain", file_size=1,
        sha256="x", storage_path="/nonexistent", status=DocumentStatus.PROCESSING.value,
    ))
    db.commit()
    assert client.delete(f"{URL}/busy").status_code == 409


# --- search + ask ---

def test_search_ranks_relevant_document_first(client):
    upload(
        client,
        ("vacation.txt", b"Employees receive 25 vacation days per year.", "text/plain"),
        ("security.txt", b"Passwords must be rotated every 90 days.", "text/plain"),
    )
    response = client.post("/api/v1/search", json={"query": "how many vacation days", "top_k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["filename"] == "vacation.txt"
    assert results[0]["score"] > results[1]["score"]


def test_search_filters_by_document(client):
    docs = upload(
        client,
        ("a.txt", b"vacation policy", "text/plain"),
        ("b.txt", b"vacation rules", "text/plain"),
    ).json()
    response = client.post(
        "/api/v1/search", json={"query": "vacation", "document_ids": [docs[1]["id"]]}
    )
    assert {r["filename"] for r in response.json()["results"]} == {"b.txt"}


def test_search_validation(client):
    assert client.post("/api/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_ask_with_no_documents_is_insufficient(client):
    response = client.post("/api/v1/ask", json={"question": "What is our vacation policy?"})
    data = response.json()
    assert response.status_code == 200
    assert data["insufficient_context"] is True
    assert data["sources"] == []


def test_ask_with_mock_llm_returns_sources_without_fabricating(client):
    upload(client, ("vacation.txt", b"Employees receive 25 vacation days per year.", "text/plain"))
    data = client.post("/api/v1/ask", json={"question": "vacation days?"}).json()

    assert data["llm"]["provider"] == "mock"
    assert "no answer was generated" in data["answer"]
    assert data["sources"][0]["filename"] == "vacation.txt"
    assert data["sources"][0]["source_number"] == 1


def test_deleted_document_not_searchable(client):
    [doc] = upload(client, ("a.txt", b"unique zebra content", "text/plain")).json()
    client.delete(f"{URL}/{doc['id']}")
    results = client.post("/api/v1/search", json={"query": "zebra"}).json()["results"]
    assert results == []


def test_health(client):
    data = client.get("/health").json()
    assert data["database"] == "ok"
    assert data["embedding_model"] == "mock:hashed-bow-v1:768"
    assert data["llm_model"] == "mock:none"


def test_repository_complete_is_idempotent(client, db):
    [doc] = upload(client, ("a.txt", b"repeatable " * 200, "text/plain")).json()
    first = chunk_count(db, doc["id"])
    client.post(f"{URL}/{doc['id']}/reprocess")
    assert chunk_count(db, doc["id"]) == first
    assert DocumentRepository(db).get(doc["id"]).status == "COMPLETED"
