
import pytest

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.api.routes.documents import get_embedding_service
from app.main import app
from app.services.embedding_service import EmbeddingError

client = TestClient(app)

DIMS = 4


class FakeEmbeddingService:
    provider = "google"
    model = "fake-model"
    dimensions = DIMS

    def __init__(self):
        # vector i is [i, i, i, i] so tests can check chunk <-> vector order
        self.embed_texts = AsyncMock(
            side_effect=lambda texts: [[float(i)] * DIMS for i in range(len(texts))]
        )


@pytest.fixture(autouse=True)
def fake_embeddings():
    # never hit the real Gemini API from route tests
    fake = FakeEmbeddingService()
    app.dependency_overrides[get_embedding_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_successful_upload(fake_embeddings):

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "test.txt",
                b"Hello Enterprise Intelligence",
                "application/octet-stream"
            )
        }
    )

    assert response.status_code == 201

    data = response.json()

    assert data["filename"] == "test.txt"
    assert data["content_type"] == "text/plain"
    assert data["status"] == "processed"
    assert data["chunk_count"] == 1
    assert data["embedding_metadata"] == {
        "provider": "google", "model": "fake-model", "dimensions": DIMS
    }
    assert data["chunks"][0]["content"] == "Hello Enterprise Intelligence"
    assert data["chunks"][0]["embedding"] == [0.0] * DIMS

    fake_embeddings.embed_texts.assert_awaited_once()


def test_embedding_failure_returns_502(fake_embeddings):
    fake_embeddings.embed_texts.side_effect = EmbeddingError("Embedding provider error (status 500)")

    response = client.post(
        "/api/v1/documents/",
        files={"file": ("test.txt", b"Hello", "text/plain")}
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Embedding provider error (status 500)"


def test_unsupported_extension():

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "test.pdf",
                b"Some content",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 415


def test_empty_file():

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "empty.txt",
                b"",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_whitespace_only_file():

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "whitespace.txt",
                b"   \n\t   ",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_invalid_utf8():

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "invalid.txt",
                b"\xff\xfe\xfa",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_oversized_upload():

    large_content = b"a" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/documents/",
        files={
            "file": (
                "large.txt",
                large_content,
                "text/plain"
            )
        }
    )

    assert response.status_code == 413


def test_markdown_upload():

    response = client.post(
        "/api/v1/documents/",
        files={"file": ("notes.md", b"# Hello", "text/plain")}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["content_type"] == "text/markdown"
    assert [c["content"] for c in data["chunks"]] == ["# Hello"]
