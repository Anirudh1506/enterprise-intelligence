
import pytest

from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)


@pytest.fixture
def mock_document_service():
    with patch(
        "app.api.routes.documents.DocumentService"
    ) as mock:

        service = mock.return_value

        service.process_document = AsyncMock(
            return_value={
                "document_id": "test-document-id",
                "filename": "test.txt",
                "content_type": "text/plain",
                "status": "processed",
                "created_at": datetime.now(timezone.utc),
                "chunks": []
            }
        )

        yield service


def test_successful_upload(mock_document_service):

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

    assert data["document_id"] == "test-document-id"
    assert data["filename"] == "test.txt"
    assert data["content_type"] == "text/plain"

    mock_document_service.process_document.assert_awaited_once()


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


def test_upload_with_real_service():

    response = client.post(
        "/api/v1/documents/",
        files={"file": ("notes.md", b"# Hello", "text/plain")}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["content_type"] == "text/markdown"
    assert [c["content"] for c in data["chunks"]] == ["# Hello"]
