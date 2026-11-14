import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.chunking_service import ChunkingService
from app.services.document_service import DocumentProcessingError, DocumentService
from app.services.embedding_service import EmbeddingError

DIMS = 4
DOC_ID = "doc-1"


def make_embedding_service(side_effect=None):
    svc = MagicMock()
    svc.provider = "google"
    svc.model = "test-model"
    svc.dimensions = DIMS
    # default: vector i is [i, i, i, i] so chunk <-> vector pairing is checkable
    svc.embed_texts = AsyncMock(
        side_effect=side_effect
        or (lambda texts: [[float(i)] * DIMS for i in range(len(texts))])
    )
    return svc


def process(service, text):
    return asyncio.run(
        service.process_document(
            document_id=DOC_ID, filename="doc.md", content_type="text/markdown", text=text
        )
    )


@pytest.fixture
def embedding_service():
    return make_embedding_service()


@pytest.fixture
def service(embedding_service):
    return DocumentService(ChunkingService(chunk_size=10, overlap=2), embedding_service)


def test_valid_document(service, embedding_service):
    result = process(service, "hello world, this is a document")

    assert result.document_id == DOC_ID
    assert result.filename == "doc.md"
    assert result.status == "processed"
    assert result.chunk_count == len(result.chunks) > 0
    assert result.embedding_metadata.model_dump() == {
        "provider": "google", "model": "test-model", "dimensions": DIMS
    }
    assert all(len(c.embedding) == DIMS for c in result.chunks)
    embedding_service.embed_texts.assert_awaited_once()


def test_single_chunk_gets_one_vector(service, embedding_service):
    result = process(service, "short")

    assert result.chunk_count == 1
    assert result.chunks[0].content == "short"
    assert result.chunks[0].embedding == [0.0] * DIMS
    embedding_service.embed_texts.assert_awaited_once_with(["short"])


def test_multiple_chunks_associated_in_order(service, embedding_service):
    text = "abcdefghijklmnopqrstuvwxyz"
    result = process(service, text)

    assert result.chunk_count > 1
    # one batched call with every chunk's content, in order
    embedding_service.embed_texts.assert_awaited_once_with([c.content for c in result.chunks])
    for i, chunk in enumerate(result.chunks):
        assert chunk.chunk_index == i
        assert chunk.embedding == [float(i)] * DIMS
        assert text[chunk.start_offset:chunk.end_offset] == chunk.content


def test_vectors_follow_provider_order_not_content():
    # provider returns distinct vectors; each must land on the chunk at the same position
    vectors = [[1.0] * DIMS, [2.0] * DIMS, [3.0] * DIMS]
    chunking = MagicMock()
    chunking.chunk.return_value = ChunkingService(chunk_size=4, overlap=0).chunk("aaaabbbbcccc", DOC_ID)
    service = DocumentService(chunking, make_embedding_service(lambda texts: vectors))

    result = process(service, "aaaabbbbcccc")

    assert [(c.content, c.embedding[0]) for c in result.chunks] == [
        ("aaaa", 1.0), ("bbbb", 2.0), ("cccc", 3.0)
    ]


@pytest.mark.parametrize("text", ["", "   \n\t", None])
def test_empty_document_rejected(service, embedding_service, text):
    with pytest.raises(ValueError):
        process(service, text)
    embedding_service.embed_texts.assert_not_awaited()


def test_no_chunks_generated(embedding_service):
    chunking = MagicMock()
    chunking.chunk.return_value = []
    service = DocumentService(chunking, embedding_service)

    with pytest.raises(DocumentProcessingError, match="No chunks"):
        process(service, "some text")
    embedding_service.embed_texts.assert_not_awaited()


def test_embedding_api_failure_propagates_safely():
    error = EmbeddingError("Embedding provider error (status 503)")
    service = DocumentService(ChunkingService(), make_embedding_service(error))

    with pytest.raises(EmbeddingError, match="status 503"):
        process(service, "some text")


@pytest.mark.parametrize("returned", [0, 1, 5])
def test_embedding_count_mismatch_fails(returned):
    # mocked service bypasses EmbeddingService's own count check, so DocumentService must catch it
    service = DocumentService(
        ChunkingService(chunk_size=4, overlap=0),
        make_embedding_service(lambda texts: [[0.0] * DIMS] * returned),
    )

    with pytest.raises(DocumentProcessingError, match="count mismatch"):
        process(service, "aaaabbbb")  # 2 chunks
