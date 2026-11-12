import pytest

from app.services.chunking_service import chunk_text

DOC_ID = "doc-1"


def contents(chunks):
    return [c.content for c in chunks]


def test_text_shorter_than_chunk_size():
    assert contents(chunk_text("hello", DOC_ID, chunkSize=10, overlap=2)) == ["hello"]


def test_text_exactly_chunk_size():
    assert contents(chunk_text("abcdefghij", DOC_ID, chunkSize=10, overlap=2)) == ["abcdefghij"]


def test_text_longer_than_chunk_size():
    chunks = chunk_text("a" * 25, DOC_ID, chunkSize=10, overlap=2)
    assert len(chunks) > 1
    assert all(len(c.content) <= 10 for c in chunks)


def test_zero_overlap_has_no_shared_characters():
    chunks = chunk_text("abcdefghij", DOC_ID, chunkSize=4, overlap=0)
    assert contents(chunks) == ["abcd", "efgh", "ij"]
    assert "".join(contents(chunks)) == "abcdefghij"


def test_overlap_shares_boundary_content():
    chunks = chunk_text("abcdefghij", DOC_ID, chunkSize=4, overlap=2)
    assert contents(chunks) == ["abcd", "cdef", "efgh", "ghij"]


def test_overlap_characters_match_exactly():
    text = "The quick brown fox jumps over the lazy dog"
    overlap = 5
    parts = contents(chunk_text(text, DOC_ID, chunkSize=12, overlap=overlap))

    for prev, nxt in zip(parts, parts[1:]):
        assert prev[-overlap:] == nxt[:overlap]

    # stitching chunks back (dropping each overlap) recovers the original text
    assert parts[0] + "".join(p[overlap:] for p in parts[1:]) == text


def test_chunk_metadata():
    text = "abcdefghij"
    chunks = chunk_text(text, DOC_ID, chunkSize=4, overlap=1)

    assert [c.chunk_index for c in chunks] == list(range(0, len(chunks)))
    assert all(c.document_id == DOC_ID for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(text)


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_text_raises(text):
    with pytest.raises(ValueError):
        chunk_text(text, DOC_ID, chunkSize=10, overlap=2)


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", DOC_ID, chunkSize=5, overlap=5)


def test_negative_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", DOC_ID, chunkSize=-1, overlap=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", DOC_ID, chunkSize=5, overlap=-1)


@pytest.mark.parametrize("size, overlap", [("5", 0), (5.0, 0), (5, "1"), (5, 1.0)])
def test_non_integer_params_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", DOC_ID, chunkSize=size, overlap=overlap)


@pytest.mark.parametrize("document_id", [None, "", "   ", 123])
def test_invalid_document_id_raises(document_id):
    with pytest.raises(ValueError):
        chunk_text("hello world", document_id, chunkSize=5, overlap=1)


def test_missing_document_id_raises():
    with pytest.raises(TypeError):
        chunk_text("hello world")
