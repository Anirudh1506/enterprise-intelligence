import fitz
import pytest

from app.services.extraction_service import ExtractionError, clean_text, extract_pages


def write_pdf(path, *pages, password=None):
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    if password:
        doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    else:
        doc.save(path)
    doc.close()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("a   b\t\tc", "a b c"),
        ("line  \n  next", "line\nnext"),
        ("p1\n\n\n\n\np2", "p1\n\np2"),
        ("win\r\nline", "win\nline"),
        ("knowl-\nedge", "knowledge"),
        ("nul\x00byte", "nulbyte"),
        ("ﬁle", "file"),  # NFKC ligature
        ("  padded  ", "padded"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_pdf_pages_numbered_and_empty_pages_dropped(tmp_path):
    path = tmp_path / "a.pdf"
    write_pdf(path, "First page", "", "Third page")

    assert extract_pages(path, "pdf") == [(1, "First page"), (3, "Third page")]


def test_text_file(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Title\n\n\n\nBody", encoding="utf-8")
    assert extract_pages(path, "md") == [(None, "# Title\n\nBody")]


def test_image_only_pdf_rejected(tmp_path):
    path = tmp_path / "blank.pdf"
    write_pdf(path, "")
    with pytest.raises(ExtractionError, match="No extractable text"):
        extract_pages(path, "pdf")


def test_encrypted_pdf_rejected(tmp_path):
    path = tmp_path / "locked.pdf"
    write_pdf(path, "secret", password="pw")
    with pytest.raises(ExtractionError, match="Password-protected"):
        extract_pages(path, "pdf")


def test_corrupt_pdf_rejected(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.7 garbage")
    with pytest.raises(ExtractionError):
        extract_pages(path, "pdf")


def test_invalid_utf8_rejected(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ExtractionError, match="UTF-8"):
        extract_pages(path, "txt")
