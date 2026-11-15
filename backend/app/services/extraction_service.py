import re
import unicodedata
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


class ExtractionError(Exception):
    """Raised when text cannot be extracted. Message is safe to show to clients."""


# (page_number, text); page_number is None for formats without pages
Page = tuple[Optional[int], str]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(path: Path, file_type: str) -> list[Page]:
    """Extract and clean text. Empty pages are dropped; raises if nothing remains."""
    if file_type == "pdf":
        raw = _extract_pdf(path)
    elif file_type in ("txt", "md"):
        try:
            raw = [(None, path.read_text(encoding="utf-8"))]
        except UnicodeDecodeError:
            raise ExtractionError("File must be UTF-8 encoded") from None
    else:
        raise ExtractionError(f"Unsupported file type: {file_type}")

    pages = [(n, clean_text(t)) for n, t in raw]
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise ExtractionError(
            "No extractable text found (scanned/image-only PDFs need OCR, which is not supported)"
        )
    return pages


def _extract_pdf(path: Path) -> list[Page]:
    try:
        doc = fitz.open(path)
    except Exception:
        raise ExtractionError("File is not a valid PDF") from None
    with doc:
        if doc.needs_pass:
            raise ExtractionError("Password-protected PDFs are not supported")
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
