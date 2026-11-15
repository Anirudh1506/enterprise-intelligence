import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileStorage:
    """Stores original uploads on local disk, named by document id (never by user filename)."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def save(self, document_id: str, extension: str, data: bytes) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{document_id}{extension}"
        path.write_bytes(data)
        return path

    def delete(self, path: str) -> None:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as e:
            # a leftover file is harmless; don't fail the request over it
            logger.warning("Could not delete stored file error=%s", type(e).__name__)
