import os
import tempfile

# must run before app modules are imported (settings + engine are created at import)
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://localhost:5432/enterprise_intelligence_test"
)
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="ei-test-uploads-")

import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.session import Base, SessionLocal, engine  # noqa: E402
import app.models.document  # noqa: E402,F401


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


DB_AVAILABLE = _db_available()


@pytest.fixture(scope="session")
def _schema():
    if not DB_AVAILABLE:
        pytest.skip("PostgreSQL test database not available (see README: Running tests)")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db(_schema):
    """Clean database per test."""
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE chunks, documents CASCADE"))
    session = SessionLocal()
    yield session
    session.close()
