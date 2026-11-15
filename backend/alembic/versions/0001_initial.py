"""documents and chunks with pgvector embeddings

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-15
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("extracted_text", sa.Text()),
        sa.Column("page_count", sa.Integer()),
        sa.Column("char_count", sa.Integer()),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("embedding_model", sa.String(200)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED')",
            name="ck_documents_status",
        ),
    )
    op.create_index("ix_documents_sha256", "documents", ["sha256"])
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("end_offset", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_embedding_model", "chunks", ["embedding_model"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
