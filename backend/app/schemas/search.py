from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    document_ids: Optional[list[str]] = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int]
    content: str
    score: float  # cosine similarity, higher is more relevant


class SearchResponse(BaseModel):
    query: str
    embedding_model: str
    results: list[SearchResult]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    document_ids: Optional[list[str]] = None


class Source(SearchResult):
    source_number: int  # the [n] the answer refers to
    cited: bool  # whether the answer actually references this source


class LLMInfo(BaseModel):
    provider: str
    model: str


class AskResponse(BaseModel):
    question: str
    answer: str
    insufficient_context: bool
    llm: LLMInfo
    sources: list[Source]
