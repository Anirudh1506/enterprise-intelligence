from fastapi import APIRouter, Depends

from app.api.deps import get_ask_service, get_search_service
from app.schemas.search import AskRequest, AskResponse, SearchRequest, SearchResponse
from app.services.ask_service import AskService
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest, service: SearchService = Depends(get_search_service)):
    """Semantic search over processed document chunks (cosine similarity)."""
    results = await service.search(request.query, request.top_k, request.document_ids)
    return SearchResponse(
        query=request.query,
        embedding_model=service.embedding_service.model_id,
        results=results,
    )


@router.post("/ask", response_model=AskResponse, tags=["Intelligence"])
async def ask(request: AskRequest, service: AskService = Depends(get_ask_service)):
    """Answer a question grounded in retrieved document passages, with citations."""
    return await service.ask(request.question, request.top_k, request.document_ids)
