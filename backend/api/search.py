from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.embeddings import get_embedding
from backend.services.vector_store import search_similar_chunks

router = APIRouter(prefix="/search", tags=["semantic-search"])


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    k: int = Field(default=3, ge=1, le=5, description="Сколько фрагментов вернуть")


class SearchResultItem(BaseModel):
    text: str
    score: float
    document_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None


class SearchResponse(BaseModel):
    question: str
    k: int
    results: list[SearchResultItem]


@router.post("", response_model=SearchResponse)
def semantic_search(payload: SearchRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не должен быть пустым")

    try:
        query_vector = get_embedding(question)
        results = search_similar_chunks(query_vector=query_vector, limit=payload.k)

        return SearchResponse(
            question=question,
            k=payload.k,
            results=[SearchResultItem(**item) for item in results],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка семантического поиска: {str(e)}")