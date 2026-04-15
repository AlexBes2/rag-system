from fastapi import APIRouter, HTTPException

from backend.schemas.rag import AskRequest, AskResponse
from backend.services.rag_service import ask_question

router = APIRouter(tags=["RAG"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        return ask_question(question=request.question, k=request.k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc