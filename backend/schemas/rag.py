from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class SourceItem(BaseModel):
    score: float
    document_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]