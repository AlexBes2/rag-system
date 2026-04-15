from typing import List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=20)


class SourceItem(BaseModel):
    document_name: str
    pages: List[int] = []
    chunk_ids: List[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    document_name: str
    chunk_count: int
    pages: List[int] = []


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]