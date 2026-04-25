import json
import os
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.qdrant_service import search_similar_chunks

router = APIRouter(tags=["query"])

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    os.getenv("OLLAMA_LLM_MODEL", "llama3"),
)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")

ANSWER_NOT_FOUND_MARKERS = [
    "в загруженных документах точного ответа не найдено",
    "точного ответа не найдено",
    "не найдено",
    "not found in the uploaded documents",
    "exact answer was not found",
    "exact answer not found",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in",
    "on", "for", "with", "by", "from", "this", "that", "these", "those", "it",
    "as", "at", "be", "if", "then", "than", "into", "about", "what", "which",
    "who", "whom", "when", "where", "why", "how",
    "это", "как", "что", "где", "когда", "или", "для", "при", "над", "под",
    "без", "есть", "был", "была", "были", "его", "ее", "её", "они", "она",
    "оно", "так", "тоже", "если", "только", "ли", "не", "нет",
    "це", "що", "як", "де", "коли", "для", "при", "над", "під", "без",
    "є", "був", "була", "були", "його", "її", "вони", "вона", "воно", "не",
}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class SourceItem(BaseModel):
    document_name: str
    page: int | None = None
    chunk_id: str | int | None = None
    file_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


def run_search(question: str, k: int) -> list[dict[str, Any]]:
    if search_similar_chunks is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Не найдена функция search_similar_chunks. "
                "Проверь импорт в backend/api/query.py"
            ),
        )

    attempts = [
        {"question": question, "k": k},
        {"query": question, "k": k},
        {"query_text": question, "k": k},
    ]

    for kwargs in attempts:
        try:
            result = search_similar_chunks(**kwargs)
            break
        except TypeError:
            result = None
    else:
        try:
            result = search_similar_chunks(question, k)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка семантического поиска: {str(e)}",
            ) from e

    if isinstance(result, dict):
        result = result.get("results", [])

    if not isinstance(result, list):
        return []

    normalized = []
    for item in result:
        if isinstance(item, dict):
            normalized.append(item)

    return normalized


def extract_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}

    return (
        item.get("text")
        or item.get("page_content")
        or metadata.get("text")
        or ""
    ).strip()


def extract_document_name(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}

    return (
        metadata.get("document_name")
        or metadata.get("filename")
        or metadata.get("source")
        or item.get("document_name")
        or item.get("filename")
        or item.get("source")
        or "unknown"
    )


def extract_page(item: dict[str, Any]) -> int | None:
    metadata = item.get("metadata") or {}
    page = metadata.get("page", item.get("page"))

    if page is None:
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def extract_chunk_id(item: dict[str, Any]) -> str | int | None:
    metadata = item.get("metadata") or {}

    return (
        metadata.get("chunk_id")
        or metadata.get("id")
        or item.get("chunk_id")
        or item.get("id")
    )


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    context_parts = []

    for index, item in enumerate(chunks, start=1):
        text = extract_text(item)
        document_name = extract_document_name(item)
        page = extract_page(item)

        if not text:
            continue

        header = f"[Источник {index}] файл: {document_name}"
        if page is not None:
            header += f", страница: {page}"

        context_parts.append(f"{header}\n{text}")

    context = "\n\n".join(context_parts).strip()

    return f"""
Ты помощник для RAG-системы.
Отвечай только по контексту ниже.
Если в контексте нет точного ответа, честно скажи: "В загруженных документах точного ответа не найдено."

Вопрос:
{question}

Контекст:
{context}

Ответ:
""".strip()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    request = Request(
        url=f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            return (data.get("response") or "").strip()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка запроса к Ollama: {str(e)}",
        ) from e


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("ё", "е")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


def is_answer_not_found(answer: str) -> bool:
    normalized = normalize_for_match(answer)
    return any(marker in normalized for marker in ANSWER_NOT_FOUND_MARKERS)


def extract_content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ0-9]{3,}", text.lower())
    return {word for word in words if word not in STOPWORDS}


def split_answer_into_segments(answer: str) -> list[str]:
    raw_parts = re.split(
        r"\n+|(?:^\s*[-*•]\s+)|;\s+|(?<=[.!?])\s+",
        answer,
        flags=re.MULTILINE,
    )

    segments: list[str] = []

    for part in raw_parts:
        cleaned = part.strip(" \t\r\n-*•")
        if len(cleaned) < 3:
            continue
        segments.append(cleaned)

    return segments


def extract_answer_signals(answer: str) -> list[str]:
    signals: list[str] = []
    cleaned = answer.strip()

    if not cleaned:
        return signals

    normalized_full = normalize_for_match(cleaned)
    if normalized_full:
        signals.append(normalized_full)

    segments = split_answer_into_segments(cleaned)
    for segment in segments:
        normalized_segment = normalize_for_match(segment)
        if len(normalized_segment) >= 3:
            signals.append(normalized_segment)

    quoted_parts = re.findall(r"[\"“”'«»](.*?)[\"“”'«»]", cleaned)
    for part in quoted_parts:
        part = normalize_for_match(part)
        if len(part) >= 3:
            signals.append(part)

    code_like_patterns = [
        r"\b[a-zA-Zа-яА-Я0-9]{2,}(?:[-/][a-zA-Zа-яА-Я0-9]{2,})+\b",
        r"\b[a-zA-Z]{2,}\d{2,}[a-zA-Z0-9-]*\b",
        r"\b\d{4,}\b",
        r"\bUA\d{6,}\b",
    ]
    for pattern in code_like_patterns:
        for match in re.findall(pattern, cleaned, flags=re.IGNORECASE):
            match = normalize_for_match(match)
            if len(match) >= 3:
                signals.append(match)

    unique_signals: list[str] = []
    seen = set()

    for signal in signals:
        signal = signal.strip(" .,:;!-")
        if len(signal) < 3:
            continue
        if signal in seen:
            continue
        seen.add(signal)
        unique_signals.append(signal)

    unique_signals.sort(key=len, reverse=True)
    return unique_signals


def chunk_supports_answer(chunk_text: str, answer: str) -> bool:
    if not chunk_text or not answer:
        return False

    if is_answer_not_found(answer):
        return False

    normalized_chunk = normalize_for_match(chunk_text)
    normalized_answer = normalize_for_match(answer)

    if normalized_answer and normalized_answer in normalized_chunk:
        return True

    signals = extract_answer_signals(answer)
    for signal in signals:
        if signal in normalized_chunk:
            return True

    chunk_words = extract_content_words(chunk_text)
    if not chunk_words:
        return False

    answer_segments = split_answer_into_segments(answer)
    if not answer_segments:
        answer_segments = [answer]

    for segment in answer_segments:
        normalized_segment = normalize_for_match(segment)
        if len(normalized_segment) >= 3 and normalized_segment in normalized_chunk:
            return True

        segment_words = extract_content_words(segment)
        if not segment_words:
            continue

        overlap = segment_words & chunk_words
        overlap_ratio = len(overlap) / max(len(segment_words), 1)

        if len(segment_words) <= 3:
            if len(overlap) >= max(1, len(segment_words)):
                return True
        else:
            if overlap_ratio >= 0.5:
                return True

    return False


def build_sources_from_supported_chunks(
    search_results: list[dict[str, Any]],
    answer: str,
) -> list[SourceItem]:
    sources: list[SourceItem] = []
    seen_pages = set()

    for item in search_results:
        chunk_text = extract_text(item)
        if not chunk_supports_answer(chunk_text, answer):
            continue

        document_name = extract_document_name(item)
        page = extract_page(item)
        chunk_id = extract_chunk_id(item)

        key = (document_name, page)
        if key in seen_pages:
            continue
        seen_pages.add(key)

        file_url = f"{BACKEND_BASE_URL}/files/{quote(document_name)}"

        sources.append(
            SourceItem(
                document_name=document_name,
                page=page,
                chunk_id=chunk_id,
                file_url=file_url,
            )
        )

    return sources


# @router.post("/query", response_model=QueryResponse)
# def query_documents(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не должен быть пустым")

    search_results = run_search(question=question, k=payload.k)

    if not search_results:
        return QueryResponse(
            answer="Не удалось найти подходящие фрагменты в документах.",
            sources=[],
        )

    prompt = build_prompt(question=question, chunks=search_results)
    answer = call_ollama(prompt)
    answer = answer or "Модель не вернула текст ответа."

    sources = build_sources_from_supported_chunks(
        search_results=search_results,
        answer=answer,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
    )

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    chunks = run_search(request.question, request.k)

    if not chunks:
        return QueryResponse(
            answer="В загруженных документах точного ответа не найдено.",
            sources=[],
        )

    prompt = build_prompt(request.question, chunks)
    answer = call_ollama(prompt)

    sources: list[SourceItem] = []
    seen = set()

    for item in chunks:
        document_name = extract_document_name(item)
        page = extract_page(item)
        chunk_id = extract_chunk_id(item)

        key = (document_name, page, chunk_id)
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            SourceItem(
                document_name=document_name,
                page=page,
                chunk_id=chunk_id,
                file_url=None,
            )
        )

    return QueryResponse(
        answer=answer or "В загруженных документах точного ответа не найдено.",
        sources=sources,
    )

@router.post("/debug-search")
async def debug_search(payload: QueryRequest):
    results = search_similar_chunks(question=payload.question, limit=payload.k)
    return {"results": results}
