import json
import os
import re
from functools import lru_cache
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.qdrant_service import (
    find_documents_containing_text,
    search_similar_chunks,
)

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

TITLE_LOOKUP_MARKERS = [
    "documents that contain the title",
    "documents containing the title",
    "documents with the title",
    "files that contain the title",
    "files containing the title",
    "files with the title",
    "find all documents",
    "find all files",
]

TEXT_LOOKUP_MARKERS = [
    "documents that contain the text",
    "documents containing the text",
    "documents with the text",
    "files that contain the text",
    "files containing the text",
    "files with the text",
    "file with text",
    "files with text",
    "find file with text",
    "find files with text",
    "find files containing text",
    "find files that contain text",
    "contain the text",
    "contains the text",
    "containing the text",
]

CONTEXT_ACRONYMS = (
    "AI",
    "BERT",
    "FinBERT",
    "LLM",
    "LLMs",
    "LVLM",
    "LVLMs",
    "MLLM",
    "MLLMs",
    "RAG",
    "VLM",
    "VLMs",
    "VQA",
)

GLUED_WORDS = {
    "a",
    "ablations",
    "abstract",
    "accuracy",
    "across",
    "again",
    "against",
    "all",
    "agent",
    "agents",
    "aim",
    "aims",
    "already",
    "also",
    "an",
    "and",
    "answer",
    "architecture",
    "as",
    "at",
    "based",
    "benchmark",
    "between",
    "by",
    "candidate",
    "can",
    "chunk",
    "chunks",
    "components",
    "completely",
    "con",
    "condition",
    "confirm",
    "context",
    "contribution",
    "contributions",
    "data",
    "dataset",
    "dense",
    "document",
    "documents",
    "dominant",
    "discards",
    "effect",
    "embedding",
    "embeddings",
    "eliminating",
    "external",
    "factual",
    "for",
    "formalize",
    "from",
    "full",
    "fundamental",
    "generate",
    "generation",
    "has",
    "hidden",
    "improve",
    "in",
    "index",
    "ineffect",
    "information",
    "inference",
    "internal",
    "introduces",
    "is",
    "it",
    "knowledge",
    "large",
    "layer",
    "layers",
    "learning",
    "loss",
    "main",
    "method",
    "model",
    "models",
    "native",
    "of",
    "on",
    "only",
    "approximate",
    "paradigm",
    "pipeline",
    "processed",
    "projection",
    "propose",
    "query",
    "quality",
    "rag",
    "reasoning",
    "redundancy",
    "retrieval",
    "retrieve",
    "search",
    "semantic",
    "sessed",
    "standard",
    "state",
    "states",
    "system",
    "systematic",
    "text",
    "that",
    "the",
    "then",
    "this",
    "three",
    "to",
    "two",
    "typically",
    "understanding",
    "via",
    "vector",
    "while",
    "with",
}

SNIPPET_GLUED_WORDS = {
    "adaptation",
    "aluminum",
    "annotated",
    "alone",
    "called",
    "can",
    "classify",
    "closest",
    "compared",
    "compare",
    "datasets",
    "document",
    "expert",
    "financial",
    "finbert",
    "forecasting",
    "including",
    "indexed",
    "investigate",
    "learn",
    "learning",
    "lexicon",
    "lightweight",
    "matched",
    "llm",
    "models",
    "news",
    "numerical",
    "predicted",
    "prices",
    "question",
    "qwen",
    "scores",
    "sentences",
    "sentiment",
    "series",
    "techniques",
    "time",
    "use",
    "using",
    "we",
}

WORD_LIST_PATHS = (
    "/usr/share/dict/words",
    "/usr/share/dict/web2",
)

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
    snippet: str | None = None
    score: float | None = None
    match_type: str | None = None


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
        {"question": question, "limit": k},
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


def extract_score(item: dict[str, Any]) -> float | None:
    score = item.get("score")
    if score is None:
        return None

    try:
        return round(float(score), 4)
    except (TypeError, ValueError):
        return None


def extract_match_type(item: dict[str, Any]) -> str | None:
    match_type = item.get("match_type")
    if match_type is None:
        return None
    return str(match_type)


def build_file_url(document_name: str) -> str:
    return f"{BACKEND_BASE_URL}/files/{quote(document_name)}"


def build_snippet(text: str, max_chars: int = 360) -> str | None:
    text = improve_context_readability(text, min_glued_len=6)
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return None
    if has_unreadable_glued_artifacts(text):
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_source_item(item: dict[str, Any]) -> SourceItem:
    document_name = extract_document_name(item)
    return SourceItem(
        document_name=document_name,
        page=extract_page(item),
        chunk_id=extract_chunk_id(item),
        file_url=build_file_url(document_name),
        snippet=build_snippet(extract_text(item)),
        score=extract_score(item),
        match_type=extract_match_type(item),
    )


def extract_quoted_title(question: str) -> str | None:
    quoted = re.findall(r'"([^"]{3,})"|\'([^\']{3,})\'', question or "")
    for pair in quoted:
        for item in pair:
            item = item.strip()
            if item:
                return item
    return None


def is_title_lookup_question(question: str) -> bool:
    normalized = normalize_for_match(question)
    if "title" not in normalized:
        return False
    if extract_quoted_title(question) is None:
        return False
    return any(marker in normalized for marker in TITLE_LOOKUP_MARKERS)


def is_text_lookup_question(question: str) -> bool:
    normalized = normalize_for_match(question)
    if "text" not in normalized:
        return False
    if extract_quoted_title(question) is None:
        return False
    return any(marker in normalized for marker in TEXT_LOOKUP_MARKERS)


def lookup_target_label(question: str) -> str:
    normalized = normalize_for_match(question)
    if "file" in normalized:
        return "file"
    return "document"


def build_title_source(document_name: str) -> SourceItem:
    return SourceItem(
        document_name=document_name,
        page=None,
        chunk_id=None,
        file_url=build_file_url(document_name),
        snippet=f"Document title match: {document_name}",
        score=1.0,
        match_type="title",
    )


@lru_cache(maxsize=1)
def glued_word_inventory() -> frozenset[str]:
    words = set(GLUED_WORDS) | set(SNIPPET_GLUED_WORDS)

    for path in WORD_LIST_PATHS:
        try:
            with open(path, encoding="utf-8", errors="ignore") as file:
                for line in file:
                    word = line.strip().lower()
                    if re.fullmatch(r"[a-z]{2,18}", word):
                        words.add(word)
        except OSError:
            continue

    words.update({"a", "i"})
    return frozenset(words)


@lru_cache(maxsize=1)
def preferred_glued_words() -> frozenset[str]:
    return frozenset(set(GLUED_WORDS) | set(SNIPPET_GLUED_WORDS))


@lru_cache(maxsize=1)
def max_glued_word_len() -> int:
    return max(len(word) for word in glued_word_inventory())


def has_unreadable_glued_artifacts(text: str) -> bool:
    sample = re.sub(r"https?://\S+|\S+@\S+", " ", text or "")
    suspicious_tokens = re.findall(r"\b[a-z]{28,}\b", sample)
    if not suspicious_tokens:
        return False

    known_words = glued_word_inventory()
    return any(token.lower() not in known_words for token in suspicious_tokens)


def build_lookup_miss_response(
    question: str,
    quoted_text: str,
    lookup_label: str,
    target_label: str,
    k: int,
) -> QueryResponse:
    answer = (
        f'No uploaded {target_label}s contain the exact {lookup_label} '
        f'"{quoted_text}".'
    )

    if lookup_label != "text":
        return QueryResponse(answer=answer, sources=[])

    semantic_hits = run_search(quoted_text, max(k, 5))
    if not semantic_hits:
        return QueryResponse(answer=answer, sources=[])

    sources: list[SourceItem] = []
    seen_documents: set[str] = set()

    for item in semantic_hits:
        document_name = extract_document_name(item)
        if document_name in seen_documents:
            continue
        seen_documents.add(document_name)
        sources.append(build_source_item(item))
        if len(sources) >= max(k, 3):
            break

    if not sources:
        return QueryResponse(answer=answer, sources=[])

    lines = "\n".join(
        f"- {source.document_name}"
        for source in sources
    )
    return QueryResponse(
        answer=(
            f"{answer}\n\nClosest semantic matches, not exact text matches:\n{lines}"
        ),
        sources=sources,
    )


def answer_title_lookup(question: str, k: int) -> QueryResponse | None:
    if is_title_lookup_question(question):
        lookup_label = "title"
    elif is_text_lookup_question(question):
        lookup_label = "text"
    else:
        return None

    quoted_text = extract_quoted_title(question)
    if not quoted_text:
        return None

    target_label = lookup_target_label(question)
    matches = find_documents_containing_text(quoted_text, max_docs=50)
    if not matches:
        return build_lookup_miss_response(
            question=question,
            quoted_text=quoted_text,
            lookup_label=lookup_label,
            target_label=target_label,
            k=k,
        )

    lines = "\n".join(
        f"- {extract_document_name(item)}"
        for item in matches
    )
    relation = "containing"
    if lookup_label == "text" and any(
        (extract_match_type(item) or "").startswith("near_")
        for item in matches
    ):
        relation = "matching"

    return QueryResponse(
        answer=(
            f'Found {len(matches)} uploaded {target_label}(s) {relation} the {lookup_label} '
            f'"{quoted_text}":\n{lines}'
        ),
        sources=[
            build_title_source(extract_document_name(item))
            if extract_match_type(item) == "title"
            else build_source_item(item)
            for item in matches
        ],
    )


def split_glued_token(token: str) -> str:
    if len(token) < 6 or not token.isalpha():
        return token

    lower_token = token.lower()
    if lower_token == "ineffect":
        return "In effect" if token[0].isupper() else "in effect"

    words = glued_word_inventory()
    preferred_words = preferred_glued_words()

    if lower_token in words:
        return token

    @lru_cache(maxsize=None)
    def search(position: int) -> tuple[tuple[int, int, int], tuple[str, ...]] | None:
        if position >= len(lower_token):
            return (0, 0, 0), ()

        best: tuple[tuple[int, int, int], tuple[str, ...]] | None = None
        end_limit = min(len(lower_token), position + max_glued_word_len())

        for end in range(end_limit, position, -1):
            part = lower_token[position:end]
            if part not in words:
                continue

            rest = search(end)
            if rest is None:
                continue

            rest_score, rest_parts = rest
            score = (
                rest_score[0] + (0 if part in preferred_words else 1),
                rest_score[1] + (1 if len(part) <= 2 else 0),
                rest_score[2] + 1,
            )
            candidate = (score, (part, *rest_parts))
            if best is None or candidate[0] < best[0]:
                best = candidate

        return best

    result = search(0)
    if result is None:
        return token

    _, parts = result
    if len(parts) < 2:
        return token

    split_text = " ".join(parts)
    if token[0].isupper():
        split_text = split_text[:1].upper() + split_text[1:]

    return split_text


def split_glued_words(text: str, min_len: int = 8) -> str:
    return re.sub(
        rf"\b[A-Za-z]{{{min_len},}}\b",
        lambda match: split_glued_token(match.group(0)),
        text,
    )


def improve_context_readability(text: str, min_glued_len: int = 8) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\((?=[A-Za-z0-9])", " (", text)
    text = re.sub(r"(?<=[A-Za-z0-9])\)(?=[A-Za-z])", ") ", text)
    text = re.sub(r"(?<=[,;:])(?=\S)", " ", text)
    text = re.sub(r"\s+", " ", text)

    for acronym in sorted(CONTEXT_ACRONYMS, key=len, reverse=True):
        plural_guard = "(?!s)" if acronym in {"LLM", "LVLM", "MLLM", "VLM"} else ""
        text = re.sub(
            rf"\b({re.escape(acronym)}){plural_guard}(?=[a-z])",
            r"\1 ",
            text,
        )

    text = split_glued_words(text, min_len=min_glued_len)
    text = re.sub(r"\b[Ff]in\s+BERT\b", "FinBERT", text)
    text = re.sub(r"\bIn effect\b", "In effect", text)
    return text.strip()


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    context_parts = []

    for index, item in enumerate(chunks, start=1):
        text = improve_context_readability(extract_text(item))
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
Отвечай только по контексту ниже и не добавляй факты извне.
Отвечай на языке вопроса.
Для вопросов про authors/авторов перечисляй только имена авторов, убирая номера аффилиаций и служебные символы.
Если вопрос про contribution/main contribution/вклад, не считай имена авторов, университеты и email вкладом статьи.
Для вопросов summary/problem/method/lessons можно делать краткий вывод из abstract, introduction, contributions и conclusion, если это поддержано контекстом.
Если релевантной информации в контексте нет, честно скажи: "В загруженных документах точного ответа не найдено."

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
        "options": {
            "temperature": 0,
            "top_p": 0.9,
        },
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

        sources.append(build_source_item(item))

    return sources


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    title_lookup_response = answer_title_lookup(request.question, request.k)
    if title_lookup_response is not None:
        return title_lookup_response

    chunks = run_search(request.question, request.k)

    if not chunks:
        return QueryResponse(
            answer="В загруженных документах точного ответа не найдено.",
            sources=[],
        )

    prompt = build_prompt(request.question, chunks)
    answer = call_ollama(prompt)

    if is_answer_not_found(answer):
        return QueryResponse(
            answer=answer,
            sources=[],
        )

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

        sources.append(build_source_item(item))

    return QueryResponse(
        answer=answer or "В загруженных документах точного ответа не найдено.",
        sources=sources,
    )

@router.post("/debug-search")
async def debug_search(payload: QueryRequest):
    results = search_similar_chunks(question=payload.question, limit=payload.k)
    return {"results": results}
