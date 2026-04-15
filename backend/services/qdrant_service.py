import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from backend.core.config import settings
from backend.services.ollama_service import embed_query, embed_texts


client = QdrantClient(
    path=str(settings.qdrant_path),
    force_disable_check_same_thread=True,
)

_document_names_cache: list[str] | None = None


def _invalidate_document_cache() -> None:
    global _document_names_cache
    _document_names_cache = None


def collection_exists() -> bool:
    collections = client.get_collections().collections
    return any(collection.name == settings.collection_name for collection in collections)


def ensure_collection(vector_size: int) -> None:
    if collection_exists():
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def delete_document_points(document_name: str) -> None:
    if not collection_exists():
        return

    client.delete(
        collection_name=settings.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_name",
                        match=models.MatchValue(value=document_name),
                    )
                ]
            )
        ),
        wait=True,
    )

    _invalidate_document_cache()


def _normalize_lookup_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    text = text.replace("_", " ").replace(":", " ").replace(".", " ")
    text = text.replace("ё", "е")
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"^[0-9]{4}\.[0-9]{5}v[0-9]\s*-\s*", "", text)
    text = re.sub(r"[\"'`“”‘’(){}\[\],;!?/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_embedding_text(document_name: str, chunk: dict[str, Any]) -> str:
    text = (chunk.get("text") or "").strip()
    page = chunk.get("page")
    chunk_id = chunk.get("chunk_id") or ""

    prefix_parts = [f"Document: {document_name}"]
    if page is not None:
        prefix_parts.append(f"Page: {page}")
    if chunk_id:
        prefix_parts.append(f"Chunk: {chunk_id}")

    prefix = "\n".join(prefix_parts)
    return f"{prefix}\n\n{text}".strip()


def _extract_title_candidates(question: str) -> list[str]:
    candidates: list[str] = []
    q = (question or "").strip()

    if not q:
        return candidates

    quoted = re.findall(r'"([^"]{5,})"|\'([^\']{5,})\'', q)
    for pair in quoted:
        for item in pair:
            item = item.strip()
            if item:
                candidates.append(item)

    if len(q) >= 20:
        candidates.append(q)

    unique: list[str] = []
    seen = set()

    for item in candidates:
        key = _normalize_lookup_text(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def _get_document_names() -> list[str]:
    global _document_names_cache

    if _document_names_cache is not None:
        return _document_names_cache

    if not collection_exists():
        _document_names_cache = []
        return _document_names_cache

    names: set[str] = set()
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            document_name = payload.get("document_name")
            if document_name:
                names.add(str(document_name))

        if next_offset is None:
            break
        offset = next_offset

    _document_names_cache = sorted(names)
    return _document_names_cache


def _find_matching_documents(title: str, max_docs: int = 3) -> list[str]:
    title_norm = _normalize_lookup_text(title)
    if not title_norm:
        return []

    title_tokens = set(title_norm.split())
    matches: list[tuple[float, str]] = []

    for document_name in _get_document_names():
        doc_norm = _normalize_lookup_text(document_name)
        doc_tokens = set(doc_norm.split())

        score = 0.0

        if title_norm in doc_norm:
            score += 100.0
        elif doc_norm in title_norm:
            score += 80.0
        else:
            overlap = len(title_tokens & doc_tokens)
            if overlap >= max(4, min(8, len(title_tokens) // 2)):
                score += float(overlap)

        if score > 0:
            matches.append((score, document_name))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [document_name for _, document_name in matches[:max_docs]]


def _search_within_document(
    document_name: str,
    question: str,
    limit: int,
    title_candidates: list[str],
) -> list[dict[str, Any]]:
    query_vector = embed_query(question)

    result = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_name",
                    match=models.MatchValue(value=document_name),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    hits: list[dict[str, Any]] = []

    for point in result.points:
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        page = payload.get("page")
        chunk_id = payload.get("chunk_id")

        bonus = 0.0
        if page == 1:
            bonus += 1.0
        if str(chunk_id) == "chunk_0":
            bonus += 1.0

        text_norm = _normalize_lookup_text(text)
        for candidate in title_candidates:
            candidate_norm = _normalize_lookup_text(candidate)
            if candidate_norm and candidate_norm in text_norm:
                bonus += 2.0
                break

        hits.append(
            {
                "score": float(point.score or 0.0) + bonus,
                "document_name": payload.get("document_name", ""),
                "page": page,
                "chunk_id": chunk_id,
                "section_index": payload.get("section_index"),
                "text": text,
                "match_type": "document_filtered",
            }
        )

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:limit]


def upsert_document_chunks(document_name: str, chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0

    embedding_texts = [
        _build_embedding_text(document_name=document_name, chunk=chunk)
        for chunk in chunks
    ]
    vectors = embed_texts(embedding_texts)

    ensure_collection(vector_size=len(vectors[0]))
    delete_document_points(document_name)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_name": document_name,
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("chunk_id"),
                    "section_index": chunk.get("section_index"),
                    "text": chunk.get("text"),
                },
            )
        )

    client.upsert(
        collection_name=settings.collection_name,
        points=points,
        wait=True,
    )

    _invalidate_document_cache()
    return len(points)


def search_similar_chunks(question: str, limit: int = 3) -> list[dict[str, Any]]:
    if not collection_exists():
        return []

    title_candidates = _extract_title_candidates(question)

    # 1) Быстрый путь: ищем совпадающий document_name и ищем только внутри него
    matched_docs: list[str] = []
    for candidate in title_candidates[:2]:
        matched_docs.extend(_find_matching_documents(candidate, max_docs=3))

    # dedupe matched docs
    unique_docs: list[str] = []
    seen_docs = set()
    for doc in matched_docs:
        if doc not in seen_docs:
            seen_docs.add(doc)
            unique_docs.append(doc)

    if unique_docs:
        results: list[dict[str, Any]] = []
        for document_name in unique_docs[:3]:
            results.extend(
                _search_within_document(
                    document_name=document_name,
                    question=question,
                    limit=max(limit * 2, 6),
                    title_candidates=title_candidates,
                )
            )

        deduped: list[dict[str, Any]] = []
        seen = set()

        results.sort(key=lambda x: x["score"], reverse=True)

        for item in results:
            key = (
                item.get("document_name"),
                item.get("page"),
                item.get("chunk_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break

        if deduped:
            return deduped

    # 2) Fallback: обычный semantic search по всей коллекции
    query_vector = embed_query(question)

    result = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=max(limit * 3, 10),
        with_payload=True,
        with_vectors=False,
    )

    hits = []
    for point in result.points:
        payload = point.payload or {}
        hits.append(
            {
                "score": float(point.score or 0.0),
                "document_name": payload.get("document_name", ""),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "section_index": payload.get("section_index"),
                "text": payload.get("text", ""),
                "match_type": "semantic",
            }
        )

    return hits[:limit]


def list_documents() -> list[dict[str, Any]]:
    if not collection_exists():
        return []

    aggregated: dict[str, dict[str, Any]] = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            document_name = payload.get("document_name")

            if not document_name:
                continue

            if document_name not in aggregated:
                aggregated[document_name] = {
                    "document_name": document_name,
                    "chunk_count": 0,
                    "pages": set(),
                }

            aggregated[document_name]["chunk_count"] += 1

            page = payload.get("page")
            if isinstance(page, int):
                aggregated[document_name]["pages"].add(page)

        if next_offset is None:
            break

        offset = next_offset

    documents = []
    for item in aggregated.values():
        documents.append(
            {
                "document_name": item["document_name"],
                "chunk_count": item["chunk_count"],
                "pages": sorted(item["pages"]),
            }
        )

    documents.sort(key=lambda x: x["document_name"].lower())
    return documents