from __future__ import annotations

from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from backend.core.config import QDRANT_COLLECTION, QDRANT_PATH
from backend.services.ollama_client import embed_text

client = QdrantClient(path=QDRANT_PATH)

COLLECTION_NAME = QDRANT_COLLECTION


def ensure_collection_exists(vector_size: int) -> None:
    collections = client.get_collections().collections
    existing_names = {collection.name for collection in collections}

    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def upsert_chunks(chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0

    points: list[PointStruct] = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        embedding = chunk.get("embedding")
        if not embedding:
            embedding = embed_text(text)

        metadata = chunk.get("metadata", {}) or {}

        chunk_id = (
            chunk.get("chunk_id")
            or metadata.get("chunk_id")
            or str(uuid4())
        )

        payload = {
            "text": text,
            "chunk_id": chunk_id,
            "document_name": (
                chunk.get("document_name")
                or metadata.get("document_name")
                or metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("source")
            ),
            "page": chunk.get("page", metadata.get("page")),
            "metadata": metadata,
        }

        points.append(
            PointStruct(
                id=str(chunk_id),
                vector=embedding,
                payload=payload,
            )
        )

    if not points:
        return 0

    vector_size = len(points[0].vector)
    ensure_collection_exists(vector_size)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


def _extract_document_name(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return (
        payload.get("document_name")
        or payload.get("file_name")
        or payload.get("filename")
        or payload.get("source")
        or metadata.get("document_name")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("source")
    )


def _extract_page(payload: dict[str, Any]) -> int | None:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    page = payload.get("page", metadata.get("page"))

    if page is None:
        return None

    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _extract_chunk_id(payload: dict[str, Any], point_id: Any) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    value = payload.get("chunk_id") or metadata.get("chunk_id") or point_id
    return str(value)


def _extract_text(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return (
        payload.get("text")
        or payload.get("chunk_text")
        or metadata.get("text")
        or metadata.get("chunk_text")
        or ""
    )


def _search_by_vector(query_vector: list[float], limit: int = 3) -> list[dict[str, Any]]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    points = result.points if hasattr(result, "points") else []

    chunks: list[dict[str, Any]] = []

    for point in points:
        payload = point.payload or {}
        text = _extract_text(payload)

        if not text.strip():
            continue

        chunks.append(
            {
                "score": float(getattr(point, "score", 0.0)),
                "document_name": _extract_document_name(payload),
                "page": _extract_page(payload),
                "chunk_id": _extract_chunk_id(payload, getattr(point, "id", None)),
                "text": text,
            }
        )

    return chunks


def retrieve_similar_chunks(question: str, limit: int = 3) -> list[dict[str, Any]]:
    query_vector = embed_text(question)
    return _search_by_vector(query_vector=query_vector, limit=limit)


def search_similar_chunks(
    query_vector: list[float] | None = None,
    question: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if query_vector is None:
        if not question:
            raise ValueError("Нужно передать query_vector или question")
        query_vector = embed_text(question)

    return _search_by_vector(query_vector=query_vector, limit=limit)