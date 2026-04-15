from typing import Any

from backend.services.ollama_service import generate_answer
from backend.services.qdrant_service import search_similar_chunks


def build_context(hits: list[dict[str, Any]]) -> str:
    blocks = []

    for idx, hit in enumerate(hits, start=1):
        document_name = hit.get("document_name", "unknown")
        page = hit.get("page")
        text = hit.get("text", "").strip()

        page_part = f", page {page}" if page is not None else ""
        blocks.append(f"[SOURCE {idx}: {document_name}{page_part}]\n{text}")

    return "\n\n".join(blocks)


def aggregate_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for hit in hits:
        document_name = hit.get("document_name", "unknown")

        if document_name not in grouped:
            grouped[document_name] = {
                "document_name": document_name,
                "pages": set(),
                "chunk_ids": set(),
            }

        page = hit.get("page")
        chunk_id = hit.get("chunk_id")

        if isinstance(page, int):
            grouped[document_name]["pages"].add(page)

        if chunk_id:
            grouped[document_name]["chunk_ids"].add(chunk_id)

    sources = []
    for item in grouped.values():
        sources.append(
            {
                "document_name": item["document_name"],
                "pages": sorted(item["pages"]),
                "chunk_ids": sorted(item["chunk_ids"]),
            }
        )

    return sources


def answer_question(question: str, k: int) -> dict[str, Any]:
    hits = search_similar_chunks(question=question, limit=k)

    if not hits:
        return {
            "answer": "В базе пока нет релевантных документов для ответа на этот вопрос.",
            "sources": [],
        }

    context = build_context(hits)
    answer = generate_answer(question=question, context=context)
    sources = aggregate_sources(hits)

    return {
        "answer": answer,
        "sources": sources,
    }