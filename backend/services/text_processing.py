import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import settings


HEADER_CHUNK_MAX_CHARS = 1800
MIN_CHUNK_TEXT_LEN = 20


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_chunk_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_header_chunk(section: dict[str, Any], global_index: int) -> dict[str, Any] | None:
    """
    Специальный чанк для начала документа.
    Особенно полезен для scientific PDF:
    title / authors / abstract чаще всего находятся на первой странице.
    """
    text = normalize_text(section.get("text", ""))
    if not text:
        return None

    header_text = text[:HEADER_CHUNK_MAX_CHARS].strip()
    if len(header_text) < MIN_CHUNK_TEXT_LEN:
        return None

    return {
        "chunk_id": f"chunk_{global_index}",
        "document_name": section.get("document_name"),
        "page": section.get("page"),
        "section_index": section.get("section_index"),
        "chunk_index": -1,
        "text": header_text,
        "source_type": section.get("source_type", "unknown"),
        "ocr_used": section.get("ocr_used", False),
        "chunk_kind": "header",
    }


def _is_first_document_section(section: dict[str, Any]) -> bool:
    page = section.get("page")
    section_index = section.get("section_index")

    return page == 1 or section_index == 0


def chunk_document_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    global_index = 0
    seen_chunk_keys: set[tuple[Any, Any, str]] = set()

    for section in sections:
        document_name = section.get("document_name")
        page = section.get("page")
        section_index = section.get("section_index")
        source_type = section.get("source_type", "unknown")
        ocr_used = section.get("ocr_used", False)

        text = normalize_text(section.get("text", ""))
        if not text:
            continue

        # Специальный header chunk для начала документа
        if _is_first_document_section(section):
            header_chunk = _build_header_chunk(section, global_index)
            if header_chunk:
                header_key = (
                    header_chunk.get("document_name"),
                    header_chunk.get("page"),
                    header_chunk.get("text"),
                )
                if header_key not in seen_chunk_keys:
                    seen_chunk_keys.add(header_key)
                    chunks.append(header_chunk)
                    global_index += 1

        pieces = splitter.split_text(text)

        for local_index, piece in enumerate(pieces):
            chunk_text = _clean_chunk_text(piece)
            if len(chunk_text) < MIN_CHUNK_TEXT_LEN:
                continue

            chunk_key = (document_name, page, chunk_text)
            if chunk_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(chunk_key)

            chunks.append(
                {
                    "chunk_id": f"chunk_{global_index}",
                    "document_name": document_name,
                    "page": page,
                    "section_index": section_index,
                    "chunk_index": local_index,
                    "text": chunk_text,
                    "source_type": source_type,
                    "ocr_used": ocr_used,
                    "chunk_kind": "regular",
                }
            )
            global_index += 1

    return chunks