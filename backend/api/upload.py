from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.config import settings
from backend.schemas.api import UploadResponse
from backend.services.document_parser import (
    SUPPORTED_EXTENSIONS,
    extract_document_sections,
)
from backend.services.qdrant_service import upsert_document_chunks
from backend.services.text_processing import chunk_document_sections

router = APIRouter(tags=["upload"])


def _supported_extensions_message() -> str:
    extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"Поддерживаются только: {extensions}"


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан")

    safe_filename = Path(file.filename).name.strip()
    if not safe_filename:
        raise HTTPException(status_code=400, detail="Некорректное имя файла")

    extension = Path(safe_filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=_supported_extensions_message(),
        )

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.upload_dir / safe_filename

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Файл пустой")

        destination.write_bytes(content)

        sections = extract_document_sections(destination)
        if not sections:
            raise HTTPException(
                status_code=400,
                detail="Не удалось извлечь текст из документа",
            )

        chunks = chunk_document_sections(sections)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Не удалось разбить документ на чанки",
            )

        chunks_indexed = upsert_document_chunks(
            document_name=safe_filename,
            chunks=chunks,
        )

        return UploadResponse(
            filename=safe_filename,
            chunks_indexed=chunks_indexed,
            message="Документ загружен, обработан и проиндексирован",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка загрузки документа: {exc}",
        ) from exc
    finally:
        await file.close()