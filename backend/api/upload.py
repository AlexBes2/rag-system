from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.config import settings
from backend.schemas.api import BatchUploadItem, BatchUploadResponse, UploadResponse
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


def _filename_for_error(file: UploadFile) -> str:
    if file is None or not file.filename:
        return "unknown"
    return Path(file.filename).name.strip() or "unknown"


async def _process_upload_file(file: UploadFile) -> UploadResponse:
    try:
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
        if file is not None:
            await file.close()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    return await _process_upload_file(file)


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> BatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Файлы не переданы")

    results: list[BatchUploadItem] = []

    for file in files:
        filename = _filename_for_error(file)

        try:
            uploaded = await _process_upload_file(file)
            results.append(
                BatchUploadItem(
                    filename=uploaded.filename,
                    success=True,
                    chunks_indexed=uploaded.chunks_indexed,
                    message=uploaded.message,
                )
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            results.append(
                BatchUploadItem(
                    filename=filename,
                    success=False,
                    error=detail,
                )
            )
        except Exception as exc:
            results.append(
                BatchUploadItem(
                    filename=filename,
                    success=False,
                    error=f"Ошибка загрузки документа: {exc}",
                )
            )

    success_count = sum(1 for item in results if item.success)
    failed_count = len(results) - success_count
    chunks_indexed = sum(item.chunks_indexed for item in results)

    return BatchUploadResponse(
        total_files=len(results),
        success_count=success_count,
        failed_count=failed_count,
        chunks_indexed=chunks_indexed,
        results=results,
        message=(
            f"Загружено {success_count} из {len(results)} файлов, "
            f"проиндексировано чанков: {chunks_indexed}"
        ),
    )
