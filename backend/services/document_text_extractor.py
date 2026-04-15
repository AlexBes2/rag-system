from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

from backend.services.ocr_service import image_to_text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
OCR_MIN_TEXT_LEN = int(os.getenv("OCR_MIN_TEXT_LEN", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "300"))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _needs_ocr(text: str) -> bool:
    return len(_normalize_text(text)) < OCR_MIN_TEXT_LEN


def _extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def _extract_image_text(file_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Не удалось открыть изображение") from exc

    return image_to_text(image)


def _extract_pdf_pages(file_bytes: bytes) -> list[dict[str, Any]]:
    reader = PdfReader(io.BytesIO(file_bytes))

    page_results: list[dict[str, Any]] = []
    pages_for_ocr: list[int] = []

    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()

        item = {
            "page": index + 1,
            "text": text,
            "ocr_used": False,
            "source_type": "pdf",
        }

        if _needs_ocr(text):
            pages_for_ocr.append(index)

        page_results.append(item)

    if not page_results:
        return []

    if pages_for_ocr:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_bytes(
                file_bytes,
                dpi=OCR_DPI,
                fmt="png",
                output_folder=temp_dir,
            )

            for index in pages_for_ocr:
                ocr_text = image_to_text(images[index])
                page_results[index]["text"] = ocr_text
                page_results[index]["ocr_used"] = True

    return page_results


def extract_document_pages(filename: str, file_bytes: bytes) -> list[dict[str, Any]]:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_pages(file_bytes)

    if ext == ".docx":
        return [
            {
                "page": 1,
                "text": _extract_docx_text(file_bytes),
                "ocr_used": False,
                "source_type": "docx",
            }
        ]

    if ext in IMAGE_EXTENSIONS:
        return [
            {
                "page": 1,
                "text": _extract_image_text(file_bytes),
                "ocr_used": True,
                "source_type": "image",
            }
        ]

    raise ValueError(
        "Поддерживаются только .pdf, .docx, .png, .jpg, .jpeg, .tif, .tiff, .bmp"
    )