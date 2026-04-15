from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


@dataclass
class DocumentPage:
    page_number: int
    text: str


def extract_document_pages(file_path: str | Path) -> list[DocumentPage]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf_pages(path)

    if suffix == ".docx":
        return _extract_docx_pages(path)

    raise ValueError(f"Неподдерживаемый формат файла: {suffix}")


def _extract_pdf_pages(path: Path) -> list[DocumentPage]:
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(DocumentPage(page_number=index, text=text))

    return pages


def _extract_docx_pages(path: Path) -> list[DocumentPage]:
    doc = Document(str(path))
    paragraphs = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    # У DOCX обычно нет надежной "страницы" после парсинга,
    # поэтому для MVP считаем весь документ одной страницей.
    return [DocumentPage(page_number=1, text=full_text)]