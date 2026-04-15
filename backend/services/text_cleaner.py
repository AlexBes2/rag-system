import re

from backend.services.document_loader import DocumentPage


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(pages: list[DocumentPage]) -> list[DocumentPage]:
    cleaned_pages: list[DocumentPage] = []

    for page in pages:
        cleaned_text = normalize_text(page.text)
        if cleaned_text:
            cleaned_pages.append(
                DocumentPage(
                    page_number=page.page_number,
                    text=cleaned_text,
                )
            )

    return cleaned_pages