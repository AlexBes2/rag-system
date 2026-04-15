from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.services.document_loader import DocumentPage


@dataclass
class DocumentChunk:
    chunk_id: str
    document_name: str
    page: int
    text: str


def chunk_document(
    pages: list[DocumentPage],
    document_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = splitter.split_text(page.text)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            clean_chunk = chunk_text.strip()
            if not clean_chunk:
                continue

            chunk_id = f"{document_name}::page_{page.page_number}::chunk_{chunk_index}"

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page=page.page_number,
                    text=clean_chunk,
                )
            )

    return chunks