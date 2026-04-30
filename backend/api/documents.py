from pathlib import Path

from fastapi import APIRouter

from backend.core.config import settings
from backend.services.qdrant_service import list_documents

router = APIRouter(tags=["documents"])


@router.get("/documents")
def get_documents():
    try:
        indexed_documents = list_documents()
    except Exception:
        indexed_documents = []

    if indexed_documents:
        return {
            "documents": indexed_documents,
            "total_documents": len(indexed_documents),
            "total_chunks": sum(
                int(item.get("chunk_count") or 0)
                for item in indexed_documents
            ),
            "source": "index",
        }

    uploads_dir = Path(settings.upload_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [file.name for file in uploads_dir.iterdir() if file.is_file()]
    )

    return {
        "documents": files,
        "total_documents": len(files),
        "total_chunks": 0,
        "source": "uploads",
    }
