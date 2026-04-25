from pathlib import Path

from fastapi import APIRouter

from backend.core.config import settings

router = APIRouter(tags=["documents"])


@router.get("/documents")
def get_documents():
    uploads_dir = Path(settings.upload_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [file.name for file in uploads_dir.iterdir() if file.is_file()]
    )

    return {"documents": files}
