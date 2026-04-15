from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["documents"])


@router.get("/documents")
def get_documents():
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [file.name for file in uploads_dir.iterdir() if file.is_file()]
    )

    return {"documents": files}