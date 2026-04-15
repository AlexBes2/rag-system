from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.upload import router as upload_router
from backend.api.query import router as query_router
from backend.api.documents import router as documents_router

app = FastAPI(title="RAG System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(documents_router)

uploads_dir = Path("data/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(uploads_dir)), name="files")