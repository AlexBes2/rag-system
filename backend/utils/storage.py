import json
import os
from pathlib import Path
from typing import Any

from backend.core.config import RAW_DIR, CHUNKS_DIR, EMBEDDINGS_DIR


def ensure_data_dirs() -> None:
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_raw_text_path(document_id: str) -> str:
    return os.path.join(RAW_DIR, f"{document_id}.txt")


def build_chunks_path(document_id: str) -> str:
    return os.path.join(CHUNKS_DIR, f"{document_id}.json")


def build_embeddings_path(document_id: str) -> str:
    return os.path.join(EMBEDDINGS_DIR, f"{document_id}.json")