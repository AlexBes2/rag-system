from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


class Settings:
    def __init__(self) -> None:
        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        ).rstrip("/")

        self.ollama_embed_model: str = os.getenv(
            "OLLAMA_EMBED_MODEL",
            "nomic-embed-text",
        )

        self.ollama_llm_model: str = os.getenv(
            "OLLAMA_LLM_MODEL",
            "llama3",
        )

        self.collection_name: str = os.getenv(
            "COLLECTION_NAME",
            "documents",
        )

        self.qdrant_path: Path = resolve_path(
            os.getenv("QDRANT_PATH", "data/qdrant")
        )

        self.upload_dir: Path = resolve_path(
            os.getenv("UPLOAD_DIR", "data/uploads")
        )

        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "75"))
        self.search_limit: int = int(os.getenv("SEARCH_LIMIT", "3"))


settings = Settings()

settings.qdrant_path.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)