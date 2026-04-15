from __future__ import annotations

from typing import Any

import requests

from backend.core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_LLM_MODEL,
)


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    return response.json()


def embed_text(text: str) -> list[float]:
    data = _post(
        "/api/embed",
        {
            "model": OLLAMA_EMBED_MODEL,
            "input": text,
        },
    )

    if "embedding" in data and isinstance(data["embedding"], list):
        return data["embedding"]

    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first_item = embeddings[0]

        if isinstance(first_item, list):
            return first_item

        if all(isinstance(x, (int, float)) for x in embeddings):
            return embeddings

    raise RuntimeError("Ollama did not return a valid embedding.")


def generate_text(prompt: str) -> str:
    data = _post(
        "/api/generate",
        {
            "model": OLLAMA_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )

    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")

    return text