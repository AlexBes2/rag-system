from __future__ import annotations

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str]) -> list[list[float]]:
    prepared_texts = [text.strip() for text in texts if text and text.strip()]

    if not prepared_texts:
        raise ValueError("Список текстов для embeddings пустой")

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "input": prepared_texts,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or not isinstance(embeddings, list):
        raise ValueError(f"Некорректный ответ Ollama: {data}")

    return embeddings


def get_embedding(text: str) -> list[float]:
    text = text.strip()

    if not text:
        raise ValueError("Пустой текст нельзя отправить на embeddings")

    embeddings = embed_texts([text])
    return embeddings[0]