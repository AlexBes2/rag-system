import requests

from backend.core.config import settings


class OllamaServiceError(Exception):
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = requests.post(
        f"{settings.ollama_base_url}/api/embed",
        json={
            "model": settings.ollama_embed_model,
            "input": texts,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise OllamaServiceError(
            f"Ошибка embeddings Ollama: {response.status_code} - {response.text}"
        )

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings:
        raise OllamaServiceError("Ollama не вернул embeddings")

    return embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
Ты помощник в RAG-системе.
Отвечай ТОЛЬКО на основе контекста ниже.
Если данных недостаточно, честно скажи: "Недостаточно данных в найденных документах".
Отвечай на языке вопроса.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
""".strip()

    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.ollama_llm_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    if response.status_code != 200:
        raise OllamaServiceError(
            f"Ошибка генерации Ollama: {response.status_code} - {response.text}"
        )

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise OllamaServiceError("Ollama не вернул текст ответа")

    return answer