import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from backend.core.config import settings
from backend.services.ollama_service import embed_query, embed_texts


client = QdrantClient(
    path=str(settings.qdrant_path),
    force_disable_check_same_thread=True,
)

_document_names_cache: list[str] | None = None

LOOKUP_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "contain",
    "containing",
    "documents",
    "find",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "title",
    "to",
    "with",
}

GENERIC_LOOKUP_TERMS = LOOKUP_STOPWORDS | {
    "about",
    "across",
    "after",
    "all",
    "also",
    "among",
    "based",
    "because",
    "before",
    "between",
    "can",
    "could",
    "data",
    "does",
    "each",
    "els",
    "few",
    "fine",
    "forecast",
    "forecasting",
    "forecasts",
    "framework",
    "further",
    "have",
    "however",
    "including",
    "improve",
    "improved",
    "improves",
    "method",
    "methods",
    "mod",
    "model",
    "models",
    "most",
    "not",
    "only",
    "other",
    "over",
    "paper",
    "performance",
    "related",
    "result",
    "results",
    "section",
    "show",
    "shown",
    "shows",
    "significant",
    "significantly",
    "such",
    "system",
    "task",
    "tasks",
    "than",
    "their",
    "these",
    "those",
    "through",
    "tuning",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
}


def _invalidate_document_cache() -> None:
    global _document_names_cache
    _document_names_cache = None


def collection_exists() -> bool:
    collections = client.get_collections().collections
    return any(collection.name == settings.collection_name for collection in collections)


def ensure_collection(vector_size: int) -> None:
    if collection_exists():
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def delete_document_points(document_name: str) -> None:
    if not collection_exists():
        return

    client.delete(
        collection_name=settings.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_name",
                        match=models.MatchValue(value=document_name),
                    )
                ]
            )
        ),
        wait=True,
    )

    _invalidate_document_cache()


def _normalize_lookup_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("—", " ").replace("–", " ").replace("-", " ")
    text = text.replace("_", " ").replace(":", " ").replace(".", " ")
    text = text.replace("ё", "е")
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"^[0-9]{4}\.[0-9]{5}v[0-9]\s*-\s*", "", text)
    text = re.sub(r"[\"'`“”‘’(){}\[\],;!?/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact_lookup_text(text: str) -> str:
    return re.sub(r"[^a-zа-яіїєґ0-9]+", "", _normalize_lookup_text(text))


def _lookup_segments(text: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []

    for term in re.findall(r"[a-zа-яіїєґ0-9]{3,}", _normalize_lookup_text(text)):
        if term in LOOKUP_STOPWORDS:
            if current:
                segments.append(_compact_lookup_text(" ".join(current)))
                current = []
            continue

        current.append(term)

    if current:
        segments.append(_compact_lookup_text(" ".join(current)))

    return [segment for segment in segments if len(segment) >= 8]


def _lookup_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for term in re.findall(r"[a-zа-яіїєґ0-9]{3,}", _normalize_lookup_text(text)):
        if term in LOOKUP_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    return terms


def _chunk_sort_key(chunk_id: Any) -> int:
    match = re.search(r"\d+", str(chunk_id or ""))
    if not match:
        return 0
    return int(match.group(0))


def _lookup_excerpt(haystack: str, needle: str, window: int = 720) -> str:
    display_text = re.sub(r"\s+", " ", haystack or "").strip()
    normalized = _normalize_lookup_text(display_text)
    if not normalized:
        return ""

    def excerpt_from(index: int) -> str:
        index = max(0, min(index, len(display_text)))
        start = max(0, index - window // 3)
        end = min(len(display_text), index + window)
        if start > 0:
            next_space = display_text.find(" ", start)
            if 0 <= next_space < index:
                start = next_space + 1
        return display_text[start:end].strip()

    needle_norm = _normalize_lookup_text(needle)
    if needle_norm:
        index = normalized.find(needle_norm)
        if index >= 0:
            return excerpt_from(index)

    compact_normalized = _compact_lookup_text(normalized)
    compact_needle = _compact_lookup_text(needle)
    if compact_needle:
        compact_index = compact_normalized.find(compact_needle)
        if compact_index >= 0:
            index = _compact_index_to_text_index(normalized, compact_index)
            return excerpt_from(index)

    for segment in _lookup_segments(needle):
        compact_index = compact_normalized.find(segment)
        if compact_index < 0:
            continue

        index = _compact_index_to_text_index(normalized, compact_index)
        return excerpt_from(index)

    for term in _lookup_terms(needle):
        index = normalized.find(term)
        if index < 0:
            continue

        return excerpt_from(index)

    return display_text[:window].strip()


def _compact_index_to_text_index(text: str, compact_index: int) -> int:
    compact_seen = 0

    for index, char in enumerate(text):
        if not re.match(r"[a-zа-яіїєґ0-9]", char):
            continue

        if compact_seen >= compact_index:
            return index

        compact_seen += 1

    return len(text)


def _segments_in_near_window(haystack: str, needle: str) -> bool:
    segments = _lookup_segments(needle)
    if len(segments) < 2:
        return False

    compact_haystack = _compact_lookup_text(haystack)
    compact_needle = _compact_lookup_text(needle)
    if not compact_haystack or not compact_needle:
        return False

    events: list[tuple[int, int]] = []
    for segment_index, segment in enumerate(segments):
        positions = [
            match.start()
            for match in re.finditer(re.escape(segment), compact_haystack)
        ]
        if not positions:
            return False
        events.extend((position, segment_index) for position in positions)

    events.sort(key=lambda item: item[0])

    max_window = max(1200, min(5000, len(compact_needle) * 30))
    counts = [0] * len(segments)
    covered = 0
    left = 0

    for right, (right_position, right_segment_index) in enumerate(events):
        if counts[right_segment_index] == 0:
            covered += 1
        counts[right_segment_index] += 1

        while covered == len(segments) and left <= right:
            left_position, left_segment_index = events[left]
            if right_position - left_position <= max_window:
                return True

            counts[left_segment_index] -= 1
            if counts[left_segment_index] == 0:
                covered -= 1
            left += 1

    return False


def _is_lookup_anchor(term: str) -> bool:
    if term in GENERIC_LOOKUP_TERMS:
        return False
    return any(char.isdigit() for char in term) or len(term) >= 5


def _terms_cover_window(
    compact_haystack: str,
    terms: list[str],
    min_covered: int,
    max_window: int,
    required_terms: set[str] | None = None,
) -> bool:
    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms or min_covered <= 0:
        return False

    required_indexes = {
        index
        for index, term in enumerate(unique_terms)
        if required_terms and term in required_terms
    }

    events: list[tuple[int, int]] = []
    for term_index, term in enumerate(unique_terms):
        events.extend(
            (match.start(), term_index)
            for match in re.finditer(re.escape(term), compact_haystack)
        )

    if not events:
        return False

    events.sort(key=lambda item: item[0])

    counts = [0] * len(unique_terms)
    covered = 0
    required_covered = 0
    left = 0

    for right_position, right_term_index in events:
        if counts[right_term_index] == 0:
            covered += 1
            if right_term_index in required_indexes:
                required_covered += 1
        counts[right_term_index] += 1

        while left < len(events) and right_position - events[left][0] > max_window:
            _, left_term_index = events[left]
            counts[left_term_index] -= 1
            if counts[left_term_index] == 0:
                covered -= 1
                if left_term_index in required_indexes:
                    required_covered -= 1
            left += 1

        if covered >= min_covered and required_covered == len(required_indexes):
            return True

    return False


def _terms_in_near_window(haystack: str, needle: str) -> bool:
    terms = [
        _compact_lookup_text(term)
        for term in _lookup_terms(needle)
    ]
    terms = [
        term
        for term in terms
        if len(term) >= 3
    ]

    if len(terms) < 4:
        return False

    compact_haystack = _compact_lookup_text(haystack)
    if not compact_haystack:
        return False

    compact_needle = _compact_lookup_text(needle)
    window = max(600, min(2600, len(compact_needle) * 28))
    anchor_terms = {term for term in terms if _is_lookup_anchor(term)}

    if len(anchor_terms) >= 2:
        min_matches = max(len(anchor_terms), (len(terms) * 65 + 99) // 100)
        return _terms_cover_window(
            compact_haystack,
            terms,
            min_matches,
            window,
            required_terms=anchor_terms,
        )

    min_matches = max(5, (len(terms) * 90 + 99) // 100)
    return _terms_cover_window(compact_haystack, terms, min_matches, window)


def _lookup_match_score(haystack: str, needle: str) -> float:
    needle_norm = _normalize_lookup_text(needle)
    if not needle_norm:
        return 0.0

    haystack_norm = _normalize_lookup_text(haystack)
    if needle_norm in haystack_norm:
        return 1.0

    needle_compact = _compact_lookup_text(needle_norm)
    if not needle_compact:
        return 0.0

    if needle_compact in _compact_lookup_text(haystack_norm):
        return 0.95

    if _segments_in_near_window(haystack_norm, needle_norm):
        return 0.85

    if _terms_in_near_window(haystack_norm, needle_norm):
        return 0.7

    return 0.0


def _lookup_contains(haystack: str, needle: str) -> bool:
    return _lookup_match_score(haystack, needle) > 0


def _build_exact_text_match(
    payload: dict[str, Any],
    text: str,
    needle: str,
    match_type: str,
    score: float = 1.0,
) -> dict[str, Any]:
    label = "Matched text" if score >= 0.85 else "Closest indexed text to"
    excerpt = f'{label}: "{needle}"'

    return {
        "score": score,
        "document_name": payload.get("document_name", ""),
        "page": payload.get("page"),
        "chunk_id": payload.get("chunk_id"),
        "section_index": payload.get("section_index"),
        "text": excerpt,
        "match_type": match_type,
    }


def _build_embedding_text(document_name: str, chunk: dict[str, Any]) -> str:
    text = (chunk.get("text") or "").strip()
    page = chunk.get("page")
    chunk_id = chunk.get("chunk_id") or ""

    prefix_parts = [f"Document: {document_name}"]
    if page is not None:
        prefix_parts.append(f"Page: {page}")
    if chunk_id:
        prefix_parts.append(f"Chunk: {chunk_id}")

    prefix = "\n".join(prefix_parts)
    return f"{prefix}\n\n{text}".strip()


def _document_filter(document_name: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_name",
                match=models.MatchValue(value=document_name),
            )
        ]
    )


def _extract_title_candidates(question: str) -> list[str]:
    candidates: list[str] = []
    q = (question or "").strip()

    if not q:
        return candidates

    quoted = re.findall(r'"([^"]{5,})"|\'([^\']{5,})\'', q)
    for pair in quoted:
        for item in pair:
            item = item.strip()
            if item:
                candidates.append(item)

    if len(q) >= 20:
        candidates.append(q)

    unique: list[str] = []
    seen = set()

    for item in candidates:
        key = _normalize_lookup_text(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def _get_document_names() -> list[str]:
    global _document_names_cache

    if _document_names_cache is not None:
        return _document_names_cache

    if not collection_exists():
        _document_names_cache = []
        return _document_names_cache

    names: set[str] = set()
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            document_name = payload.get("document_name")
            if document_name:
                names.add(str(document_name))

        if next_offset is None:
            break
        offset = next_offset

    _document_names_cache = sorted(names)
    return _document_names_cache


def _find_matching_documents(title: str, max_docs: int = 3) -> list[str]:
    title_norm = _normalize_lookup_text(title)
    if not title_norm:
        return []

    title_tokens = set(title_norm.split())
    matches: list[tuple[float, str]] = []

    for document_name in _get_document_names():
        doc_norm = _normalize_lookup_text(document_name)
        doc_tokens = set(doc_norm.split())

        score = 0.0

        if title_norm in doc_norm:
            score += 100.0
        elif doc_norm in title_norm:
            score += 80.0
        else:
            overlap = len(title_tokens & doc_tokens)
            if overlap >= max(4, min(8, len(title_tokens) // 2)):
                score += float(overlap)

        if score > 0:
            matches.append((score, document_name))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [document_name for _, document_name in matches[:max_docs]]


def find_documents_by_title(title: str, max_docs: int = 50) -> list[str]:
    title_norm = _normalize_lookup_text(title)
    if not title_norm:
        return []

    matches: list[str] = []
    for document_name in _get_document_names():
        doc_norm = _normalize_lookup_text(document_name)
        if title_norm in doc_norm:
            matches.append(document_name)

    return sorted(matches, key=str.lower)[:max_docs]


def find_documents_containing_text(text: str, max_docs: int = 50) -> list[dict[str, Any]]:
    if not collection_exists() or not _normalize_lookup_text(text):
        return []

    results: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    best_matches: dict[str, dict[str, Any]] = {}
    page_groups: dict[tuple[str, Any], dict[str, Any]] = {}

    def is_better_match(
        candidate: dict[str, Any],
        current: dict[str, Any] | None,
    ) -> bool:
        if current is None:
            return True

        candidate_score = float(candidate.get("score") or 0)
        current_score = float(current.get("score") or 0)
        if candidate_score != current_score:
            return candidate_score > current_score

        candidate_page = candidate.get("page")
        current_page = current.get("page")
        candidate_page_key = candidate_page if candidate_page is not None else 10**9
        current_page_key = current_page if current_page is not None else 10**9
        if candidate_page_key != current_page_key:
            return candidate_page_key < current_page_key

        return _chunk_sort_key(candidate.get("chunk_id")) < _chunk_sort_key(
            current.get("chunk_id")
        )

    for document_name in _get_document_names():
        if _lookup_contains(document_name, text):
            seen_docs.add(document_name)
            results.append(
                {
                    "score": 1.0,
                    "document_name": document_name,
                    "page": None,
                    "chunk_id": None,
                    "section_index": None,
                    "text": f"Document name: {document_name}",
                    "match_type": "title",
                }
            )
            if len(results) >= max_docs:
                return results

    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            document_name = str(payload.get("document_name") or "")
            chunk_text = str(payload.get("text") or "")

            if not document_name or document_name in seen_docs:
                continue

            score = _lookup_match_score(chunk_text, text)
            if score > 0:
                match_type = "exact_text" if score >= 0.85 else "near_text"
                candidate = _build_exact_text_match(
                    payload=payload,
                    text=chunk_text,
                    needle=text,
                    match_type=match_type,
                    score=score,
                )
                if is_better_match(candidate, best_matches.get(document_name)):
                    best_matches[document_name] = candidate

            page_key = (document_name, payload.get("page"))
            if page_key not in page_groups:
                page_groups[page_key] = {
                    "payload": payload,
                    "texts": [],
                }
            page_groups[page_key]["texts"].append(
                (_chunk_sort_key(payload.get("chunk_id")), chunk_text)
            )

        if next_offset is None:
            break
        offset = next_offset

    for group in page_groups.values():
        payload = group["payload"]
        document_name = str(payload.get("document_name") or "")
        if not document_name or document_name in seen_docs:
            continue

        page_text = "\n".join(
            item_text
            for _, item_text in sorted(group["texts"], key=lambda item: item[0])
        )
        score = _lookup_match_score(page_text, text)
        if score <= 0:
            continue

        match_type = "exact_page_text" if score >= 0.85 else "near_page_text"
        candidate = _build_exact_text_match(
            payload=payload,
            text=page_text,
            needle=text,
            match_type=match_type,
            score=score,
        )
        if is_better_match(candidate, best_matches.get(document_name)):
            best_matches[document_name] = candidate

    results.extend(
        sorted(
            best_matches.values(),
            key=lambda item: (
                -float(item.get("score") or 0),
                str(item.get("document_name", "")).lower(),
            ),
        )
    )
    results = results[:max_docs]
    return results


def _collect_document_payloads(document_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=_document_filter(document_name),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            if payload.get("document_name") == document_name:
                payloads.append(payload)

        if next_offset is None:
            break
        offset = next_offset

    payloads.sort(
        key=lambda payload: (
            payload.get("page") or 0,
            _chunk_sort_key(payload.get("chunk_id")),
        )
    )
    return payloads


def _payload_key(payload: dict[str, Any]) -> tuple[Any, Any]:
    return (payload.get("page"), payload.get("chunk_id"))


def _payload_to_hit(
    payload: dict[str, Any],
    score: float,
    match_type: str,
) -> dict[str, Any]:
    return {
        "score": score,
        "document_name": payload.get("document_name", ""),
        "page": payload.get("page"),
        "chunk_id": payload.get("chunk_id"),
        "section_index": payload.get("section_index"),
        "text": str(payload.get("text") or ""),
        "match_type": match_type,
    }


def _best_title_anchor(
    payloads: list[dict[str, Any]],
    title_candidates: list[str],
) -> dict[str, Any] | None:
    first_page_payloads = [
        payload
        for payload in payloads
        if payload.get("page") in (None, 1)
    ][:12]

    if not first_page_payloads:
        return payloads[0] if payloads else None

    for payload in first_page_payloads:
        text = str(payload.get("text") or "")
        if any(_lookup_contains(text, candidate) for candidate in title_candidates):
            return payload

    return first_page_payloads[0]


def _expand_hit_text(
    hit: dict[str, Any],
    payloads: list[dict[str, Any]],
    neighbor_count: int = 1,
    max_chars: int = 3600,
) -> dict[str, Any]:
    if not payloads:
        return hit

    hit_key = (hit.get("page"), hit.get("chunk_id"))
    index_by_key = {
        _payload_key(payload): index
        for index, payload in enumerate(payloads)
    }
    index = index_by_key.get(hit_key)
    if index is None:
        return hit

    start = max(0, index - neighbor_count)
    end = min(len(payloads), index + neighbor_count + 1)
    texts: list[str] = []
    seen_texts: set[str] = set()

    for payload in payloads[start:end]:
        text = str(payload.get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        texts.append(text)
        seen_texts.add(text)

    combined = "\n\n".join(texts).strip()
    if not combined:
        return hit

    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip()

    expanded = dict(hit)
    expanded["text"] = combined
    if expanded.get("match_type") == "document_filtered":
        expanded["match_type"] = "document_filtered_expanded"
    return expanded


def _search_within_document(
    document_name: str,
    question: str,
    limit: int,
    title_candidates: list[str],
) -> list[dict[str, Any]]:
    document_payloads = _collect_document_payloads(document_name)
    query_vector = embed_query(question)

    result = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        query_filter=_document_filter(document_name),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    hits: list[dict[str, Any]] = []
    title_anchor = _best_title_anchor(document_payloads, title_candidates)
    if title_anchor:
        hits.append(
            _payload_to_hit(
                payload=title_anchor,
                score=3.0,
                match_type="title_anchor",
            )
        )

    for point in result.points:
        payload = point.payload or {}
        text = str(payload.get("text") or "")
        page = payload.get("page")
        chunk_id = payload.get("chunk_id")

        bonus = 0.0
        if page == 1:
            bonus += 1.0
        if str(chunk_id) == "chunk_0":
            bonus += 1.0

        text_norm = _normalize_lookup_text(text)
        for candidate in title_candidates:
            candidate_norm = _normalize_lookup_text(candidate)
            if candidate_norm and (
                candidate_norm in text_norm or _lookup_contains(text, candidate)
            ):
                bonus += 2.0
                break

        hits.append(
            _payload_to_hit(
                payload=payload,
                score=float(point.score or 0.0) + bonus,
                match_type="document_filtered",
            )
        )

    hits.sort(key=lambda x: x["score"], reverse=True)
    expanded_hits = [
        _expand_hit_text(hit, document_payloads)
        for hit in hits
    ]
    return expanded_hits[:limit]


def upsert_document_chunks(document_name: str, chunks: list[dict[str, Any]]) -> int:
    if not chunks:
        return 0

    embedding_texts = [
        _build_embedding_text(document_name=document_name, chunk=chunk)
        for chunk in chunks
    ]
    vectors = embed_texts(embedding_texts)

    ensure_collection(vector_size=len(vectors[0]))
    delete_document_points(document_name)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_name": document_name,
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("chunk_id"),
                    "section_index": chunk.get("section_index"),
                    "text": chunk.get("text"),
                },
            )
        )

    client.upsert(
        collection_name=settings.collection_name,
        points=points,
        wait=True,
    )

    _invalidate_document_cache()
    return len(points)


def search_similar_chunks(question: str, limit: int = 3) -> list[dict[str, Any]]:
    if not collection_exists():
        return []

    title_candidates = _extract_title_candidates(question)

    # 1) Быстрый путь: ищем совпадающий document_name и ищем только внутри него
    matched_docs: list[str] = []
    for candidate in title_candidates[:2]:
        matched_docs.extend(_find_matching_documents(candidate, max_docs=3))

    # dedupe matched docs
    unique_docs: list[str] = []
    seen_docs = set()
    for doc in matched_docs:
        if doc not in seen_docs:
            seen_docs.add(doc)
            unique_docs.append(doc)

    if unique_docs:
        results: list[dict[str, Any]] = []
        for document_name in unique_docs[:3]:
            results.extend(
                _search_within_document(
                    document_name=document_name,
                    question=question,
                    limit=max(limit * 2, 6),
                    title_candidates=title_candidates,
                )
            )

        deduped: list[dict[str, Any]] = []
        seen = set()

        results.sort(key=lambda x: x["score"], reverse=True)

        for item in results:
            key = (
                item.get("document_name"),
                item.get("page"),
                item.get("chunk_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break

        if deduped:
            return deduped

    # 2) Fallback: обычный semantic search по всей коллекции
    query_vector = embed_query(question)

    result = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=max(limit * 3, 10),
        with_payload=True,
        with_vectors=False,
    )

    hits = []
    for point in result.points:
        payload = point.payload or {}
        hits.append(
            {
                "score": float(point.score or 0.0),
                "document_name": payload.get("document_name", ""),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id"),
                "section_index": payload.get("section_index"),
                "text": payload.get("text", ""),
                "match_type": "semantic",
            }
        )

    return hits[:limit]


def list_documents() -> list[dict[str, Any]]:
    if not collection_exists():
        return []

    aggregated: dict[str, dict[str, Any]] = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=settings.collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            document_name = payload.get("document_name")

            if not document_name:
                continue

            if document_name not in aggregated:
                aggregated[document_name] = {
                    "document_name": document_name,
                    "chunk_count": 0,
                    "pages": set(),
                }

            aggregated[document_name]["chunk_count"] += 1

            page = payload.get("page")
            if isinstance(page, int):
                aggregated[document_name]["pages"].add(page)

        if next_offset is None:
            break

        offset = next_offset

    documents = []
    for item in aggregated.values():
        documents.append(
            {
                "document_name": item["document_name"],
                "chunk_count": item["chunk_count"],
                "pages": sorted(item["pages"]),
            }
        )

    documents.sort(key=lambda x: x["document_name"].lower())
    return documents
