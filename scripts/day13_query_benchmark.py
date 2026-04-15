#!/usr/bin/env python3

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_QUERY_URL = "http://127.0.0.1:8000/query"
DEFAULT_REPORTS_DIR = Path("data/reports")
DEFAULT_METADATA_FILE = Path("data/test_documents/arxiv_1500/metadata.jsonl")


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def shorten(text: str, max_len: int = 180) -> str:
    text = text.replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def flatten_for_search(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, list):
        return " ".join(flatten_for_search(x) for x in obj)
    if isinstance(obj, dict):
        parts = []
        for key, value in obj.items():
            parts.append(str(key))
            parts.append(flatten_for_search(value))
        return " ".join(parts)
    return str(obj)


def extract_answer(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        preferred_keys = [
            "answer",
            "response",
            "text",
            "result",
            "output",
            "generated_text",
            "message",
        ]
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, dict):
                nested = extract_answer(value)
                if nested:
                    return nested

    return ""


def extract_sources(payload: Any) -> list[str]:
    collected: list[str] = []

    source_like_keys = {
        "sources",
        "source_documents",
        "documents",
        "results",
        "matches",
        "retrieved",
        "chunks",
        "contexts",
        "citations",
    }

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in source_like_keys:
                    if isinstance(value, list):
                        for item in value:
                            collected.append(flatten_source_item(item))
                    elif value is not None:
                        collected.append(flatten_source_item(value))
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)

    cleaned = []
    seen = set()

    for item in collected:
        text = " ".join(item.split())
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)

    return cleaned


def flatten_source_item(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        preferred = [
            "filename",
            "file_name",
            "document",
            "document_name",
            "source",
            "path",
            "title",
            "page",
            "chunk_id",
            "score",
            "text",
        ]
        parts = []
        for key in preferred:
            if key in item:
                parts.append(f"{key}={normalize_text(item.get(key))}")

        if not parts:
            parts.append(normalize_text(item))

        return " | ".join(parts)

    return normalize_text(item)


def contains_all(haystack: str, needles: list[str]) -> bool:
    haystack_l = haystack.lower()
    return all(n.lower() in haystack_l for n in needles if n.strip())


def contains_any_in_list(items: list[str], needles: list[str]) -> bool:
    if not needles:
        return False
    low_needles = [n.lower() for n in needles if n.strip()]
    if not low_needles:
        return False

    for item in items:
        item_l = item.lower()
        if any(n in item_l for n in low_needles):
            return True
    return False


def load_metadata_questions(
    metadata_file: Path,
    auto_count: int,
    seed: int,
) -> list[dict]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    docs = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not docs:
        raise RuntimeError("Metadata file is empty or invalid.")

    random.seed(seed)
    random.shuffle(docs)

    selected = docs[: max(1, auto_count)]
    questions = []

    for doc in selected:
        title = (doc.get("title") or "").strip()
        arxiv_id = (doc.get("arxiv_id") or "").strip()
        authors = doc.get("authors") or []
        first_author = authors[0].strip() if authors else ""

        title_prefix = title[:80].strip() if title else ""
        source_hints = [x for x in [arxiv_id, title_prefix] if x]

        if title:
            questions.append(
                {
                    "question": f'Who are the authors of the paper "{title}"?',
                    "expected_source_substrings": source_hints,
                    "expected_answer_substrings": [first_author] if first_author else [],
                    "tags": ["auto", "authors"],
                }
            )

            questions.append(
                {
                    "question": f'Summarize the paper "{title}" in 2-3 sentences.',
                    "expected_source_substrings": source_hints,
                    "expected_answer_substrings": [],
                    "tags": ["auto", "summary"],
                }
            )

    return questions[: auto_count]


def load_questions_from_file(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise RuntimeError("JSON file must contain a list.")
        return normalize_question_items(data)

    if suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return normalize_question_items(items)

    if suffix == ".txt":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append({"question": line})
        return normalize_question_items(items)

    raise RuntimeError("Supported question file formats: .json, .jsonl, .txt")


def normalize_question_items(items: list[Any]) -> list[dict]:
    normalized = []

    for item in items:
        if isinstance(item, str):
            normalized.append(
                {
                    "question": item,
                    "expected_source_substrings": [],
                    "expected_answer_substrings": [],
                    "tags": [],
                }
            )
            continue

        if isinstance(item, dict):
            question = normalize_text(item.get("question"))
            if not question:
                continue

            expected_sources = item.get("expected_source_substrings") or []
            expected_answers = item.get("expected_answer_substrings") or []
            tags = item.get("tags") or []

            if isinstance(expected_sources, str):
                expected_sources = [expected_sources]
            if isinstance(expected_answers, str):
                expected_answers = [expected_answers]
            if isinstance(tags, str):
                tags = [tags]

            normalized.append(
                {
                    "question": question,
                    "expected_source_substrings": [normalize_text(x) for x in expected_sources],
                    "expected_answer_substrings": [normalize_text(x) for x in expected_answers],
                    "tags": [normalize_text(x) for x in tags],
                }
            )

    if not normalized:
        raise RuntimeError("No valid questions found.")

    return normalized


def send_query(
    query_url: str,
    question: str,
    k: int,
    timeout: int,
) -> dict:
    payload = {
        "question": question,
        "k": k,
    }
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        query_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            status_code = response.getcode()

            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None

            return {
                "ok": 200 <= status_code < 300,
                "status_code": status_code,
                "elapsed_sec": elapsed,
                "raw_body": body,
                "payload": parsed,
                "error": None,
            }

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None

        return {
            "ok": False,
            "status_code": e.code,
            "elapsed_sec": elapsed,
            "raw_body": body,
            "payload": parsed,
            "error": f"HTTPError {e.code}",
        }

    except Exception as e:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "status_code": 0,
            "elapsed_sec": elapsed,
            "raw_body": "",
            "payload": None,
            "error": str(e),
        }


def make_summary_markdown(report: dict) -> str:
    lines = []
    lines.append("# Day 13 Query Benchmark")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Query URL: {report['query_url']}")
    lines.append(f"- Questions: {report['question_count']}")
    lines.append(f"- Repeats: {report['repeats']}")
    lines.append(f"- k: {report['k']}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Success rate: {report['aggregate']['success_rate']:.2f}%")
    lines.append(f"- Avg latency: {report['aggregate']['avg_latency_sec']:.3f} sec")
    lines.append(f"- P50 latency: {report['aggregate']['p50_latency_sec']:.3f} sec")
    lines.append(f"- P95 latency: {report['aggregate']['p95_latency_sec']:.3f} sec")
    lines.append(f"- Min latency: {report['aggregate']['min_latency_sec']:.3f} sec")
    lines.append(f"- Max latency: {report['aggregate']['max_latency_sec']:.3f} sec")
    lines.append(f"- Source hit rate: {report['aggregate']['source_hit_rate']:.2f}%")
    lines.append(f"- Answer hit rate: {report['aggregate']['answer_hit_rate']:.2f}%")
    lines.append("")
    lines.append("## Slowest queries")
    lines.append("")

    slowest = sorted(
        report["results"],
        key=lambda x: x["elapsed_sec"],
        reverse=True,
    )[:10]

    for item in slowest:
        lines.append(
            f"- {item['elapsed_sec']:.3f}s | status={item['status_code']} | "
            f"source_hit={item['source_hit']} | answer_hit={item['answer_hit']} | "
            f"{shorten(item['question'], 140)}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 13 benchmark for /query endpoint")
    parser.add_argument(
        "--query-url",
        type=str,
        default=DEFAULT_QUERY_URL,
        help=f"Query endpoint URL. Default: {DEFAULT_QUERY_URL}",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Path to .json / .jsonl / .txt with questions.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA_FILE,
        help=f"Metadata file for auto-generated questions. Default: {DEFAULT_METADATA_FILE}",
    )
    parser.add_argument(
        "--auto-count",
        type=int,
        default=20,
        help="How many questions to auto-generate from metadata if no questions file is provided. Default: 20",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for auto-generated question sampling. Default: 42",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Value of k sent to /query. Default: 3",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat each question. Default: 1",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=0,
        help="Pause between requests in milliseconds. Default: 0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions after loading. 0 = no limit",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Directory for reports. Default: {DEFAULT_REPORTS_DIR}",
    )
    args = parser.parse_args()

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    if args.questions_file:
        questions = load_questions_from_file(args.questions_file)
        questions_source = str(args.questions_file)
    else:
        questions = load_metadata_questions(
            metadata_file=args.metadata_file,
            auto_count=args.auto_count,
            seed=args.seed,
        )
        questions_source = f"auto-generated from {args.metadata_file}"

    if args.limit > 0:
        questions = questions[: args.limit]

    if not questions:
        raise SystemExit("No questions available.")

    total_runs = len(questions) * max(1, args.repeats)

    print(f"Questions source: {questions_source}")
    print(f"Questions loaded: {len(questions)}")
    print(f"Repeats: {args.repeats}")
    print(f"Total runs: {total_runs}")
    print(f"Query URL: {args.query_url}")
    print(f"k: {args.k}")
    print()

    results = []
    run_index = 0

    for q_index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_source_substrings = item.get("expected_source_substrings", [])
        expected_answer_substrings = item.get("expected_answer_substrings", [])
        tags = item.get("tags", [])

        for repeat_index in range(1, args.repeats + 1):
            run_index += 1

            response = send_query(
                query_url=args.query_url,
                question=question,
                k=args.k,
                timeout=args.timeout,
            )

            answer = ""
            sources = []

            if response["payload"] is not None:
                answer = extract_answer(response["payload"])
                sources = extract_sources(response["payload"])

            source_hit = (
                contains_any_in_list(sources, expected_source_substrings)
                if expected_source_substrings
                else None
            )

            answer_hit = (
                contains_all(answer, expected_answer_substrings)
                if expected_answer_substrings
                else None
            )

            result_item = {
                "question_index": q_index,
                "repeat_index": repeat_index,
                "question": question,
                "tags": tags,
                "status_code": response["status_code"],
                "ok": response["ok"],
                "elapsed_sec": round(response["elapsed_sec"], 4),
                "error": response["error"],
                "expected_source_substrings": expected_source_substrings,
                "expected_answer_substrings": expected_answer_substrings,
                "source_hit": source_hit,
                "answer_hit": answer_hit,
                "answer_preview": shorten(answer, 250),
                "sources_preview": [shorten(x, 250) for x in sources[:5]],
                "raw_body_preview": shorten(response["raw_body"], 500),
            }
            results.append(result_item)

            print(
                f"[{run_index}/{total_runs}] "
                f"status={response['status_code']} | "
                f"time={response['elapsed_sec']:.3f}s | "
                f"source_hit={source_hit} | "
                f"answer_hit={answer_hit} | "
                f"{shorten(question, 100)}"
            )

            if args.pause_ms > 0:
                time.sleep(args.pause_ms / 1000.0)

    latencies = [x["elapsed_sec"] for x in results]
    ok_count = sum(1 for x in results if x["ok"])
    success_rate = (ok_count / len(results) * 100.0) if results else 0.0

    source_scored = [x for x in results if x["source_hit"] is not None]
    source_hits = sum(1 for x in source_scored if x["source_hit"] is True)
    source_hit_rate = (source_hits / len(source_scored) * 100.0) if source_scored else 0.0

    answer_scored = [x for x in results if x["answer_hit"] is not None]
    answer_hits = sum(1 for x in answer_scored if x["answer_hit"] is True)
    answer_hit_rate = (answer_hits / len(answer_scored) * 100.0) if answer_scored else 0.0

    report = {
        "timestamp": now_ts(),
        "query_url": args.query_url,
        "questions_source": questions_source,
        "question_count": len(questions),
        "repeats": args.repeats,
        "k": args.k,
        "timeout": args.timeout,
        "aggregate": {
            "total_runs": len(results),
            "ok_count": ok_count,
            "success_rate": round(success_rate, 3),
            "avg_latency_sec": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
            "p50_latency_sec": round(percentile(latencies, 50), 4) if latencies else 0.0,
            "p95_latency_sec": round(percentile(latencies, 95), 4) if latencies else 0.0,
            "min_latency_sec": round(min(latencies), 4) if latencies else 0.0,
            "max_latency_sec": round(max(latencies), 4) if latencies else 0.0,
            "source_scored_count": len(source_scored),
            "source_hits": source_hits,
            "source_hit_rate": round(source_hit_rate, 3),
            "answer_scored_count": len(answer_scored),
            "answer_hits": answer_hits,
            "answer_hit_rate": round(answer_hit_rate, 3),
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.reports_dir / f"day13_query_benchmark_{stamp}.json"
    md_path = args.reports_dir / f"day13_query_benchmark_{stamp}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(make_summary_markdown(report), encoding="utf-8")

    print("\n=== ГОТОВО ===")
    print(f"Всего прогонов: {len(results)}")
    print(f"Success rate: {report['aggregate']['success_rate']:.2f}%")
    print(f"Avg latency: {report['aggregate']['avg_latency_sec']:.3f} sec")
    print(f"P50 latency: {report['aggregate']['p50_latency_sec']:.3f} sec")
    print(f"P95 latency: {report['aggregate']['p95_latency_sec']:.3f} sec")
    print(f"Source hit rate: {report['aggregate']['source_hit_rate']:.2f}%")
    print(f"Answer hit rate: {report['aggregate']['answer_hit_rate']:.2f}%")
    print(f"JSON report: {json_path}")
    print(f"Markdown summary: {md_path}")


if __name__ == "__main__":
    main()