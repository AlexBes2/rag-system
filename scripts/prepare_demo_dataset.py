#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("data/test_documents/arxiv_1500")
DEFAULT_OUTPUT_DIR = Path("data/demo_documents/technical_300")
DEFAULT_LIMIT = 300

CURATED_TITLES = [
    "LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture",
    "Deep Expert Injection for Anchoring Retinal VLMs with Domain-Specific Knowledge",
    "Experiences Build Characters: The Linguistic Origins and Functional Impact of LLM Personality",
    "Progressive Residual Warmup for Language Model Pretraining",
    "Learning to Reflect and Correct: Towards Better Decoding Trajectories for Large-Scale Generative Recommendation",
    "Benchmarking Large Language Models for Quebec Insurance: From Closed-Book to Retrieval-Augmented Generation",
    "JARVIS: An Evidence-Grounded Retrieval System for Interpretable Deceptive Reviews Adjudication",
    "Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from an Industry Deployment",
]

QUESTION_TEMPLATES = [
    'Who are the authors of the paper "{title}"?',
    'Summarize the paper "{title}" in 2-3 sentences.',
    'What problem does the paper "{title}" try to solve?',
    'What method or system is proposed in "{title}"?',
]


def load_metadata(metadata_file: Path) -> list[dict[str, Any]]:
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    records: list[dict[str, Any]] = []
    with metadata_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {metadata_file}:{line_number}: {exc}"
                ) from exc
            if isinstance(item, dict):
                records.append(item)

    return records


def resolve_source_path(item: dict[str, Any], source_dir: Path) -> Path | None:
    local_path = item.get("local_path")
    candidates: list[Path] = []

    if local_path:
        candidates.append(Path(str(local_path)))

    arxiv_id = item.get("arxiv_id")
    title = item.get("title")
    if arxiv_id and title:
        safe_title = str(title).replace(":", "_").replace("/", "_")
        candidates.append(source_dir / f"{arxiv_id} - {safe_title}.pdf")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def normalized_title(title: str) -> str:
    return " ".join(title.lower().replace("_", " ").split())


def choose_records(
    records: list[dict[str, Any]],
    source_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    by_title = {
        normalized_title(str(item.get("title", ""))): item
        for item in records
        if item.get("title")
    }

    selected: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def add(item: dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        source_path = resolve_source_path(item, source_dir)
        if source_path is None:
            return
        source_path = source_path.resolve()
        if source_path in seen_paths:
            return
        enriched = dict(item)
        enriched["_source_path"] = source_path
        selected.append(enriched)
        seen_paths.add(source_path)

    for title in CURATED_TITLES:
        item = by_title.get(normalized_title(title))
        if item:
            add(item)

    for item in records:
        add(item)
        if len(selected) >= limit:
            break

    return selected


def link_or_copy(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        try:
            if target.samefile(source):
                return "existing"
        except OSError:
            pass
        target.unlink()

    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"

    if mode == "symlink":
        target.symlink_to(source.resolve())
        return "symlink"

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy-fallback"


def build_manifest_item(
    item: dict[str, Any],
    source_path: Path,
    target_path: Path,
    materialized_as: str,
) -> dict[str, Any]:
    return {
        "arxiv_id": item.get("arxiv_id"),
        "title": item.get("title"),
        "authors": item.get("authors", []),
        "published": item.get("published"),
        "updated": item.get("updated"),
        "categories": item.get("categories", []),
        "pdf_url": item.get("pdf_url"),
        "entry_id": item.get("entry_id"),
        "filename": target_path.name,
        "source_path": str(source_path),
        "demo_path": str(target_path),
        "materialized_as": materialized_as,
    }


def build_demo_questions(manifest_items: list[dict[str, Any]], max_questions: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    for index, item in enumerate(manifest_items):
        title = item.get("title")
        if not title:
            continue

        template = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
        questions.append(
            {
                "question": template.format(title=title),
                "expected_source": item.get("filename"),
                "arxiv_id": item.get("arxiv_id"),
                "title": title,
            }
        )

        if len(questions) >= max_questions:
            break

    return questions


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic 300-document technical demo dataset."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="JSONL metadata file. Default: <source-dir>/metadata.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="hardlink keeps the demo folder separate without duplicating PDF bytes.",
    )
    parser.add_argument("--max-questions", type=int, default=12)
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    source_dir = args.source_dir
    metadata_file = args.metadata_file or source_dir / "metadata.jsonl"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(metadata_file)
    selected = choose_records(records, source_dir, args.limit)

    if len(selected) < args.limit:
        raise SystemExit(
            f"Only found {len(selected)} usable PDF files, expected {args.limit}."
        )

    manifest_items: list[dict[str, Any]] = []
    materialized_counts: dict[str, int] = {}

    for item in selected:
        source_path = item["_source_path"]
        target_path = output_dir / source_path.name
        materialized_as = link_or_copy(source_path, target_path, args.mode)
        materialized_counts[materialized_as] = materialized_counts.get(materialized_as, 0) + 1
        manifest_items.append(
            build_manifest_item(
                item=item,
                source_path=source_path,
                target_path=target_path,
                materialized_as=materialized_as,
            )
        )

    manifest = {
        "name": "technical_300",
        "description": "Demo dataset with 300 technical arXiv PDF documents for RAG upload, indexing, chat, and source-link demonstration.",
        "source_dir": str(source_dir),
        "metadata_file": str(metadata_file),
        "output_dir": str(output_dir),
        "document_count": len(manifest_items),
        "materialized_counts": materialized_counts,
        "documents": manifest_items,
    }

    questions = build_demo_questions(manifest_items, args.max_questions)

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "demo_questions.json", questions)

    print("Demo dataset prepared")
    print(f"Output dir: {output_dir}")
    print(f"Documents: {len(manifest_items)}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Demo questions: {output_dir / 'demo_questions.json'}")
    print(f"Materialized: {materialized_counts}")


if __name__ == "__main__":
    main()
