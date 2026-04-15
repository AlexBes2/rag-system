#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_REPORT = Path("data/reports/bulk_ingest_report_20260311_135923.json")


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Отчёт не найден: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percent(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100.0


def shorten(text: str, max_len: int = 140) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def analyze_report(data: dict) -> dict:
    total_files = int(data.get("total_files", 0))
    success_count = int(data.get("success_count", 0))
    failed_count = int(data.get("failed_count", 0))
    total_time_sec = float(data.get("total_time_sec", 0))
    avg_time_per_file_sec = float(data.get("avg_time_per_file_sec", 0))
    files_per_min = float(data.get("files_per_min", 0))
    mode = data.get("mode", "unknown")
    batch_size = data.get("batch_size", "unknown")
    input_dir = data.get("input_dir", "")
    upload_url = data.get("upload_url", "")
    timestamp = data.get("timestamp", "")

    results = data.get("results", [])
    failed_items = [item for item in results if not item.get("success", False)]

    error_counter = Counter()
    files_by_error = defaultdict(list)

    for item in failed_items:
        error_text = (item.get("error") or "Неизвестная ошибка").strip()
        filename = item.get("filename") or item.get("path") or "unknown"
        error_counter[error_text] += 1
        files_by_error[error_text].append(filename)

    return {
        "timestamp": timestamp,
        "input_dir": input_dir,
        "upload_url": upload_url,
        "mode": mode,
        "batch_size": batch_size,
        "total_files": total_files,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": percent(success_count, total_files),
        "failed_rate": percent(failed_count, total_files),
        "total_time_sec": total_time_sec,
        "avg_time_per_file_sec": avg_time_per_file_sec,
        "files_per_min": files_per_min,
        "error_counter": error_counter,
        "files_by_error": files_by_error,
    }


def make_summary(stats: dict) -> list[str]:
    success_rate = stats["success_rate"]
    failed_count = stats["failed_count"]
    speed = stats["files_per_min"]

    summary = []

    if success_rate >= 98:
        summary.append("Стабильность индексации отличная.")
    elif success_rate >= 95:
        summary.append("Стабильность хорошая, но ошибки ещё стоит добить.")
    else:
        summary.append("Стабильность средняя, сначала лучше уменьшить количество ошибок.")

    if speed >= 40:
        summary.append("Скорость индексации высокая.")
    elif speed >= 20:
        summary.append("Скорость индексации нормальная для локального RAG.")
    else:
        summary.append("Скорость индексации низковата, есть смысл искать узкое место.")

    if failed_count == 0:
        summary.append("Можно переходить к тестам качества поиска и chunk size.")
    else:
        summary.append("Перед сравнением chunk size лучше понять природу ошибок.")

    return summary


def print_report(stats: dict, top_errors: int, files_per_error: int) -> None:
    print("=== АНАЛИЗ BULK INGEST REPORT ===")
    print(f"Время отчёта: {stats['timestamp']}")
    print(f"Папка: {stats['input_dir']}")
    print(f"Upload URL: {stats['upload_url']}")
    print(f"Режим: {stats['mode']}")
    print(f"Batch size: {stats['batch_size']}")
    print()

    print("=== ОСНОВНЫЕ МЕТРИКИ ===")
    print(f"Всего файлов: {stats['total_files']}")
    print(f"Успешно: {stats['success_count']} ({stats['success_rate']:.2f}%)")
    print(f"Ошибок: {stats['failed_count']} ({stats['failed_rate']:.2f}%)")
    print(f"Общее время: {stats['total_time_sec']:.2f} сек")
    print(f"Среднее на файл: {stats['avg_time_per_file_sec']:.2f} сек")
    print(f"Скорость: {stats['files_per_min']:.2f} files/min")
    print()

    print("=== КРАТКИЙ ВЫВОД ===")
    for line in make_summary(stats):
        print(f"- {line}")
    print()

    error_counter = stats["error_counter"]
    files_by_error = stats["files_by_error"]

    if not error_counter:
        print("=== ОШИБКИ ===")
        print("Ошибок нет.")
        return

    print("=== ТОП ОШИБОК ===")
    for index, (error_text, count) in enumerate(error_counter.most_common(top_errors), start=1):
        print(f"{index}. {count}x | {shorten(error_text, 180)}")

        sample_files = files_by_error[error_text][:files_per_error]
        for filename in sample_files:
            print(f"   - {filename}")

        extra = len(files_by_error[error_text]) - len(sample_files)
        if extra > 0:
            print(f"   ... и ещё {extra} файлов")
        print()


def main():
    parser = argparse.ArgumentParser(description="Анализ отчёта bulk ingest")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Путь к JSON-отчёту. По умолчанию: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=10,
        help="Сколько типов ошибок показать. По умолчанию: 10",
    )
    parser.add_argument(
        "--files-per-error",
        type=int,
        default=5,
        help="Сколько примеров файлов показывать на каждую ошибку. По умолчанию: 5",
    )
    args = parser.parse_args()

    try:
        data = load_report(args.report)
        stats = analyze_report(data)
        print_report(stats, args.top_errors, args.files_per_error)
    except Exception as e:
        print(f"Ошибка: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()