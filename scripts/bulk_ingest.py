#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/test_documents/arxiv_1500")
DEFAULT_UPLOAD_URL = "http://127.0.0.1:8000/upload"
DEFAULT_REPORTS_DIR = Path("data/reports")


def collect_files(input_dir: Path, allowed_extensions: set[str]) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_dir}")

    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_extensions
    ]
    files.sort()
    return files


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def response_error_text(payload, raw_body: str, stderr: str) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, list) and detail:
            try:
                return json.dumps(detail, ensure_ascii=False)
            except Exception:
                return str(detail)

    raw_body = (raw_body or "").strip()
    stderr = (stderr or "").strip()

    if raw_body:
        return raw_body[:500]
    if stderr:
        return stderr[:500]
    return "Неизвестная ошибка"


def make_curl_form_file(field_name: str, file_path: Path) -> str:
    """
    Безопасно формирует аргумент для curl -F, чтобы пути с запятыми, пробелами
    и кавычками не ломали загрузку.
    """
    path_str = str(file_path)
    path_str = path_str.replace("\\", "\\\\").replace('"', '\\"')
    return f'{field_name}=@"{path_str}"'


def run_curl_upload(url: str, field_name: str, files: list[Path], max_time: int = 3600) -> dict:
    marker = "__HTTP_STATUS__:"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "--connect-timeout",
        "10",
        "--max-time",
        str(max_time),
        "-w",
        f"\n{marker}%{{http_code}}",
    ]

    for file_path in files:
        cmd.extend(["-F", make_curl_form_file(field_name, file_path)])

    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    status_code = 0
    raw_body = stdout

    if marker in stdout:
        body, _, status_part = stdout.rpartition(marker)
        raw_body = body
        try:
            status_code = int(status_part.strip())
        except ValueError:
            status_code = 0

    payload = None
    body_text = raw_body.strip()
    if body_text:
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError:
            payload = None

    return {
        "status_code": status_code,
        "payload": payload,
        "raw_body": raw_body,
        "stderr": stderr,
        "elapsed_sec": elapsed,
        "returncode": result.returncode,
        "command": cmd,
    }


def parse_multi_result(batch: list[Path], response: dict) -> list[dict]:
    status_code = response["status_code"]
    payload = response["payload"]
    raw_body = response["raw_body"]
    stderr = response["stderr"]

    if not (200 <= status_code < 300):
        error_text = response_error_text(payload, raw_body, stderr)
        return [
            {
                "filename": p.name,
                "path": str(p),
                "success": False,
                "error": error_text,
            }
            for p in batch
        ]

    if isinstance(payload, dict):
        uploaded_items = payload.get("uploaded")
        error_items = payload.get("errors")

        success_names = set()
        errors_by_name = {}

        if isinstance(uploaded_items, list):
            for item in uploaded_items:
                if isinstance(item, dict):
                    name = item.get("filename") or item.get("name")
                    if name:
                        success_names.add(name)

        if isinstance(error_items, list):
            for item in error_items:
                if isinstance(item, dict):
                    name = item.get("filename") or item.get("name")
                    err = item.get("error") or item.get("detail") or "Ошибка"
                    if name:
                        errors_by_name[name] = str(err)

        if success_names or errors_by_name:
            results = []
            for p in batch:
                if p.name in errors_by_name:
                    results.append({
                        "filename": p.name,
                        "path": str(p),
                        "success": False,
                        "error": errors_by_name[p.name],
                    })
                elif p.name in success_names:
                    results.append({
                        "filename": p.name,
                        "path": str(p),
                        "success": True,
                        "error": None,
                    })
                else:
                    results.append({
                        "filename": p.name,
                        "path": str(p),
                        "success": True,
                        "error": None,
                    })
            return results

    return [
        {
            "filename": p.name,
            "path": str(p),
            "success": True,
            "error": None,
        }
        for p in batch
    ]


def parse_single_result(file_path: Path, response: dict) -> dict:
    status_code = response["status_code"]
    payload = response["payload"]
    raw_body = response["raw_body"]
    stderr = response["stderr"]

    if 200 <= status_code < 300:
        return {
            "filename": file_path.name,
            "path": str(file_path),
            "success": True,
            "error": None,
        }

    return {
        "filename": file_path.name,
        "path": str(file_path),
        "success": False,
        "error": response_error_text(payload, raw_body, stderr),
    }


def detect_mode(upload_url: str, sample_batch: list[Path]) -> tuple[str, str]:
    """
    Возвращает режим:
    - ("multi", "files")
    - ("single", "file")
    - ("single", "files")
    """

    multi_try = run_curl_upload(upload_url, "files", sample_batch)
    if 200 <= multi_try["status_code"] < 300:
        return "multi", "files"

    one = sample_batch[0]

    single_file_try = run_curl_upload(upload_url, "file", [one])
    if 200 <= single_file_try["status_code"] < 300:
        return "single", "file"

    single_files_try = run_curl_upload(upload_url, "files", [one])
    if 200 <= single_files_try["status_code"] < 300:
        return "single", "files"

    debug = {
        "multi_status": multi_try["status_code"],
        "multi_returncode": multi_try["returncode"],
        "multi_body": (multi_try["raw_body"] or "").strip()[:500],
        "multi_stderr": (multi_try["stderr"] or "").strip()[:500],
        "single_file_status": single_file_try["status_code"],
        "single_file_returncode": single_file_try["returncode"],
        "single_file_body": (single_file_try["raw_body"] or "").strip()[:500],
        "single_file_stderr": (single_file_try["stderr"] or "").strip()[:500],
        "single_files_status": single_files_try["status_code"],
        "single_files_returncode": single_files_try["returncode"],
        "single_files_body": (single_files_try["raw_body"] or "").strip()[:500],
        "single_files_stderr": (single_files_try["stderr"] or "").strip()[:500],
    }

    raise RuntimeError(
        "Не удалось определить формат загрузки для /upload.\n"
        + json.dumps(debug, ensure_ascii=False, indent=2)
    )


def ensure_reports_dir(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Массовая загрузка документов в RAG backend через /upload")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Папка с документами. По умолчанию: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--upload-url",
        type=str,
        default=DEFAULT_UPLOAD_URL,
        help=f"URL backend upload endpoint. По умолчанию: {DEFAULT_UPLOAD_URL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Размер батча для многофайловой загрузки. По умолчанию: 10",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Ограничить количество файлов для теста. 0 = без ограничения",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help=f"Куда сохранять JSON-отчёт. По умолчанию: {DEFAULT_REPORTS_DIR}",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".pdf", ".docx"],
        help='Какие расширения брать. По умолчанию: ".pdf .docx"',
    )
    args = parser.parse_args()

    allowed_extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.extensions
    }

    if args.batch_size < 1:
        print("Ошибка: --batch-size должен быть >= 1")
        sys.exit(1)

    ensure_reports_dir(args.reports_dir)

    try:
        files = collect_files(args.input_dir, allowed_extensions)
    except Exception as e:
        print(f"Ошибка при чтении папки: {e}")
        sys.exit(1)

    if args.limit > 0:
        files = files[:args.limit]

    if not files:
        print(f"В папке {args.input_dir} нет подходящих файлов: {sorted(allowed_extensions)}")
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}")
    print(f"Папка: {args.input_dir}")
    print(f"Upload URL: {args.upload_url}")
    print(f"Расширения: {', '.join(sorted(allowed_extensions))}")

    sample_batch = files[: min(len(files), max(2, args.batch_size))]
    try:
        mode, field_name = detect_mode(args.upload_url, sample_batch)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    print(f"Определён режим загрузки: {mode} | field={field_name}")

    started_total = time.perf_counter()

    all_results = []
    total_success = 0
    total_failed = 0

    if mode == "multi":
        batches = list(chunked(files, args.batch_size))
        total_batches = len(batches)

        for batch_index, batch in enumerate(batches, start=1):
            response = run_curl_upload(args.upload_url, field_name, batch)
            batch_results = parse_multi_result(batch, response)

            batch_success = sum(1 for item in batch_results if item["success"])
            batch_failed = len(batch_results) - batch_success

            total_success += batch_success
            total_failed += batch_failed
            all_results.extend(batch_results)

            elapsed_total = time.perf_counter() - started_total
            processed = total_success + total_failed
            rate = (processed / elapsed_total) * 60 if elapsed_total > 0 else 0

            print(
                f"[batch {batch_index}/{total_batches}] "
                f"processed={processed}/{len(files)} | "
                f"ok={total_success} | err={total_failed} | "
                f"batch_time={format_seconds(response['elapsed_sec'])}s | "
                f"rate={rate:.2f} files/min"
            )
    else:
        total_files = len(files)

        for index, file_path in enumerate(files, start=1):
            response = run_curl_upload(args.upload_url, field_name, [file_path])
            item_result = parse_single_result(file_path, response)

            if item_result["success"]:
                total_success += 1
            else:
                total_failed += 1

            all_results.append(item_result)

            elapsed_total = time.perf_counter() - started_total
            processed = total_success + total_failed
            rate = (processed / elapsed_total) * 60 if elapsed_total > 0 else 0

            print(
                f"[file {index}/{total_files}] "
                f"{file_path.name} | "
                f"ok={total_success} | err={total_failed} | "
                f"time={format_seconds(response['elapsed_sec'])}s | "
                f"rate={rate:.2f} files/min"
            )

    elapsed_total = time.perf_counter() - started_total
    avg_per_file = elapsed_total / len(files) if files else 0
    files_per_min = (len(files) / elapsed_total) * 60 if elapsed_total > 0 else 0

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "upload_url": args.upload_url,
        "mode": mode,
        "field_name": field_name,
        "batch_size": args.batch_size if mode == "multi" else 1,
        "extensions": sorted(allowed_extensions),
        "total_files": len(files),
        "success_count": total_success,
        "failed_count": total_failed,
        "total_time_sec": round(elapsed_total, 3),
        "avg_time_per_file_sec": round(avg_per_file, 3),
        "files_per_min": round(files_per_min, 3),
        "results": all_results,
    }

    report_name = f"bulk_ingest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = args.reports_dir / report_name
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== ГОТОВО ===")
    print(f"Всего файлов: {len(files)}")
    print(f"Успешно: {total_success}")
    print(f"Ошибок: {total_failed}")
    print(f"Общее время: {format_seconds(elapsed_total)} сек")
    print(f"Среднее на файл: {format_seconds(avg_per_file)} сек")
    print(f"Скорость: {files_per_min:.2f} files/min")
    print(f"Отчёт: {report_path}")


if __name__ == "__main__":
    main()