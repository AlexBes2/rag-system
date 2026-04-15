import json
import shutil
from pathlib import Path

REPORT = Path("data/reports/bulk_ingest_report_20260311_135923.json")
TARGET_DIR = Path("data/retry_failed")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

with REPORT.open("r", encoding="utf-8") as f:
    data = json.load(f)

copied = 0
for item in data["results"]:
    if item.get("success") is False:
        src = Path(item["path"])
        if src.exists():
            shutil.copy2(src, TARGET_DIR / src.name)
            copied += 1

print(f"Скопировано файлов: {copied}")
print(f"Папка: {TARGET_DIR}")