import json
import os
import re
import sys
import time
import random
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Можно менять категории под свои тесты
CATEGORIES = [
    ("cs.AI", 400),
    ("cs.CL", 400),
    ("cs.LG", 400),
    ("cs.IR", 300),
]

OUTPUT_DIR = Path("data/test_documents/arxiv_1500")
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"

# arXiv в примерах рекомендует 3 секунды между API-вызовами
API_SLEEP_SECONDS = 3

# Размер одной страницы выдачи API
BATCH_SIZE = 100

# Таймауты/повторы
HTTP_TIMEOUT = 60
MAX_RETRIES = 3

USER_AGENT = "RAG-Day13-Test/1.0 (educational testing; contact: local-user)"


def safe_filename(text: str, max_len: int = 140) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", text)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text or "untitled"


def http_get(url: str, timeout: int = HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_feed(search_query: str, start: int, max_results: int) -> bytes:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = ARXIV_API_URL + "?" + urllib.parse.urlencode(params)
    return http_get(url)


def parse_entries(feed_xml: bytes) -> list[dict]:
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(feed_xml)
    entries = []

    for entry in root.findall("atom:entry", ns):
        entry_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        published = entry.findtext("atom:published", default="", namespaces=ns).strip()
        updated = entry.findtext("atom:updated", default="", namespaces=ns).strip()

        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns).strip()
            if name:
                authors.append(name)

        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.attrib.get("term", "").strip()
            if term:
                categories.append(term)

        # entry_id обычно вида http://arxiv.org/abs/1234.5678v1
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""

        if arxiv_id and pdf_url:
            entries.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "published": published,
                    "updated": updated,
                    "authors": authors,
                    "categories": categories,
                    "pdf_url": pdf_url,
                    "entry_id": entry_id,
                }
            )

    return entries


def download_file(url: str, dest_path: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = http_get(url)
            if not data.startswith(b"%PDF"):
                # иногда может вернуться HTML/ошибка вместо PDF
                raise ValueError("Response is not a PDF")
            dest_path.write_bytes(data)
            return True
        except Exception as e:
            print(f"  retry {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(1 + attempt + random.random())
    return False


def already_downloaded_ids() -> set[str]:
    downloaded = set()
    if METADATA_FILE.exists():
        with METADATA_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    arxiv_id = obj.get("arxiv_id")
                    if arxiv_id:
                        downloaded.add(arxiv_id)
                except json.JSONDecodeError:
                    continue
    return downloaded


def append_metadata(obj: dict) -> None:
    with METADATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def collect_for_category(category: str, target_count: int, existing_ids: set[str]) -> int:
    print(f"\n=== Category: {category} | target: {target_count} ===")
    downloaded_count = 0
    start = 0

    while downloaded_count < target_count:
        remaining = target_count - downloaded_count
        batch_size = min(BATCH_SIZE, remaining)

        search_query = f"cat:{category}"
        print(f"Fetching feed: start={start}, batch={batch_size}")

        try:
            feed_xml = fetch_feed(search_query, start=start, max_results=batch_size)
            entries = parse_entries(feed_xml)
        except Exception as e:
            print(f"Feed error for {category} start={start}: {e}")
            print("Sleeping and retrying next page...")
            time.sleep(API_SLEEP_SECONDS)
            start += batch_size
            continue

        if not entries:
            print("No more entries returned by API.")
            break

        for entry in entries:
            arxiv_id = entry["arxiv_id"]
            if arxiv_id in existing_ids:
                continue

            title_part = safe_filename(entry["title"])
            filename = f"{arxiv_id} - {title_part}.pdf"
            pdf_path = OUTPUT_DIR / filename

            print(f"Downloading: {filename}")

            ok = download_file(entry["pdf_url"], pdf_path)
            if not ok:
                print(f"  failed: {arxiv_id}")
                continue

            meta = {
                "arxiv_id": arxiv_id,
                "title": entry["title"],
                "authors": entry["authors"],
                "published": entry["published"],
                "updated": entry["updated"],
                "categories": entry["categories"],
                "pdf_url": entry["pdf_url"],
                "entry_id": entry["entry_id"],
                "local_path": str(pdf_path),
            }
            append_metadata(meta)
            existing_ids.add(arxiv_id)
            downloaded_count += 1

            if downloaded_count >= target_count:
                break

        start += batch_size
        print(f"Downloaded in {category}: {downloaded_count}/{target_count}")

        # Важно: пауза между API-вызовами, а не между PDF-загрузками
        if downloaded_count < target_count:
            time.sleep(API_SLEEP_SECONDS)

    return downloaded_count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = already_downloaded_ids()
    if existing_ids:
        print(f"Found existing metadata with {len(existing_ids)} downloaded docs.")

    total_target = sum(count for _, count in CATEGORIES)
    total_before = len(existing_ids)

    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Total target across categories: {total_target}")

    total_new = 0
    for category, target_count in CATEGORIES:
        got = collect_for_category(category, target_count, existing_ids)
        total_new += got

    total_after = len(existing_ids)

    print("\n=== DONE ===")
    print(f"Previously downloaded: {total_before}")
    print(f"Newly downloaded now: {total_new}")
    print(f"Total unique downloaded: {total_after}")
    print(f"Files saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)