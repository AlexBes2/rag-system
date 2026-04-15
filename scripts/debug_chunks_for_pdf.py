from pathlib import Path
from pprint import pprint

from backend.services.document_parser import extract_document_sections
from backend.services.text_processing import chunk_document_sections

PDF_PATH = Path("data/test_documents/arxiv_1500/learning_reflect_test.pdf")

def main():
    sections = extract_document_sections(PDF_PATH)

    print("=" * 80)
    print("SECTIONS")
    print("=" * 80)
    print(f"sections count: {len(sections)}")
    for i, section in enumerate(sections[:3], start=1):
        print(f"\n--- SECTION {i} ---")
        print("page:", section.get("page"))
        print("source_type:", section.get("source_type"))
        print("ocr_used:", section.get("ocr_used"))
        print(section.get("text", "")[:2000])

    chunks = chunk_document_sections(sections)

    print()
    print("=" * 80)
    print("CHUNKS")
    print("=" * 80)
    print(f"chunks count: {len(chunks)}")

    for i, chunk in enumerate(chunks[:10], start=1):
        print(f"\n--- CHUNK {i} ---")
        if isinstance(chunk, dict):
            for key in ["document_name", "page", "chunk_id", "section_index"]:
                if key in chunk:
                    print(f"{key}: {chunk[key]}")
            print(chunk.get("text", "")[:1500])
        else:
            pprint(chunk)

if __name__ == "__main__":
    main()