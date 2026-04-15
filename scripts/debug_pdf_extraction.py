from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path("data/test_documents/arxiv_1500/learning_reflect_test.pdf")

def main():
    reader = PdfReader(str(PDF_PATH))
    page = reader.pages[0]

    default_text = (page.extract_text() or "").strip()

    try:
        layout_text = (page.extract_text(extraction_mode="layout") or "").strip()
    except TypeError:
        layout_text = "[layout mode not supported by this pypdf version]"

    print("=" * 80)
    print("DEFAULT")
    print("=" * 80)
    print(default_text[:3000])

    print()
    print("=" * 80)
    print("LAYOUT")
    print("=" * 80)
    print(layout_text[:3000])

    print()
    print("=" * 80)
    print("DEFAULT FIRST 30 LINES")
    print("=" * 80)
    for i, line in enumerate(default_text.splitlines()[:30], start=1):
        print(f"{i:02d}: {line}")

    print()
    print("=" * 80)
    print("LAYOUT FIRST 30 LINES")
    print("=" * 80)
    if isinstance(layout_text, str):
        for i, line in enumerate(layout_text.splitlines()[:30], start=1):
            print(f"{i:02d}: {line}")

if __name__ == "__main__":
    main()