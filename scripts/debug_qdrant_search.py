from pathlib import Path
import sys
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.qdrant_service import search_similar_chunks

QUESTION = 'Who are the authors of the paper "Learning to Reflect and Correct: Towards Better Decoding Trajectories for Large-Scale Generative Recommendation"?'

def main():
    results = search_similar_chunks(question=QUESTION, limit=5)

    print("=" * 80)
    print("RESULT COUNT")
    print("=" * 80)
    print(len(results))

    for i, item in enumerate(results, start=1):
        print()
        print("=" * 80)
        print(f"RESULT {i}")
        print("=" * 80)

        if isinstance(item, dict):
            for key in ["score", "document_name", "page", "chunk_id", "section_index"]:
                if key in item:
                    print(f"{key}: {item[key]}")
            print()
            print((item.get("text") or "")[:1500])
        else:
            pprint(item)

if __name__ == "__main__":
    main()