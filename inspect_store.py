"""
Quick inspection tool — dump chunk contents by content_type or a keyword,
so you can SEE what's actually stored instead of guessing from query logs.

Usage:
    python inspect_store.py --type table_note
    python inspect_store.py --type table --contains "16A"
    python inspect_store.py --stats          # just print counts per content_type
"""

import argparse
import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help="content_type to filter (table, text, table_note, figure_caption)")
    parser.add_argument("--contains", help="only show chunks whose text contains this substring")
    parser.add_argument("--stats", action="store_true", help="just print counts per content_type")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    if args.stats:
        counts = {}
        for meta in data["metadatas"]:
            ct = meta.get("content_type", "unknown")
            counts[ct] = counts.get(ct, 0) + 1
        print(f"Total chunks: {len(data['ids'])}")
        for ct, n in sorted(counts.items()):
            print(f"  {ct}: {n}")
        return

    shown = 0
    for doc_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        if args.type and meta.get("content_type") != args.type:
            continue
        if args.contains and args.contains.lower() not in doc_text.lower():
            continue
        shown += 1
        print(f"\n{'=' * 70}")
        print(f"id: {doc_id}")
        print(f"metadata: {meta}")
        print(f"text:\n{doc_text}")

    print(f"\n\n({shown} chunk(s) shown)")


if __name__ == "__main__":
    main()