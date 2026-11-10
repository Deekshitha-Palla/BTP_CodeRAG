
import os
import re
import pickle
import ollama
import chromadb
import pymupdf4llm
import fitz  # still used for OCR fallback on image-only pages
import pytesseract
from PIL import Image
import io
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:4b-instruct"

VERIFIED_TABLES = []


def build_verified_chunks(verified_tables=VERIFIED_TABLES):
    chunks = []
    for i, vt in enumerate(verified_tables):
        chunks.append({
            "text": vt["text"],
            "page_num": vt["page_num"],
            "source_file": vt["source_file"],
            "extraction_method": "verified",
            "chunk_id": f"verified_{i:03d}",
            "_keywords": vt["keywords"],
        })
    return chunks


def force_include_verified(query, chunks, candidate_ids):
    query_lower = query.lower()
    forced = []
    for c in chunks:
        if c.get("extraction_method") == "verified":
            if any(kw.lower() in query_lower for kw in c.get("_keywords", [])):
                forced.append(c["chunk_id"])
    return forced + [cid for cid in candidate_ids if cid not in forced]


# ------------------------- 1. DATA INGESTION + CHUNKING --------------------

def ingest_and_chunk(pdf_path, max_chars=1200, ocr_dpi=300):
    chunks = []
    chunk_counter = 0
    source_name = os.path.basename(pdf_path)

    md_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    doc = fitz.open(pdf_path)  # for OCR fallback checks

    for page_data in md_pages:
        page_num = page_data.get("metadata", {}).get("page_number", 0)
        page_text = page_data.get("text", "").strip()

        if not page_text:
            # no text extracted at all implies likely an image-only page, OCR it
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=ocr_dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img).strip()
            if ocr_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:05d}",
                    "text": ocr_text,
                    "page_num": page_num,
                    "source_file": source_name,
                    "extraction_method": "ocr",
                })
                chunk_counter += 1
            else:
                print(f"  Warning: page {page_num} produced no text "
                      f"(native or OCR) — likely blank or unreadable.")
            continue
        blocks = page_text.split('\n\n')
        buffer = ""

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if len(buffer) + len(block) > max_chars and buffer:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:05d}",
                    "text": buffer.strip(),
                    "page_num": page_num,
                    "source_file": source_name,
                    "extraction_method": "markdown",
                })
                chunk_counter += 1
                buffer = ""

            buffer = (buffer + "\n\n" + block).strip()

            if len(buffer) > max_chars:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:05d}",
                    "text": buffer.strip(),
                    "page_num": page_num,
                    "source_file": source_name,
                    "extraction_method": "markdown",
                })
                chunk_counter += 1
                buffer = ""

        if buffer:
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "text": buffer.strip(),
                "page_num": page_num,
                "source_file": source_name,
                "extraction_method": "markdown",
            })
            chunk_counter += 1

    doc.close()
    return enforce_max_chunk_size(chunks, max_chars=max_chars)


def enforce_max_chunk_size(chunks, max_chars=1200):
    fixed = []
    counter = 0
    for c in chunks:
        text = c["text"]
        if len(text) <= max_chars:
            c["chunk_id"] = f"chunk_{counter:05d}"
            fixed.append(c)
            counter += 1
            continue

        lines = text.split("\n")
        buffer = ""
        for line in lines:
            if len(buffer) + len(line) > max_chars and buffer:
                fixed.append({**c, "text": buffer.strip(), "chunk_id": f"chunk_{counter:05d}"})
                counter += 1
                buffer = ""
            if len(line) > max_chars:
                # a single line alone is too long — hard-slice it
                for i in range(0, len(line), max_chars):
                    piece = line[i:i + max_chars]
                    fixed.append({**c, "text": piece, "chunk_id": f"chunk_{counter:05d}"})
                    counter += 1
                buffer = ""
            else:
                buffer = (buffer + "\n" + line).strip()
        if buffer:
            fixed.append({**c, "text": buffer.strip(), "chunk_id": f"chunk_{counter:05d}"})
            counter += 1

    return fixed


#  DENSE (EMBEDDING) SEARCH 

def generate_embeddings(chunks, query, collection_name="structural_analysis",
                         persist_dir="./chroma_store", max_retries=3):
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(collection_name)
        existing_ids = set(collection.get()["ids"])
    except Exception:
        collection = client.create_collection(collection_name)
        existing_ids = set()

    to_embed = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  {len(existing_ids)} chunks already embedded, {len(to_embed)} remaining.")

    for i, c in enumerate(to_embed, start=1):
        text_to_embed = c["text"]
        for attempt in range(1, max_retries + 1):
            try:
                emb = ollama.embeddings(model=EMBED_MODEL, prompt=text_to_embed)["embedding"]
                collection.add(ids=[c["chunk_id"]], embeddings=[emb], documents=[c["text"]])
                break
            except Exception as e:
                if "context length" in str(e).lower() and attempt < max_retries:
                    # fallback: truncate and retry rather than losing the chunk entirely
                    text_to_embed = text_to_embed[: len(text_to_embed) // 2]
                    print(f"    {c['chunk_id']} too long, truncating and retrying "
                          f"(new length {len(text_to_embed)})...")
                    continue
                print(f"    Retry {attempt}/{max_retries} for {c['chunk_id']} after error: {e}")
                if attempt == max_retries:
                    raise
        if i % 50 == 0 or i == len(to_embed):
            print(f"  Embedded {i}/{len(to_embed)} new chunks...")

    q_embed = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    result = collection.query(query_embeddings=[q_embed], n_results=min(50, len(chunks)))
    return result["ids"][0]


# SPARSE (BM25) SEARCH 

def bm25_search(chunks, query):
    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked]


# RRF FUSION 

def reciprocal_rank_fusion(ranked_lists, k=60):
    score_chunk = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            score_chunk[chunk_id] = score_chunk.get(chunk_id, 0.0) + (1.0 / (rank + k + 1))
    fused = sorted(score_chunk.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


# METADATA FILTERING

def metadata_filtering(fused_ids, chunks, source_file=None, page_range=None):
    lookup = {c["chunk_id"]: c for c in chunks}
    filtered = []
    for chunk_id in fused_ids:
        c = lookup[chunk_id]
        if source_file and c["source_file"] != source_file:
            continue
        if page_range and not (page_range[0] <= c["page_num"] <= page_range[1]):
            continue
        filtered.append(chunk_id)
    return filtered


#  RERANKING 

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reranking(candidate_ids, chunks, query, top_k=5):
    lookup = {c["chunk_id"]: c for c in chunks}
    candidates = [lookup[cid] for cid in candidate_ids if cid in lookup]
    model = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]


#  Agentic Loop

def agent(query, reranked_chunks):
    context = "\n\n".join(
        f"[Page {c['page_num']}, {c['source_file']}]\n{c['text']}"
        for c in reranked_chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite page numbers. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = ollama.generate(model=GEN_MODEL, prompt=prompt)
    return response["response"]


# Bump this any time ingest_and_chunk() or enforce_max_chunk_size() logic
# changes — it invalidates old caches automatically, no manual deletion needed.
EXTRACTION_LOGIC_VERSION = "pymupdf4llm-v1"


def compute_source_fingerprint(pdf_path, logic_version=EXTRACTION_LOGIC_VERSION):
    """Fingerprint = hash of (file content + extraction logic version).
    If either changes, the fingerprint changes, and the cache auto-rebuilds —
    removing any need to remember to manually delete cache folders."""
    import hashlib
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    hasher.update(logic_version.encode())
    return hasher.hexdigest()


# multi turn query

if __name__ == "__main__":
    print(f"=== chatbot_v2.py — build: {EXTRACTION_LOGIC_VERSION} ===\n")
    path = r"C:\Users\DELL\Downloads\trail\data\is.456.2000.pdf"
    chunks_cache_path = "./chunks_cache.pkl"

    current_fingerprint = compute_source_fingerprint(path)
    cache_valid = False

    if os.path.exists(chunks_cache_path):
        with open(chunks_cache_path, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data.get("fingerprint") == current_fingerprint:
            chunks = cached_data["chunks"]
            cache_valid = True
            print(f"  -> Loaded {len(chunks)} chunks from cache (fingerprint matches).")
        else:
            print("  -> Cache fingerprint mismatch (PDF or extraction logic changed) "
                  "— rebuilding automatically.")

    if not cache_valid:
        print("Extracting PDF via pymupdf4llm...")
        chunks = ingest_and_chunk(path)

        verified_chunks = build_verified_chunks()
        chunks = chunks + verified_chunks
        print(f"  -> {len(chunks) - len(verified_chunks)} chunks extracted "
              f"+ {len(verified_chunks)} verified entries = {len(chunks)} total")

        with open(chunks_cache_path, "wb") as f:
            pickle.dump({"fingerprint": current_fingerprint, "chunks": chunks}, f)
        print(f"  -> Cached to {chunks_cache_path}.\n")
        import shutil
        if os.path.exists("./chroma_store"):
            shutil.rmtree("./chroma_store")
            print("  -> Cleared ./chroma_store (fingerprint changed, old embeddings invalid).\n")

    while True:
        query = input("\n[User] (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break

        print("  Searching...")
        dense_ranking = generate_embeddings(chunks, query)
        sparse_ranking = bm25_search(chunks, query)
        fused_ids = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        filtered_ids = metadata_filtering(fused_ids, chunks)
        filtered_ids = force_include_verified(query, chunks, filtered_ids)

        print("  Reranking...")
        top_chunks = reranking(filtered_ids[:25], chunks, query, top_k=5)

        print("  Generating answer...")
        response = agent(query, top_chunks)

        print("\n[Assistant]:", response)
        print("\nSources used:")
        for c in top_chunks:
            method = c.get("extraction_method", "unknown")
            print(f"  Page {c['page_num']} ({c['source_file']}) [{method}]")