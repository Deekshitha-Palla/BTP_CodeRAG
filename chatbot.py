import os
import re
import json
import pickle
import requests
import numpy as np
import fitz  # PyMuPDF
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

#  CONFIG 

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "qwen3:4b-instruct"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "structural_docs"
BM25_INDEX_PATH = "./bm25_index.pkl"

MIN_CHARS = 200
MAX_CHARS = 800


# ------------------------ 1. PDF INGESTION ------------------------------

def load_pdfs(pdf_paths):
    """Extract text blocks with layout metadata (page, bbox) from PDFs."""
    all_pages = []
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            for block in blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                if block_type == 0 and text.strip():  # 0 = text block
                    all_pages.append({
                        "text": text.strip(),
                        "page_num": page_num,
                        "source_file": os.path.basename(pdf_path),
                        "bbox": (x0, y0, x1, y1),
                    })
        doc.close()
    return all_pages


# ------------------------ 2. LAYOUT-AWARE CHUNKING -----------------------

def chunk_pages(all_pages, min_chars=MIN_CHARS, max_chars=MAX_CHARS):
    """Merge tiny adjacent blocks and split oversized ones, per page."""
    chunks = []
    buffer = ""
    buffer_meta = None

    def flush():
        if buffer.strip():
            if len(buffer) > max_chars:
                for i in range(0, len(buffer), max_chars):
                    piece = buffer[i:i + max_chars].strip()
                    if piece:
                        chunks.append({"text": piece, **buffer_meta})
            else:
                chunks.append({"text": buffer.strip(), **buffer_meta})

    for block in all_pages:
        if buffer_meta and block["page_num"] != buffer_meta["page_num"]:
            flush()
            buffer, buffer_meta = "", None

        buffer = (buffer + " " + block["text"]).strip()
        buffer_meta = {
            "page_num": block["page_num"],
            "source_file": block["source_file"],
        }

        if len(buffer) >= min_chars:
            flush()
            buffer, buffer_meta = "", None

    if buffer.strip() and buffer_meta:
        flush()

    # assign a stable unique id to every chunk
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"chunk_{i:05d}"

    return chunks


# ------------------------ 3. OLLAMA EMBEDDING CALL -----------------------

def get_ollama_embedding(text, model=EMBED_MODEL):
    """Call Ollama's embeddings endpoint for a single piece of text."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ------------------------ 3b. QUERY REWRITING / DECOMPOSITION -----------

def rewrite_query(query, model=GEN_MODEL):
    """Normalize/expand the query before retrieval, and split multi-part
    questions into separate sub-queries. Does NOT answer the question --
    only rewrites/decomposes it, so it can't inject hallucinated facts
    into retrieval the way HyDE would."""
    prompt = (
        "You are an assistant preparing search queries for structural engineering "
        "standards (IS 456).\n"
        "1. Strip out conversational fluff or unrelated adjectives.\n"
        "2. Expand abbreviations (e.g., RCC -> Reinforced Cement Concrete, "
        "M20 -> M20 grade concrete, Fe415 -> Fe415 grade steel).\n"
        "3. If the user asks multiple distinct questions, output each as a "
        "separate search query on its own line.\n"
        "Do NOT answer the question. Only output search queries.\n\n"
        f"Question: {query}\n\nSearch Queries:"
    )
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    lines = resp.json()["response"].strip().split("\n")
    cleaned = [l.strip() for l in lines if l.strip()]
    # fallback: if the model returns nothing usable, don't lose the query
    return cleaned or [query]


# ------------------------ 4. EMBED + STORE (Chroma + BM25) ---------------

def embed_and_store(chunks, collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR):
    """Embed every chunk with nomic-embed-text and store in ChromaDB.
    Also builds and persists a BM25 index over the same chunk text."""
    client = chromadb.PersistentClient(path=persist_dir)

    # fresh collection each ingest run to avoid duplicate ids on re-runs
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    ids, embeddings, documents, metadatas = [], [], [], []
    for c in chunks:
        emb = get_ollama_embedding(c["text"])
        ids.append(c["chunk_id"])
        embeddings.append(emb)
        documents.append(c["text"])
        metadatas.append({"page_num": c["page_num"], "source_file": c["source_file"]})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    # persist BM25 index alongside the raw chunks (needed since Chroma
    # doesn't give us tokenized sparse retrieval)
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    return collection, bm25


def load_index(collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR):
    """Load an already-built Chroma collection + BM25 index from disk."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return collection, data["bm25"], data["chunks"]


def tokenize(text):
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ------------------------ 5. SPARSE (BM25) RETRIEVAL ----------------------

def bm25_search(query, bm25, chunks, top_k=10):
    """Return top_k chunks ranked by BM25 score. Returns list of chunk_ids in rank order."""
    scores = bm25.get_scores(tokenize(query))
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i]["chunk_id"] for i in ranked_idx]


# ------------------------ 6. DENSE RETRIEVAL -------------------------------

def dense_search(query, collection, top_k=10, metadata_filter=None):
    """Return top_k chunk_ids ranked by embedding similarity.
    metadata_filter: optional Chroma 'where' dict, e.g. {"source_file": "AISC.pdf"}"""
    query_emb = get_ollama_embedding(query)
    kwargs = {"query_embeddings": [query_emb], "n_results": top_k}
    if metadata_filter:
        kwargs["where"] = metadata_filter
    results = collection.query(**kwargs)
    return results["ids"][0]  # list of chunk_ids in rank order


# ------------------------ 7. HYBRID FUSION (RRF) ---------------------------

def reciprocal_rank_fusion(ranked_lists, k=60):
    """Combine multiple ranked lists of ids into one fused ranking.
    RRF score for an id = sum over lists of 1 / (k + rank_in_that_list)."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in fused]


def hybrid_search(query, bm25, chunks, collection, top_k=10, metadata_filter=None):
    """Run BM25 + dense search and fuse with RRF. Returns fused chunk_ids."""
    sparse_ids = bm25_search(query, bm25, chunks, top_k=top_k * 2)
    dense_ids = dense_search(query, collection, top_k=top_k * 2, metadata_filter=metadata_filter)
    fused_ids = reciprocal_rank_fusion([sparse_ids, dense_ids])
    return fused_ids[:top_k]


def get_chunks_by_ids(chunk_ids, chunks):
    """Helper: map chunk_ids back to full chunk dicts (text + metadata)."""
    lookup = {c["chunk_id"]: c for c in chunks}
    return [lookup[cid] for cid in chunk_ids if cid in lookup]


# ------------------------ 8. CROSS-ENCODER RERANKING ------------------------

_cross_encoder = None

def get_reranker():
    """Lazy-load the cross-encoder so it's only loaded once per run."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def rerank(query, candidate_chunks, top_k=5):
    """Rerank candidate chunks against a SINGLE query with a cross-encoder.
    candidate_chunks: list of chunk dicts."""
    if not candidate_chunks:
        return []
    model = get_reranker()
    pairs = [[query, c["text"]] for c in candidate_chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def rerank_multi(sub_queries, candidate_chunks, top_k=5):
    """Rerank candidate chunks against MULTIPLE sub-queries.

    Why this exists instead of just joining sub-queries into one string:
    a cross-encoder scores (query, doc) pairs as a single relevance signal.
    If you concatenate "What is Ec for M20?" + "What is formwork striking
    time?" into one string and score a chunk that only answers the first
    question, the encoder sees an irrelevant second half dragging down
    the pair's overall relevance -- the chunk gets penalized for not
    answering a question it was never meant to answer.

    Instead, score every chunk against EVERY sub-query separately and keep
    each chunk's best (max) score across sub-queries. A chunk that's a
    perfect match for just one sub-question gets full credit for that,
    rather than being dragged down by the other sub-question.
    """
    if not candidate_chunks:
        return []
    model = get_reranker()
    best_scores = {}  # chunk_id -> (chunk_dict, best_score)

    for sq in sub_queries:
        pairs = [[sq, c["text"]] for c in candidate_chunks]
        scores = model.predict(pairs)
        for c, s in zip(candidate_chunks, scores):
            cid = c["chunk_id"]
            if cid not in best_scores or s > best_scores[cid][1]:
                best_scores[cid] = (c, s)

    ranked = sorted(best_scores.values(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


# ------------------------ 9. GENERATION (Ollama chat) -----------------------

def generate_answer(query, context_chunks, model=GEN_MODEL):
    """Build a grounded prompt from reranked chunks and call qwen3:4b-instruct."""
    context_text = "\n\n".join(
        f"[Page {c['page_num']}, {c['source_file']}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = (
        "You are a structural engineering assistant. Answer the question using "
        "ONLY the context below. Cite page numbers when relevant. If the answer "
        "isn't in the context, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ------------------------ 10. RETRIEVAL EVALUATION ---------------------------

def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return None
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return None
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(retrieved_ids)


def mrr(retrieved_ids, relevant_ids):
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=None):
    k = k or len(retrieved_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / np.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(eval_set, retrieval_fn, k=5):
    """
    eval_set: list of {"query": str, "relevant_chunk_ids": [str, ...]}
    retrieval_fn: function(query, top_k) -> list of chunk_ids
    Returns averaged Recall@k, Precision@k, MRR, nDCG@k across the eval set.
    """
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    for item in eval_set:
        retrieved = retrieval_fn(item["query"], k)
        relevant = item["relevant_chunk_ids"]
        recalls.append(recall_at_k(retrieved, relevant))
        precisions.append(precision_at_k(retrieved, relevant))
        mrrs.append(mrr(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k=k))

    return {
        f"Recall@{k}": np.mean([r for r in recalls if r is not None]),
        f"Precision@{k}": np.mean([p for p in precisions if p is not None]),
        "MRR": np.mean(mrrs),
        f"nDCG@{k}": np.mean(ndcgs),
    }


# ------------------------ MAIN WIRING ---------------------------------------

if __name__ == "__main__":
    import sys

    PDF_FILES = [
        r"C:\Users\DELL\Downloads\trail\data\is.456.2000.pdf"
    ]

    if not PDF_FILES:
        print("Add your PDF paths to PDF_FILES before running.")
        sys.exit(0)

    print("Step 1: Checking for existing vector index and BM25 store...")
    # NOTE: this cache check only looks for the files' existence, not whether
    # they were built from the current PDF_FILES / chunking params. If you
    # change PDF_FILES, MIN_CHARS, or MAX_CHARS, delete CHROMA_DIR and
    # BM25_INDEX_PATH first, otherwise you'll silently load a stale index.
    if os.path.exists(CHROMA_DIR) and os.path.exists(BM25_INDEX_PATH):
        print("  -> Loading index from cache...")
        collection, bm25, chunks = load_index()
        print(f"  -> Loaded {len(chunks)} chunks from disk.")
    else:
        print("  -> No index found. Loading and chunking PDFs...")
        pages = load_pdfs(PDF_FILES)
        chunks = chunk_pages(pages)
        print(f"  -> {len(pages)} raw blocks -> {len(chunks)} chunks")

        print("  -> Embedding chunks and building indexes (this may take a while on CPU)...")
        collection, bm25 = embed_and_store(chunks)
        print("  -> Chroma collection + BM25 index built and cached.")

    # ---- interactive query loop ----
    print("\nReady. Type a question (or 'quit' to exit, 'eval' to run evaluation).")
    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit"):
            break

        if query.lower() == "eval":
            # Minimal example eval set — replace with your own 10-15 curated pairs
            eval_set = [
                {
                    "query": "What is the allowable bending stress for A992 steel?",
                    "relevant_chunk_ids": [chunks[0]["chunk_id"]],  # placeholder
                },
            ]

            def retrieval_fn(q, k):
                return hybrid_search(q, bm25, chunks, collection, top_k=k)

            metrics = evaluate(eval_set, retrieval_fn, k=5)
            print("Retrieval metrics:", json.dumps(metrics, indent=2))
            continue

        # optional metadata filter example:
        # metadata_filter = {"source_file": "AISC_manual.pdf"}
        metadata_filter = None

        # 1. Parse and decompose query into search-friendly sub-queries
        sub_queries = rewrite_query(query)
        print(f"  [Parsed sub-queries]: {sub_queries}")

        # 2. Hybrid search per sub-query, fuse across all of them with RRF
        all_fused = [
            hybrid_search(sq, bm25, chunks, collection, top_k=10, metadata_filter=metadata_filter)
            for sq in sub_queries
        ]
        fused_ids = reciprocal_rank_fusion(all_fused)[:15]
        candidates = get_chunks_by_ids(fused_ids, chunks)

        # 3. Rerank each candidate against EACH sub-query, keep best score per chunk
        #    (see rerank_multi docstring for why this beats reranking against
        #    a single joined/concatenated query string)
        reranked = rerank_multi(sub_queries, candidates, top_k=5)

        # 4. Generate answer, grounded in the reranked chunks
        answer = generate_answer(query, reranked)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for c in reranked:
            print(f"  Page {c['page_num']} ({c['source_file']})")