"""
IS 456 RAG -- Query Pipeline (v2)
=================================
Hybrid retrieval (BM25 + semantic, RRF) -> explicit-reference boost -> rerank -> generate.

Older versions are kept on the HPC as GENERATION.py (pre-v2) / GENERATION_before_*.py.

Each retrieval change can be switched off with an environment variable, so it
can be A/B tested without editing code:
    RAG_TOKENIZER=split        old BM25 tokenizer (lower().split())
    RAG_REF_BOOST=0            disable the Table/Fig/Clause reference boost
    RAG_RERANK_MAX_LENGTH=128  shorter reranker truncation
    RAG_BM25_TOP_K / RAG_SEMANTIC_TOP_K / RAG_RERANK_TOP_N / RAG_NUM_CTX

Usage:
    python GENERATION.py                 interactive CLI loop ('exit' to quit)
    python GENERATION.py --probe "q"     retrieval only, no LLM: prints the chunks
"""

import os
import re
import sys
import time
from dataclasses import dataclass

import ollama
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "is_codes"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "gemma4:26b")
FLASHRANK_CACHE = os.environ.get("FLASHRANK_CACHE", "/home/civil/btech/ce1231034/flashrank_cache")

BM25_TOP_K = _env_int("RAG_BM25_TOP_K", 30)          # was 15
SEMANTIC_TOP_K = _env_int("RAG_SEMANTIC_TOP_K", 30)  # was 15
RRF_K = 60
RERANK_TOP_N = _env_int("RAG_RERANK_TOP_N", 6)       # was 5
RERANK_MAX_LENGTH = _env_int("RAG_RERANK_MAX_LENGTH", 512)
MIN_RERANK_SCORE = float(os.environ.get("RAG_MIN_RERANK_SCORE", 0.15))
TOKENIZER = os.environ.get("RAG_TOKENIZER", "regex")  # "regex" (new) or "split" (old)

# Explicit references in the question ("Table 26", "Fig. 4", "Clause 41.3.1",
# "Annex E") pull the matching chunks into the candidate pool and give them a
# score bonus at rerank time. The cross-encoder alone often ranks a table body
# below prose that merely *mentions* the table.
REF_BOOST = os.environ.get("RAG_REF_BOOST", "1") == "1"
REF_BONUS = 0.35
MAX_REF_CHUNKS_PER_REF = 3

AMENDMENT_PENALTY = 0.5
AMENDMENT_HEADER_RE = re.compile(r'^\W*(IS\s*456[^\n]{0,120}AMENDMENT|AMENDMENT|Amendment)', re.I)

# Ollama's default context window is small (2k-4k tokens depending on version).
# 6 chunks + the system prompt can exceed it, and Ollama then silently drops
# the START of the prompt -- i.e. the system prompt and the first excerpts.
NUM_CTX = _env_int("RAG_NUM_CTX", 16384)
NUM_PREDICT = _env_int("RAG_NUM_PREDICT", 2048)   # was 8192; real answers are < 1500 tokens

# Retry: a 500/EOF is usually the Ollama runner subprocess dying under memory
# pressure; the server respawns it, so a short pause + retry recovers.
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_DELAY_SECONDS = [5, 20, 60]

# Shown to the user instead of internal file names (the DB stores "sample.pdf").
SOURCE_DISPLAY = {"sample.pdf": "IS 456:2000", "IS456.pdf": "IS 456:2000"}

REFUSAL = "I could not find this in the provided IS code excerpts."


# ---------------------------------------------------------------------------
# OCR clean-up (applied at read time only; the database is not modified)
# ---------------------------------------------------------------------------
# The PDF text layer is OCR output with systematic confusions, e.g.
# "CDnstructiDn JDints" (D for o) and "4l.3.1" (l for 1). The rules are narrow
# on purpose so real symbols such as D (overall depth) in "l/D" or "bD" survive.
_OCR_FIXES = [
    (re.compile(r"(?<=[a-z])D(?=[a-z])"), "o"),         # inspectiDn -> inspection
    (re.compile(r"(?<=\b[A-Z])D(?=[a-z]{2})"), "o"),    # CDnstruction -> Construction
    (re.compile(r"(?<=\d)[lI](?=[\d.])"), "1"),         # 4l.3 -> 41.3
    (re.compile(r"(?<=[\d.])[lI](?=\d)"), "1"),         # 2.l5 -> 2.15
    (re.compile(r"(?<=\d\.)[lI](?![A-Za-z])"), "1"),    # 41.3.l -> 41.3.1
    (re.compile(r"(?<=\d)O(?=\d)"), "0"),               # 1O0 -> 100
]


def clean_ocr(text: str) -> str:
    for pattern, repl in _OCR_FIXES:
        text = pattern.sub(repl, text)
    return text


# Keeps clause numbers (26.5.1.3) and decimals (0.0035) whole, drops the
# punctuation that the old .split() left attached ("16," "fig." "(see"),
# and splits "Fe415"/"M20" into letters + number so "Fe 415" == "Fe415".
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)+|\d+|[^\W\d_]+")


def tokenize(text: str) -> list[str]:
    if TOKENIZER == "split":
        return text.lower().split()
    return _TOKEN_RE.findall(clean_ocr(text).lower())


# ---------------------------------------------------------------------------
# Corpus loading + BM25 index
# ---------------------------------------------------------------------------
def load_corpus(collection):
    """Pulls everything out of Chroma once so BM25 has a fixed corpus to index."""
    raw = collection.get(include=["documents", "metadatas"])
    return {
        "ids": raw["ids"],
        "documents": raw["documents"],
        "metadatas": [m or {} for m in raw["metadatas"]],
    }


def build_bm25_index(corpus):
    return BM25Okapi([tokenize(doc) for doc in corpus["documents"]])


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------
@dataclass
class Filters:
    source: str | None = None          # exact match, e.g. "sample.pdf"
    content_type: str | None = None    # "text" or "table"
    clause_prefix: str | None = None   # e.g. "6.2" matches 6.2, 6.2.1, ...


def matches_filters(metadata: dict, filters: Filters) -> bool:
    if filters.source and metadata.get("source") != filters.source:
        return False
    if filters.content_type and metadata.get("content_type") != filters.content_type:
        return False
    if filters.clause_prefix:
        if not str(metadata.get("clause", "")).startswith(filters.clause_prefix):
            return False
    return True


def chroma_where_clause(filters: Filters) -> dict | None:
    conditions = []
    if filters.source:
        conditions.append({"source": filters.source})
    if filters.content_type:
        conditions.append({"content_type": filters.content_type})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Explicit references: "Table 16", "Figs 13 and 14", "Clause 41.3.1", "Annex E"
# ---------------------------------------------------------------------------
_NUM = r"[0-9]+[A-Za-z]?(?:\.[0-9]+)*"
_REF_RE = re.compile(
    rf"\b(tables?|figs?\.?|figures?|clauses?|cl\.|annex)\s*"
    rf"((?:{_NUM}|[A-H]\b)(?:\s*(?:,|and|&|to|–|-)\s*{_NUM})*)",
    re.I,
)


def extract_refs(text: str) -> list[tuple[str, str]]:
    refs = []
    for m in _REF_RE.finditer(text):
        word = m.group(1).lower()
        kind = ("table" if word.startswith("tab") else
                "fig" if word.startswith("fig") else
                "annex" if word.startswith("annex") else "clause")
        for num in re.findall(rf"{_NUM}|\b[A-H]\b", m.group(2)):
            if (kind, num) not in refs:
                refs.append((kind, num))
    return refs


def ref_match_strength(kind: str, num: str, text: str, metadata: dict) -> int:
    """2 = the chunk IS that table/figure/clause (label in its heading or metadata),
    1 = the chunk merely mentions it, 0 = no match."""
    n = re.escape(num)
    tail = r"(?![0-9A-Za-z])"          # Table 16 != Table 16A, Fig 2 != Fig 21
    clause_meta = str(metadata.get("clause", ""))
    text = clean_ocr(text)
    head = text[:120]
    if kind == "clause":
        if clause_meta == num or clause_meta.startswith(num + "."):
            return 2
        if re.match(rf"\W*{n}(?![0-9]|\.[0-9])", head):
            return 2
        return 0   # bare clause numbers in running text are mostly cross-references
    if kind == "annex" and clause_meta.upper().startswith(f"{num.upper()}-"):
        return 2
    word = {"table": r"table", "fig": r"fig(?:ure)?\.?", "annex": r"annex"}[kind]
    pat = re.compile(rf"\b{word}\s*{n}{tail}", re.I)
    # INGEST.py stores a table note's parent in metadata["related_table"],
    # e.g. "the table on sample.pdf page 47 (Table 16 Nominal Cover ...)".
    if kind == "table" and pat.search(str(metadata.get("related_table", ""))):
        return 2
    # A caption/heading starts its own line ("TABLE 16 NOMINAL COVER ...",
    # "FIG. 4 MODIFICATION FACTOR ..."); a mention ("as given in Table 16") doesn't.
    if re.search(rf"(?m)^\W*{word}\s*{n}{tail}", text, re.I):
        return 2
    if kind == "table" and metadata.get("content_type") == "table" and pat.search(text):
        return 2
    return 1 if pat.search(text) else 0


def find_reference_chunks(query: str, corpus) -> dict[str, int]:
    """Returns {doc_id: strength} for chunks the question names explicitly."""
    found: dict[str, int] = {}
    for kind, num in extract_refs(query):
        hits = []
        for doc_id, doc, meta in zip(corpus["ids"], corpus["documents"], corpus["metadatas"]):
            s = ref_match_strength(kind, num, doc, meta)
            if s:
                hits.append((s, doc_id))
        strong = sorted(d for s, d in hits if s == 2)
        chosen = ([(2, d) for d in strong[:MAX_REF_CHUNKS_PER_REF]] if strong
                  else sorted(hits, key=lambda h: h[1])[:2])
        for s, doc_id in chosen:
            found[doc_id] = max(found.get(doc_id, 0), s)
    return found


# ---------------------------------------------------------------------------
# Hybrid retrieval — BM25 + semantic, fused with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def hybrid_retrieve(query: str, corpus, bm25_index, collection, filters: Filters):
    bm25_scores = bm25_index.get_scores(tokenize(query))
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
    bm25_ranks = {corpus["ids"][i]: rank for rank, i in enumerate(bm25_ranked_idx)}

    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    semantic_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEMANTIC_TOP_K,
        where=chroma_where_clause(filters),
    )
    semantic_ranks = {doc_id: rank for rank, doc_id in enumerate(semantic_result["ids"][0])}

    fused_scores = {}
    for doc_id in set(bm25_ranks) | set(semantic_ranks):
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[doc_id])
        if doc_id in semantic_ranks:
            score += 1.0 / (RRF_K + semantic_ranks[doc_id])
        fused_scores[doc_id] = score
    # Sort by (-score, doc_id): deterministic tie-break across runs (string
    # hashing is randomised per process, so set order is not stable).
    fused_ranked = [d for d, _ in sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))]

    ref_hits = find_reference_chunks(query, corpus) if REF_BOOST else {}
    ordered = [d for d in ref_hits if d not in fused_scores] + fused_ranked

    id_to_idx = {doc_id: i for i, doc_id in enumerate(corpus["ids"])}
    candidates = []
    for doc_id in ordered:
        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue
        metadata = corpus["metadatas"][idx]
        if not matches_filters(metadata, filters):
            continue
        candidates.append({
            "id": doc_id,
            "text": corpus["documents"][idx],
            "metadata": metadata,
            "ref_match": ref_hits.get(doc_id, 0),
        })
    return candidates


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
# max_length is set explicitly: the cross-encoder only "sees" this many tokens
# of each chunk, so with a short limit the bottom of a table or its NOTES are
# invisible to the reranker.
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=FLASHRANK_CACHE,
                 max_length=RERANK_MAX_LENGTH)


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    passages = [{"id": c["id"], "text": clean_ocr(c["text"])} for c in candidates]
    results = _ranker.rerank(RerankRequest(query=query, passages=passages))

    by_id = {c["id"]: c for c in candidates}
    is_amendment_query = "amendment" in query.lower()
    scored = []
    for r in results:
        chunk = by_id[r["id"]]
        score = float(r["score"])
        if not is_amendment_query and AMENDMENT_HEADER_RE.match(chunk["text"]):
            score -= AMENDMENT_PENALTY
        if chunk.get("ref_match") == 2:
            score += REF_BONUS
        elif chunk.get("ref_match") == 1:
            score += REF_BONUS / 3
        chunk["rerank_score"] = score
        scored.append(chunk)
    scored.sort(key=lambda c: (-c["rerank_score"], c["id"]))
    return scored[:RERANK_TOP_N]


# ---------------------------------------------------------------------------
# Generation — grounded, citation-forced, refuses when evidence is weak
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an assistant answering questions about IS 456:2000 (Indian Standard \
for plain and reinforced concrete), using ONLY the excerpts provided below.

Rules:
- Answer strictly from the provided excerpts. Do not use outside knowledge.
- Every claim must cite its source: clause number (e.g. "as per Cl. 6.2.1") or table/figure reference.
- If the excerpts do not contain enough information to answer, say exactly: \
"{REFUSAL}" Do not guess or extrapolate.
- If the question is a yes/no question, begin the answer with "Yes" or "No".
- If a rule has a condition (for example "bars up to 12 mm diameter", "M35 and above", \
"web depth exceeding 750 mm"), check whether the situation in the question meets that \
condition and say explicitly whether the rule applies.
- If an excerpt from an Amendment modifies a clause, the amended text governs. Say that \
the amendment applies and quote the amended rule.
- Only state a formula if it appears in the excerpts. Never reconstruct, derive or recall \
a formula from memory.
- Some excerpts are marked [LOW-CONFIDENCE OCR]: the text may be corrupted (broken symbols, \
garbled tokens, missing operators). If the answer depends on such an excerpt, say that the \
source text appears corrupted and do NOT "clean up" or guess a formula.
- A table's NOTES (numbered notes below a table) modify that table's values; read them together.
- If the excerpts give a formula, multiplier or adjustment AND the question gives the numbers, \
perform the calculation and state the final value, showing the arithmetic.
- Keep answers precise and technical; this is for engineering use.
"""


def source_label(meta: dict) -> str:
    src = SOURCE_DISPLAY.get(meta.get("source", ""), meta.get("source", "IS 456"))
    flag = " [LOW-CONFIDENCE OCR]" if meta.get("low_confidence") else ""
    return (f"[Source: {src}, page {meta.get('page', '?')}, clause {meta.get('clause', '?')}, "
            f"type {meta.get('content_type', '?')}{flag}]")


def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(f"{source_label(c['metadata'])}\n{clean_ocr(c['text'])}" for c in chunks)


def _call_ollama_chat_with_retry(prompt: str) -> dict:
    last_error = None
    for attempt in range(1, GENERATION_MAX_RETRIES + 1):
        try:
            return ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0, "seed": 42,
                         "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
                think=False,
            )
        except Exception as e:  # ollama.ResponseError, ConnectionError, etc.
            last_error = e
            if attempt < GENERATION_MAX_RETRIES:
                delay = GENERATION_RETRY_DELAY_SECONDS[attempt - 1]
                print(f"  [warn] ollama.chat failed (attempt {attempt}/{GENERATION_MAX_RETRIES}): {e} "
                      f"-- retrying in {delay}s")
                time.sleep(delay)
    raise last_error


def generate_answer(query: str, reranked_chunks: list[dict]) -> str:
    """Never raises: an infrastructure failure comes back as a visible error string."""
    if not reranked_chunks:
        return REFUSAL
    top_score = reranked_chunks[0].get("rerank_score", 0)
    if top_score < MIN_RERANK_SCORE:
        return f"{REFUSAL[:-1]} (best match confidence too low: {top_score:.2f})."

    prompt = f"Excerpts:\n\n{build_context(reranked_chunks)}\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = _call_ollama_chat_with_retry(prompt)
    except Exception as e:
        return f"[GENERATION ERROR after {GENERATION_MAX_RETRIES} attempts: {e}]"
    return response["message"]["content"]


def describe_chunk(c: dict) -> str:
    m = c["metadata"]
    return (f"p{m.get('page', '?')} cl.{m.get('clause', '?')} ({m.get('content_type', '?')}) "
            f"score={c.get('rerank_score', 0):.2f}{' REF' if c.get('ref_match') else ''} id={c['id']}")


# ---------------------------------------------------------------------------
# Pipeline object: load once, ask many times (used by the chatbot front end)
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, chroma_dir: str = CHROMA_DIR):
        client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = client.get_or_create_collection(COLLECTION_NAME)
        self.corpus = load_corpus(self.collection)
        if not self.corpus["ids"]:
            raise RuntimeError("Collection is empty -- run the ingestion script first.")
        self.bm25 = build_bm25_index(self.corpus)

    def retrieve(self, query: str, filters: Filters | None = None) -> list[dict]:
        candidates = hybrid_retrieve(query, self.corpus, self.bm25, self.collection, filters or Filters())
        return rerank(query, candidates)

    def ask(self, query: str, filters: Filters | None = None) -> dict:
        chunks = self.retrieve(query, filters)
        return {"answer": generate_answer(query, chunks), "chunks": chunks}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    pipe = Pipeline()
    print(f"Ready. {len(pipe.corpus['ids'])} chunks indexed.\n")

    # Retrieval-only probe: python GENERATION.py --probe "question" ["question" ...]
    if len(sys.argv) > 2 and sys.argv[1] == "--probe":
        for q in sys.argv[2:]:
            print(f"=== {q}\n    refs detected: {extract_refs(q)}")
            for c in pipe.retrieve(q):
                print("  " + describe_chunk(c))
                print("      " + clean_ocr(c["text"])[:160].replace("\n", " "))
            print()
        return

    while True:
        query = input("Ask a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        out = pipe.ask(query)
        print("\n--- Retrieved (post-rerank) ---")
        for c in out["chunks"]:
            print("  " + describe_chunk(c))
        print(f"\n--- Answer ---\n{out['answer']}\n")


if __name__ == "__main__":
    main()
