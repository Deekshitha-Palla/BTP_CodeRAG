# # """
# # IS-Code RAG — Query Pipeline (v1 / MVP)
# # ==========================================
# # Hybrid retrieval (BM25 + semantic) -> metadata filtering -> rerank -> generate.

# # Install (in addition to ingest.py's requirements):
# #     pip install rank_bm25 flashrank

# # Usage:
# #     python query_pipeline.py
# #     (interactive CLI loop; type 'exit' to quit)
# # """

# # import re
# # from dataclasses import dataclass

# # import ollama
# # import chromadb
# # from rank_bm25 import BM25Okapi
# # from flashrank import Ranker, RerankRequest

# # # ---------------------------------------------------------------------------
# # # Config — keep in sync with ingest.py
# # # ---------------------------------------------------------------------------
# # CHROMA_DIR = "./chroma_db"
# # COLLECTION_NAME = "is_codes"
# # EMBED_MODEL = "nomic-embed-text"
# # LLM_MODEL = "gemma4:26b"

# # BM25_TOP_K = 15
# # SEMANTIC_TOP_K = 15
# # RRF_K = 60                 # standard reciprocal-rank-fusion constant
# # RERANK_TOP_N = 5           # how many chunks actually go to the LLM
# # MIN_RERANK_SCORE = 0.15    # below this, we refuse rather than guess — tune this against your own eval set, don't trust the default


# # # ---------------------------------------------------------------------------
# # # Corpus loading + BM25 index
# # # ---------------------------------------------------------------------------
# # def load_corpus(collection):
# #     """Pulls everything out of Chroma once so BM25 has a fixed corpus to index."""
# #     raw = collection.get(include=["documents", "metadatas"])
# #     return {
# #         "ids": raw["ids"],
# #         "documents": raw["documents"],
# #         "metadatas": raw["metadatas"],
# #     }


# # def build_bm25_index(corpus):
# #     tokenized = [doc.lower().split() for doc in corpus["documents"]]
# #     return BM25Okapi(tokenized)


# # # ---------------------------------------------------------------------------
# # # Metadata filtering
# # # ---------------------------------------------------------------------------
# # @dataclass
# # class Filters:
# #     source: str | None = None          # exact match, e.g. "IS456.pdf"
# #     content_type: str | None = None    # "text" or "table"
# #     clause_prefix: str | None = None   # e.g. "6.2" matches 6.2, 6.2.1, 6.2.1.3...


# # def matches_filters(metadata: dict, filters: Filters) -> bool:
# #     if filters.source and metadata.get("source") != filters.source:
# #         return False
# #     if filters.content_type and metadata.get("content_type") != filters.content_type:
# #         return False
# #     if filters.clause_prefix:
# #         clause = metadata.get("clause", "")
# #         if not clause.startswith(filters.clause_prefix):
# #             return False
# #     return True


# # def chroma_where_clause(filters: Filters) -> dict | None:
# #     """Chroma's `where` handles exact-match fields natively; clause_prefix
# #     is a prefix match, which Chroma doesn't support directly, so that one
# #     is applied as a post-filter instead (see matches_filters)."""
# #     conditions = []
# #     if filters.source:
# #         conditions.append({"source": filters.source})
# #     if filters.content_type:
# #         conditions.append({"content_type": filters.content_type})
# #     if not conditions:
# #         return None
# #     if len(conditions) == 1:
# #         return conditions[0]
# #     return {"$and": conditions}


# # # ---------------------------------------------------------------------------
# # # Hybrid retrieval — BM25 + semantic, fused with Reciprocal Rank Fusion
# # # ---------------------------------------------------------------------------
# # def hybrid_retrieve(query: str, corpus, bm25_index, collection, filters: Filters):
# #     # --- BM25 side ---
# #     bm25_scores = bm25_index.get_scores(query.lower().split())
# #     bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
# #     bm25_ranks = {corpus["ids"][i]: rank for rank, i in enumerate(bm25_ranked_idx)}

# #     # --- Semantic side ---
# #     query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
# #     semantic_result = collection.query(
# #         query_embeddings=[query_embedding],
# #         n_results=SEMANTIC_TOP_K,
# #         where=chroma_where_clause(filters),
# #     )
# #     semantic_ids = semantic_result["ids"][0]
# #     semantic_ranks = {doc_id: rank for rank, doc_id in enumerate(semantic_ids)}

# #     # --- Reciprocal Rank Fusion ---
# #     all_ids = set(bm25_ranks) | set(semantic_ranks)
# #     fused_scores = {}
# #     for doc_id in all_ids:
# #         score = 0.0
# #         if doc_id in bm25_ranks:
# #             score += 1.0 / (RRF_K + bm25_ranks[doc_id])
# #         if doc_id in semantic_ranks:
# #             score += 1.0 / (RRF_K + semantic_ranks[doc_id])
# #         fused_scores[doc_id] = score

# #     fused_ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

# #     # --- Build candidate list, applying clause_prefix post-filter here ---
# #     id_to_idx = {doc_id: i for i, doc_id in enumerate(corpus["ids"])}
# #     candidates = []
# #     for doc_id, _score in fused_ranked:
# #         idx = id_to_idx.get(doc_id)
# #         if idx is None:
# #             continue
# #         metadata = corpus["metadatas"][idx]
# #         if not matches_filters(metadata, filters):
# #             continue
# #         candidates.append({
# #             "id": doc_id,
# #             "text": corpus["documents"][idx],
# #             "metadata": metadata,
# #         })

# #     return candidates


# # # ---------------------------------------------------------------------------
# # # Reranking
# # # ---------------------------------------------------------------------------
# # _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # small, CPU-friendly


# # def rerank(query: str, candidates: list[dict]) -> list[dict]:
# #     if not candidates:
# #         return []
# #     passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
# #     request = RerankRequest(query=query, passages=passages)
# #     results = _ranker.rerank(request)  # returns list sorted by score desc

# #     by_id = {c["id"]: c for c in candidates}
# #     reranked = []
# #     for r in results[:RERANK_TOP_N]:
# #         chunk = by_id[r["id"]]
# #         chunk["rerank_score"] = r["score"]
# #         reranked.append(chunk)
# #     return reranked


# # # ---------------------------------------------------------------------------
# # # Generation — grounded, citation-forced, refuses when evidence is weak
# # # ---------------------------------------------------------------------------
# # SYSTEM_PROMPT = """You are an assistant answering questions about Indian Standard (IS) codes \
# # for civil engineering, using ONLY the excerpts provided below.

# # Rules:
# # - Answer strictly from the provided excerpts. Do not use outside knowledge.
# # - Every claim must cite its source: clause number (e.g. "as per Cl. 6.2.1") or \
# # table/page reference.
# # - If the excerpts do not contain enough information to answer, say exactly: \
# # "I could not find this in the provided IS code excerpts." Do not guess or extrapolate.
# # - Keep answers precise and technical; this is for engineering use, not casual conversation.
# # """


# # def build_context(chunks: list[dict]) -> str:
# #     parts = []
# #     for c in chunks:
# #         meta = c["metadata"]
# #         label = f"[Source: {meta['source']}, page {meta['page']}, clause {meta['clause']}, type {meta['content_type']}]"
# #         parts.append(f"{label}\n{c['text']}")
# #     return "\n\n---\n\n".join(parts)


# # def generate_answer(query: str, reranked_chunks: list[dict]) -> str:
# #     if not reranked_chunks:
# #         return "I could not find this in the provided IS code excerpts."

# #     top_score = reranked_chunks[0].get("rerank_score", 0)
# #     if top_score < MIN_RERANK_SCORE:
# #         return (
# #             "I could not find this in the provided IS code excerpts "
# #             f"(best match confidence too low: {top_score:.2f})."
# #         )

# #     context = build_context(reranked_chunks)
# #     prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

# #     response = ollama.chat(
# #         model=LLM_MODEL,
# #         messages=[
# #             {"role": "system", "content": SYSTEM_PROMPT},
# #             {"role": "user", "content": prompt},
# #         ],
# #         options={"temperature": 0.1},
# #     )
# #     return response["message"]["content"]


# # # ---------------------------------------------------------------------------
# # # Interactive CLI
# # # ---------------------------------------------------------------------------
# # def main():
# #     client = chromadb.PersistentClient(path=CHROMA_DIR)
# #     collection = client.get_or_create_collection(COLLECTION_NAME)

# #     print("Loading corpus + building BM25 index ...")
# #     corpus = load_corpus(collection)
# #     if not corpus["ids"]:
# #         print("Collection is empty — run ingest.py first.")
# #         return
# #     bm25_index = build_bm25_index(corpus)
# #     print(f"Ready. {len(corpus['ids'])} chunks indexed.\n")

# #     while True:
# #         query = input("Ask a question (or 'exit'): ").strip()
# #         if query.lower() == "exit":
# #             break
# #         if not query:
# #             continue

# #         # Example: pass Filters(content_type="table") to restrict to tables,
# #         # or Filters(clause_prefix="6.2") to scope to a clause family.
# #         filters = Filters()

# #         candidates = hybrid_retrieve(query, corpus, bm25_index, collection, filters)
# #         reranked = rerank(query, candidates)

# #         print("\n--- Retrieved (post-rerank) ---")
# #         for c in reranked:
# #             meta = c["metadata"]
# #             print(f"  [{c['rerank_score']:.3f}] {meta['source']} p{meta['page']} cl.{meta['clause']} ({meta['content_type']})")

# #         answer = generate_answer(query, reranked)
# #         print(f"\n--- Answer ---\n{answer}\n")


# # if __name__ == "__main__":
# #     main()
# """
# IS-Code RAG — Query Pipeline (v1 / MVP)
# ==========================================
# Hybrid retrieval (BM25 + semantic) -> metadata filtering -> rerank -> generate.

# Install (in addition to ingest.py's requirements):
#     pip install rank_bm25 flashrank

# Usage:
#     python query_pipeline.py
#     (interactive CLI loop; type 'exit' to quit)
# """

# import re
# from dataclasses import dataclass

# import ollama
# import chromadb
# from rank_bm25 import BM25Okapi
# from flashrank import Ranker, RerankRequest

# # ---------------------------------------------------------------------------
# # Config — keep in sync with ingest.py
# # ---------------------------------------------------------------------------
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"
# LLM_MODEL = "gemma4:26b"

# BM25_TOP_K = 15
# SEMANTIC_TOP_K = 15
# RRF_K = 60                 # standard reciprocal-rank-fusion constant
# RERANK_TOP_N = 5           # how many chunks actually go to the LLM
# MIN_RERANK_SCORE = 0.15    # below this, we refuse rather than guess — tune this against your own eval set, don't trust the default


# # ---------------------------------------------------------------------------
# # Corpus loading + BM25 index
# # ---------------------------------------------------------------------------
# def load_corpus(collection):
#     """Pulls everything out of Chroma once so BM25 has a fixed corpus to index."""
#     raw = collection.get(include=["documents", "metadatas"])
#     return {
#         "ids": raw["ids"],
#         "documents": raw["documents"],
#         "metadatas": raw["metadatas"],
#     }


# def build_bm25_index(corpus):
#     tokenized = [doc.lower().split() for doc in corpus["documents"]]
#     return BM25Okapi(tokenized)


# # ---------------------------------------------------------------------------
# # Metadata filtering
# # ---------------------------------------------------------------------------
# @dataclass
# class Filters:
#     source: str | None = None          # exact match, e.g. "IS456.pdf"
#     content_type: str | None = None    # "text" or "table"
#     clause_prefix: str | None = None   # e.g. "6.2" matches 6.2, 6.2.1, 6.2.1.3...


# def matches_filters(metadata: dict, filters: Filters) -> bool:
#     if filters.source and metadata.get("source") != filters.source:
#         return False
#     if filters.content_type and metadata.get("content_type") != filters.content_type:
#         return False
#     if filters.clause_prefix:
#         clause = metadata.get("clause", "")
#         if not clause.startswith(filters.clause_prefix):
#             return False
#     return True


# def chroma_where_clause(filters: Filters) -> dict | None:
#     """Chroma's `where` handles exact-match fields natively; clause_prefix
#     is a prefix match, which Chroma doesn't support directly, so that one
#     is applied as a post-filter instead (see matches_filters)."""
#     conditions = []
#     if filters.source:
#         conditions.append({"source": filters.source})
#     if filters.content_type:
#         conditions.append({"content_type": filters.content_type})
#     if not conditions:
#         return None
#     if len(conditions) == 1:
#         return conditions[0]
#     return {"$and": conditions}


# # ---------------------------------------------------------------------------
# # Hybrid retrieval — BM25 + semantic, fused with Reciprocal Rank Fusion
# # ---------------------------------------------------------------------------
# def hybrid_retrieve(query: str, corpus, bm25_index, collection, filters: Filters):
#     # --- BM25 side ---
#     bm25_scores = bm25_index.get_scores(query.lower().split())
#     bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
#     bm25_ranks = {corpus["ids"][i]: rank for rank, i in enumerate(bm25_ranked_idx)}

#     # --- Semantic side ---
#     query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
#     semantic_result = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=SEMANTIC_TOP_K,
#         where=chroma_where_clause(filters),
#     )
#     semantic_ids = semantic_result["ids"][0]
#     semantic_ranks = {doc_id: rank for rank, doc_id in enumerate(semantic_ids)}

#     # --- Reciprocal Rank Fusion ---
#     all_ids = set(bm25_ranks) | set(semantic_ranks)
#     fused_scores = {}
#     for doc_id in all_ids:
#         score = 0.0
#         if doc_id in bm25_ranks:
#             score += 1.0 / (RRF_K + bm25_ranks[doc_id])
#         if doc_id in semantic_ranks:
#             score += 1.0 / (RRF_K + semantic_ranks[doc_id])
#         fused_scores[doc_id] = score

#     fused_ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

#     # --- Build candidate list, applying clause_prefix post-filter here ---
#     id_to_idx = {doc_id: i for i, doc_id in enumerate(corpus["ids"])}
#     candidates = []
#     for doc_id, _score in fused_ranked:
#         idx = id_to_idx.get(doc_id)
#         if idx is None:
#             continue
#         metadata = corpus["metadatas"][idx]
#         if not matches_filters(metadata, filters):
#             continue
#         candidates.append({
#             "id": doc_id,
#             "text": corpus["documents"][idx],
#             "metadata": metadata,
#         })

#     return candidates


# # ---------------------------------------------------------------------------
# # Reranking
# # ---------------------------------------------------------------------------
# _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # small, CPU-friendly


# def rerank(query: str, candidates: list[dict]) -> list[dict]:
#     if not candidates:
#         return []
#     passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
#     request = RerankRequest(query=query, passages=passages)
#     results = _ranker.rerank(request)  # returns list sorted by score desc

#     by_id = {c["id"]: c for c in candidates}
#     reranked = []
#     for r in results[:RERANK_TOP_N]:
#         chunk = by_id[r["id"]]
#         chunk["rerank_score"] = r["score"]
#         reranked.append(chunk)
#     return reranked


# # ---------------------------------------------------------------------------
# # Generation — grounded, citation-forced, refuses when evidence is weak
# # ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """You are an assistant answering questions about Indian Standard (IS) codes \
# for civil engineering, using ONLY the excerpts provided below.

# Rules:
# - Answer strictly from the provided excerpts. Do not use outside knowledge.
# - Every claim must cite its source: clause number (e.g. "as per Cl. 6.2.1") or \
# table/page reference.
# - If the excerpts do not contain enough information to answer, say exactly: \
# "I could not find this in the provided IS code excerpts." Do not guess or extrapolate.
# - Some excerpts are marked [LOW-CONFIDENCE OCR]. This means the text extraction may be \
# corrupted, especially for formulas/equations (broken symbols, garbled tokens, missing \
# operators). If the information needed to answer comes from a low-confidence excerpt, \
# say explicitly that the source text appears corrupted/incomplete and do NOT invent, \
# guess, or "clean up" a formula to fill the gap — a wrong formula is worse than admitting \
# the source couldn't be read cleanly.
# - A table's "NOTES" (numbered notes below a table) are conditions that modify that \
# table's values — read them together, not independently.
# - Keep answers precise and technical; this is for engineering use, not casual conversation.
# """


# def build_context(chunks: list[dict]) -> str:
#     parts = []
#     for c in chunks:
#         meta = c["metadata"]
#         flag = " [LOW-CONFIDENCE OCR]" if meta.get("low_confidence") else ""
#         label = f"[Source: {meta['source']}, page {meta['page']}, clause {meta['clause']}, type {meta['content_type']}{flag}]"
#         parts.append(f"{label}\n{c['text']}")
#     return "\n\n---\n\n".join(parts)


# def generate_answer(query: str, reranked_chunks: list[dict]) -> str:
#     if not reranked_chunks:
#         return "I could not find this in the provided IS code excerpts."

#     top_score = reranked_chunks[0].get("rerank_score", 0)
#     if top_score < MIN_RERANK_SCORE:
#         return (
#             "I could not find this in the provided IS code excerpts "
#             f"(best match confidence too low: {top_score:.2f})."
#         )

#     context = build_context(reranked_chunks)
#     prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

#     response = ollama.chat(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt},
#         ],
#         options={"temperature": 0.1},
#     )
#     return response["message"]["content"]


# # ---------------------------------------------------------------------------
# # Interactive CLI
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     print("Loading corpus + building BM25 index ...")
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty — run ingest.py first.")
#         return
#     bm25_index = build_bm25_index(corpus)
#     print(f"Ready. {len(corpus['ids'])} chunks indexed.\n")

#     while True:
#         query = input("Ask a question (or 'exit'): ").strip()
#         if query.lower() == "exit":
#             break
#         if not query:
#             continue

#         # Example: pass Filters(content_type="table") to restrict to tables,
#         # or Filters(clause_prefix="6.2") to scope to a clause family.
#         filters = Filters()

#         candidates = hybrid_retrieve(query, corpus, bm25_index, collection, filters)
#         reranked = rerank(query, candidates)

#         print("\n--- Retrieved (post-rerank) ---")
#         for c in reranked:
#             meta = c["metadata"]
#             print(f"  [{c['rerank_score']:.3f}] {meta['source']} p{meta['page']} cl.{meta['clause']} ({meta['content_type']})")

#         answer = generate_answer(query, reranked)
#         print(f"\n--- Answer ---\n{answer}\n")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Query Pipeline (v1 / MVP)
# ==========================================
# Hybrid retrieval (BM25 + semantic) -> metadata filtering -> rerank -> generate.

# Install (in addition to ingest.py's requirements):
#     pip install rank_bm25 flashrank

# Usage:
#     python query_pipeline.py
#     (interactive CLI loop; type 'exit' to quit)
# """

# import re
# from dataclasses import dataclass

# import ollama
# import chromadb
# from rank_bm25 import BM25Okapi
# from flashrank import Ranker, RerankRequest

# # ---------------------------------------------------------------------------
# # Config — keep in sync with ingest.py
# # ---------------------------------------------------------------------------
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"
# LLM_MODEL = "gemma4:26b"

# BM25_TOP_K = 15
# SEMANTIC_TOP_K = 15
# RRF_K = 60                 # standard reciprocal-rank-fusion constant
# RERANK_TOP_N = 5           # how many chunks actually go to the LLM
# MIN_RERANK_SCORE = 0.15    # below this, we refuse rather than guess — tune this against your own eval set, don't trust the default


# # ---------------------------------------------------------------------------
# # Corpus loading + BM25 index
# # ---------------------------------------------------------------------------
# def load_corpus(collection):
#     """Pulls everything out of Chroma once so BM25 has a fixed corpus to index."""
#     raw = collection.get(include=["documents", "metadatas"])
#     return {
#         "ids": raw["ids"],
#         "documents": raw["documents"],
#         "metadatas": raw["metadatas"],
#     }


# def build_bm25_index(corpus):
#     tokenized = [doc.lower().split() for doc in corpus["documents"]]
#     return BM25Okapi(tokenized)


# # ---------------------------------------------------------------------------
# # Metadata filtering
# # ---------------------------------------------------------------------------
# @dataclass
# class Filters:
#     source: str | None = None          # exact match, e.g. "IS456.pdf"
#     content_type: str | None = None    # "text" or "table"
#     clause_prefix: str | None = None   # e.g. "6.2" matches 6.2, 6.2.1, 6.2.1.3...


# def matches_filters(metadata: dict, filters: Filters) -> bool:
#     if filters.source and metadata.get("source") != filters.source:
#         return False
#     if filters.content_type and metadata.get("content_type") != filters.content_type:
#         return False
#     if filters.clause_prefix:
#         clause = metadata.get("clause", "")
#         if not clause.startswith(filters.clause_prefix):
#             return False
#     return True


# def chroma_where_clause(filters: Filters) -> dict | None:
#     """Chroma's `where` handles exact-match fields natively; clause_prefix
#     is a prefix match, which Chroma doesn't support directly, so that one
#     is applied as a post-filter instead (see matches_filters)."""
#     conditions = []
#     if filters.source:
#         conditions.append({"source": filters.source})
#     if filters.content_type:
#         conditions.append({"content_type": filters.content_type})
#     if not conditions:
#         return None
#     if len(conditions) == 1:
#         return conditions[0]
#     return {"$and": conditions}


# # ---------------------------------------------------------------------------
# # Hybrid retrieval — BM25 + semantic, fused with Reciprocal Rank Fusion
# # ---------------------------------------------------------------------------
# def hybrid_retrieve(query: str, corpus, bm25_index, collection, filters: Filters):
#     # --- BM25 side ---
#     bm25_scores = bm25_index.get_scores(query.lower().split())
#     bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
#     bm25_ranks = {corpus["ids"][i]: rank for rank, i in enumerate(bm25_ranked_idx)}

#     # --- Semantic side ---
#     query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
#     semantic_result = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=SEMANTIC_TOP_K,
#         where=chroma_where_clause(filters),
#     )
#     semantic_ids = semantic_result["ids"][0]
#     semantic_ranks = {doc_id: rank for rank, doc_id in enumerate(semantic_ids)}

#     # --- Reciprocal Rank Fusion ---
#     all_ids = set(bm25_ranks) | set(semantic_ranks)
#     fused_scores = {}
#     for doc_id in all_ids:
#         score = 0.0
#         if doc_id in bm25_ranks:
#             score += 1.0 / (RRF_K + bm25_ranks[doc_id])
#         if doc_id in semantic_ranks:
#             score += 1.0 / (RRF_K + semantic_ranks[doc_id])
#         fused_scores[doc_id] = score

#     # Deterministic tiebreak by doc_id: Python randomizes string-hash seeds
#     # per process, so plain `set`/`dict` iteration order for these chunk-ID
#     # strings differs across separate script runs. With no secondary sort
#     # key, tied/near-tied RRF scores (common among a table's sibling chunks)
#     # would resolve to a DIFFERENT winner each run -- which is almost
#     # certainly why the same query returned 4 different answers across runs
#     # even after temperature was fixed to 0. Sorting by (-score, doc_id)
#     # makes the order reproducible regardless of hash randomization.
#     fused_ranked = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))

#     # --- Build candidate list, applying clause_prefix post-filter here ---
#     id_to_idx = {doc_id: i for i, doc_id in enumerate(corpus["ids"])}
#     candidates = []
#     for doc_id, _score in fused_ranked:
#         idx = id_to_idx.get(doc_id)
#         if idx is None:
#             continue
#         metadata = corpus["metadatas"][idx]
#         if not matches_filters(metadata, filters):
#             continue
#         candidates.append({
#             "id": doc_id,
#             "text": corpus["documents"][idx],
#             "metadata": metadata,
#         })

#     return candidates


# # ---------------------------------------------------------------------------
# # Reranking
# # ---------------------------------------------------------------------------
# _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # small, CPU-friendly


# def rerank(query: str, candidates: list[dict]) -> list[dict]:
#     if not candidates:
#         return []
#     passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
#     request = RerankRequest(query=query, passages=passages)
#     results = _ranker.rerank(request)  # returns list sorted by score desc

#     by_id = {c["id"]: c for c in candidates}
#     reranked = []
#     for r in results[:RERANK_TOP_N]:
#         chunk = by_id[r["id"]]
#         chunk["rerank_score"] = r["score"]
#         reranked.append(chunk)
#     return reranked


# # ---------------------------------------------------------------------------
# # Generation — grounded, citation-forced, refuses when evidence is weak
# # ---------------------------------------------------------------------------
# SYSTEM_PROMPT = """You are an assistant answering questions about Indian Standard (IS) codes \
# for civil engineering, using ONLY the excerpts provided below.

# Rules:
# - Answer strictly from the provided excerpts. Do not use outside knowledge.
# - Every claim must cite its source: clause number (e.g. "as per Cl. 6.2.1") or \
# table/page reference.
# - If the excerpts do not contain enough information to answer, say exactly: \
# "I could not find this in the provided IS code excerpts." Do not guess or extrapolate.
# - Some excerpts are marked [LOW-CONFIDENCE OCR]. This means the text extraction may be \
# corrupted, especially for formulas/equations (broken symbols, garbled tokens, missing \
# operators). If the information needed to answer comes from a low-confidence excerpt, \
# say explicitly that the source text appears corrupted/incomplete and do NOT invent, \
# guess, or "clean up" a formula to fill the gap — a wrong formula is worse than admitting \
# the source couldn't be read cleanly.
# - A table's "NOTES" (numbered notes below a table) are conditions that modify that \
# table's values — read them together, not independently.
# - Keep answers precise and technical; this is for engineering use, not casual conversation.
# - If the excerpts state a formula, multiplier, or adjustment (e.g. "multiply by 0.8", "reduce by 5 mm") AND the question gives you the specific numbers to apply it to, perform that calculation explicitly and state the final numeric result — do not just restate the formula/multiplier and stop there. Show the arithmetic step, not just the rule.
# """


# def build_context(chunks: list[dict]) -> str:
#     parts = []
#     for c in chunks:
#         meta = c["metadata"]
#         flag = " [LOW-CONFIDENCE OCR]" if meta.get("low_confidence") else ""
#         label = f"[Source: {meta['source']}, page {meta['page']}, clause {meta['clause']}, type {meta['content_type']}{flag}]"
#         parts.append(f"{label}\n{c['text']}")
#     return "\n\n---\n\n".join(parts)


# def generate_answer(query: str, reranked_chunks: list[dict]) -> str:
#     if not reranked_chunks:
#         return "I could not find this in the provided IS code excerpts."

#     top_score = reranked_chunks[0].get("rerank_score", 0)
#     if top_score < MIN_RERANK_SCORE:
#         return (
#             "I could not find this in the provided IS code excerpts "
#             f"(best match confidence too low: {top_score:.2f})."
#         )

#     context = build_context(reranked_chunks)
#     prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

#     response = ollama.chat(
#         model=LLM_MODEL,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": prompt},
#         ],
#         # temperature=0 (not 0.1) -- your test suite showed the same question
#         # producing different wrong answers across runs on identical data,
#         # which makes both manual debugging and automated evaluation unreliable.
#         # seed pins Ollama's sampler for reproducibility; note this reduces
#         # variance but doesn't guarantee bit-identical output every time.
#         options={"temperature": 0, "seed": 42, "num_predict": 8192}, think=False,
#     )
#     return response["message"]["content"]


# # ---------------------------------------------------------------------------
# # Interactive CLI
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     print("Loading corpus + building BM25 index ...")
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty — run ingest.py first.")
#         return
#     bm25_index = build_bm25_index(corpus)
#     print(f"Ready. {len(corpus['ids'])} chunks indexed.\n")

#     while True:
#         query = input("Ask a question (or 'exit'): ").strip()
#         if query.lower() == "exit":
#             break
#         if not query:
#             continue

#         # Example: pass Filters(content_type="table") to restrict to tables,
#         # or Filters(clause_prefix="6.2") to scope to a clause family.
#         filters = Filters()

#         candidates = hybrid_retrieve(query, corpus, bm25_index, collection, filters)
#         reranked = rerank(query, candidates)

#         print("\n--- Retrieved (post-rerank) ---")
#         for c in reranked:
#             meta = c["metadata"]
#             print(f"  [{c['rerank_score']:.3f}] {meta['source']} p{meta['page']} cl.{meta['clause']} ({meta['content_type']})")

#         answer = generate_answer(query, reranked)
#         print(f"\n--- Answer ---\n{answer}\n")


# if __name__ == "__main__":
#     main()
"""
IS-Code RAG — Query Pipeline (v1 / MVP)
==========================================
Hybrid retrieval (BM25 + semantic) -> metadata filtering -> rerank -> generate.

Install (in addition to ingest.py's requirements):
    pip install rank_bm25 flashrank

Usage:
    python GENERATION.py
    (interactive CLI loop; type 'exit' to quit)
"""

import re
import time
from dataclasses import dataclass

import ollama
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# ---------------------------------------------------------------------------
# Config — keep in sync with ingest.py
# ---------------------------------------------------------------------------
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:26b"

BM25_TOP_K = 15
SEMANTIC_TOP_K = 15
RRF_K = 60                 # standard reciprocal-rank-fusion constant
RERANK_TOP_N = 5           # how many chunks actually go to the LLM
MIN_RERANK_SCORE = 0.15    # below this, we refuse rather than guess — tune this against your own eval set, don't trust the default

# Fix #11: a long run of distinct prompts can accumulate enough server-side
# KV-cache across requests to push a memory-constrained VM into swap --
# confirmed via journalctl on a real crash: decode speed collapsed from
# ~100 tok/s to ~2 tok/s over the run, swap was at 3.7/4.0GiB, and the
# request that finally failed got a bare "500: EOF" from Ollama's health
# endpoint (the runner subprocess died, not the server itself -- it kept
# answering HEAD/GET afterward). Retrying after a short pause lets Ollama
# respawn that subprocess instead of losing the whole run to one bad request.
GENERATION_MAX_RETRIES = 3
GENERATION_RETRY_DELAY_SECONDS = [5, 20, 60]


# ---------------------------------------------------------------------------
# Corpus loading + BM25 index
# ---------------------------------------------------------------------------
def load_corpus(collection):
    """Pulls everything out of Chroma once so BM25 has a fixed corpus to index."""
    raw = collection.get(include=["documents", "metadatas"])
    return {
        "ids": raw["ids"],
        "documents": raw["documents"],
        "metadatas": raw["metadatas"],
    }


def build_bm25_index(corpus):
    tokenized = [doc.lower().split() for doc in corpus["documents"]]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------
@dataclass
class Filters:
    source: str | None = None          # exact match, e.g. "IS456.pdf"
    content_type: str | None = None    # "text" or "table"
    clause_prefix: str | None = None   # e.g. "6.2" matches 6.2, 6.2.1, 6.2.1.3...


def matches_filters(metadata: dict, filters: Filters) -> bool:
    if filters.source and metadata.get("source") != filters.source:
        return False
    if filters.content_type and metadata.get("content_type") != filters.content_type:
        return False
    if filters.clause_prefix:
        clause = metadata.get("clause", "")
        if not clause.startswith(filters.clause_prefix):
            return False
    return True


def chroma_where_clause(filters: Filters) -> dict | None:
    """Chroma's `where` handles exact-match fields natively; clause_prefix
    is a prefix match, which Chroma doesn't support directly, so that one
    is applied as a post-filter instead (see matches_filters)."""
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
# Hybrid retrieval — BM25 + semantic, fused with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def hybrid_retrieve(query: str, corpus, bm25_index, collection, filters: Filters):
    # --- BM25 side ---
    bm25_scores = bm25_index.get_scores(query.lower().split())
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]
    bm25_ranks = {corpus["ids"][i]: rank for rank, i in enumerate(bm25_ranked_idx)}

    # --- Semantic side ---
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    semantic_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=SEMANTIC_TOP_K,
        where=chroma_where_clause(filters),
    )
    semantic_ids = semantic_result["ids"][0]
    semantic_ranks = {doc_id: rank for rank, doc_id in enumerate(semantic_ids)}

    # --- Reciprocal Rank Fusion ---
    all_ids = set(bm25_ranks) | set(semantic_ranks)
    fused_scores = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[doc_id])
        if doc_id in semantic_ranks:
            score += 1.0 / (RRF_K + semantic_ranks[doc_id])
        fused_scores[doc_id] = score

    # Deterministic tiebreak by doc_id: Python randomizes string-hash seeds
    # per process, so plain `set`/`dict` iteration order for these chunk-ID
    # strings differs across separate script runs. With no secondary sort
    # key, tied/near-tied RRF scores (common among a table's sibling chunks)
    # would resolve to a DIFFERENT winner each run -- which is almost
    # certainly why the same query returned 4 different answers across runs
    # even after temperature was fixed to 0. Sorting by (-score, doc_id)
    # makes the order reproducible regardless of hash randomization.
    fused_ranked = sorted(fused_scores.items(), key=lambda kv: (-kv[1], kv[0]))

    # --- Build candidate list, applying clause_prefix post-filter here ---
    id_to_idx = {doc_id: i for i, doc_id in enumerate(corpus["ids"])}
    candidates = []
    for doc_id, _score in fused_ranked:
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
        })

    return candidates


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/home/civil/btech/ce1231034/flashrank_cache")  # small, CPU-friendly


AMENDMENT_PENALTY = 0.5
AMENDMENT_HEADER_RE = re.compile(r'^\W*(IS\s*456[^\n]{0,120}AMENDMENT|AMENDMENT|Amendment)', re.I)

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
    request = RerankRequest(query=query, passages=passages)
    results = _ranker.rerank(request)  # sorted by score desc

    by_id = {c["id"]: c for c in candidates}
    is_amendment_query = 'amendment' in query.lower()
    scored = []
    for r in results:
        chunk = by_id[r["id"]]
        score = r["score"]
        if not is_amendment_query and AMENDMENT_HEADER_RE.match(chunk["text"]):
            score -= AMENDMENT_PENALTY
        chunk["rerank_score"] = score
        scored.append(chunk)
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:RERANK_TOP_N]

# ---------------------------------------------------------------------------
# Generation — grounded, citation-forced, refuses when evidence is weak
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an assistant answering questions about Indian Standard (IS) codes \
for civil engineering, using ONLY the excerpts provided below.

Rules:
- Answer strictly from the provided excerpts. Do not use outside knowledge.
- Every claim must cite its source: clause number (e.g. "as per Cl. 6.2.1") or \
table/page reference.
- If the excerpts do not contain enough information to answer, say exactly: \
"I could not find this in the provided IS code excerpts." Do not guess or extrapolate.
- Some excerpts are marked [LOW-CONFIDENCE OCR]. This means the text extraction may be \
corrupted, especially for formulas/equations (broken symbols, garbled tokens, missing \
operators). If the information needed to answer comes from a low-confidence excerpt, \
say explicitly that the source text appears corrupted/incomplete and do NOT invent, \
guess, or "clean up" a formula to fill the gap — a wrong formula is worse than admitting \
the source couldn't be read cleanly.
- A table's "NOTES" (numbered notes below a table) are conditions that modify that \
table's values — read them together, not independently.
- Keep answers precise and technical; this is for engineering use, not casual conversation.
- If the excerpts state a formula, multiplier, or adjustment (e.g. "multiply by 0.8", "reduce by 5 mm") AND the question gives you the specific numbers to apply it to, perform that calculation explicitly and state the final numeric result — do not just restate the formula/multiplier and stop there. Show the arithmetic step, not just the rule.
"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c["metadata"]
        flag = " [LOW-CONFIDENCE OCR]" if meta.get("low_confidence") else ""
        label = f"[Source: {meta['source']}, page {meta['page']}, clause {meta['clause']}, type {meta['content_type']}{flag}]"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _call_ollama_chat_with_retry(prompt: str) -> dict:
    """Wraps ollama.chat() with retry+backoff. See the module-level comment
    on GENERATION_MAX_RETRIES for why this exists -- a 500/EOF here is often
    the runner subprocess dying under memory pressure, not a bad request,
    and the Ollama *server* usually survives and can serve the retry once
    it respawns."""
    last_error = None
    for attempt in range(1, GENERATION_MAX_RETRIES + 1):
        try:
            return ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0 (not 0.1) -- your test suite showed the same question
                # producing different wrong answers across runs on identical data,
                # which makes both manual debugging and automated evaluation unreliable.
                # seed pins Ollama's sampler for reproducibility; note this reduces
                # variance but doesn't guarantee bit-identical output every time.
                options={"temperature": 0, "seed": 42, "num_predict": 8192}, think=False,
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
    if not reranked_chunks:
        return "I could not find this in the provided IS code excerpts."

    top_score = reranked_chunks[0].get("rerank_score", 0)
    if top_score < MIN_RERANK_SCORE:
        return (
            "I could not find this in the provided IS code excerpts "
            f"(best match confidence too low: {top_score:.2f})."
        )

    context = build_context(reranked_chunks)
    prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Fix #11: generate_answer now never raises -- after exhausting retries
    # it returns a visible error string instead, so one bad question can't
    # take down a 70-question run. The caller (test_suite.py) treats this
    # like any other answer string and grades it (it'll score as a FAIL,
    # which is correct -- it IS a failure, just an infra one, not a RAG one).
    try:
        response = _call_ollama_chat_with_retry(prompt)
    except Exception as e:
        return f"[GENERATION ERROR after {GENERATION_MAX_RETRIES} attempts: {e}]"
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------
def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print("Loading corpus + building BM25 index ...")
    corpus = load_corpus(collection)
    if not corpus["ids"]:
        print("Collection is empty — run ingest.py first.")
        return
    bm25_index = build_bm25_index(corpus)
    print(f"Ready. {len(corpus['ids'])} chunks indexed.\n")

    while True:
        query = input("Ask a question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        # Example: pass Filters(content_type="table") to restrict to tables,
        # or Filters(clause_prefix="6.2") to scope to a clause family.
        filters = Filters()

        candidates = hybrid_retrieve(query, corpus, bm25_index, collection, filters)
        reranked = rerank(query, candidates)

        print("\n--- Retrieved (post-rerank) ---")
        for c in reranked:
            meta = c["metadata"]
            print(f"  [{c['rerank_score']:.3f}] {meta['source']} p{meta['page']} cl.{meta['clause']} ({meta['content_type']})")

        answer = generate_answer(query, reranked)
        print(f"\n--- Answer ---\n{answer}\n")


if __name__ == "__main__":
    main()