import chromadb
from GENERATION import (CHROMA_DIR, COLLECTION_NAME, load_corpus, build_bm25_index,
                        hybrid_retrieve, Filters, _ranker, RerankRequest, AMENDMENT_HEADER_RE)
from test_suite import QUESTIONS

WANT = ["E02", "M04", "MH10", "M01", "A08"]
TOP = 6

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
corpus = load_corpus(collection)
bm25_index = build_bm25_index(corpus)

for q in QUESTIONS:
    if q["id"] not in WANT:
        continue
    cands = hybrid_retrieve(q["query"], corpus, bm25_index, collection, Filters())
    passages = [{"id": c["id"], "text": c["text"]} for c in cands]
    results = _ranker.rerank(RerankRequest(query=q["query"], passages=passages))
    by_id = {c["id"]: c for c in cands}
    print(f"\n=== {q['id']} (pool {len(cands)}): {q['query'][:90]}")
    for rank, r in enumerate(results[:TOP], 1):
        c = by_id[r["id"]]
        amend = "Y" if AMENDMENT_HEADER_RE.match(c["text"]) else "-"
        snippet = c["text"][:70].replace("\n", " ")
        print(f"  {rank}. raw={r['score']:7.3f} amend={amend} {str(r['id'])[:28]:28} {snippet}")
