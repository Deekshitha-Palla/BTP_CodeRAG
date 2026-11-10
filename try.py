import pickle
from ragbot import generate_embeddings, bm25_search, reciprocal_rank_fusion

with open("./chunks_cache.pkl", "rb") as f:
    cached_data = pickle.load(f)
chunks = cached_data["chunks"]

print(f"Total chunks in cache: {len(chunks)}")

# ---- 1. Search for any chunk that looks like it holds the real clause 23.2.1 values ----
print("\n=== CHUNKS MENTIONING '23.2' AND ('span' or 'depth') ===")
candidates = [
    c for c in chunks
    if "23.2" in c["text"] and ("span" in c["text"].lower() or "depth" in c["text"].lower())
]
for c in candidates:
    print(f"\n[{c['chunk_id']}] page={c['page_num']} method={c.get('extraction_method')}")
    print(repr(c["text"]))

if not candidates:
    print("  NONE FOUND mentioning both '23.2' and span/depth — might use different wording.")
    print("\n  Trying broader search: any chunk mentioning 'effective depth' near pages 46-49...")
    broader = [c for c in chunks if c["page_num"] in (46, 47, 48, 49)]
    for c in broader:
        print(f"\n[{c['chunk_id']}] page={c['page_num']}")
        print(repr(c["text"][:400]))

# ---- 2. Retrieval rank check for whichever candidate(s) we found ----
def rank_of(target_id, ranked_list):
    return ranked_list.index(target_id) + 1 if target_id in ranked_list else "NOT IN RESULTS"

query = "What are the basic values of span to effective depth ratios for structural spans up to 10 meters to control vertical deflection in beams?"

print(f"\n=== RETRIEVAL RANKS FOR SPAN/DEPTH QUERY ===")
dense = generate_embeddings(chunks, query)
sparse = bm25_search(chunks, query)
fused = reciprocal_rank_fusion([dense, sparse])

lookup = {c["chunk_id"]: c for c in chunks}
print("Top 5 fused results:")
for i, cid in enumerate(fused[:5], start=1):
    c = lookup[cid]
    print(f"  {i}. [{cid}] page={c['page_num']} method={c.get('extraction_method')}")
    print(f"     {c['text'][:200]}")

if candidates:
    for c in candidates:
        cid = c["chunk_id"]
        print(f"\nRanks for {cid} (page {c['page_num']}):")
        print(f"  Dense rank: {rank_of(cid, dense)}")
        print(f"  BM25 rank: {rank_of(cid, sparse)}")
        print(f"  Fused rank: {rank_of(cid, fused)}")

# ---- 3. Full dump of pages near the TOC-listed location (37) AND the
# cross-reference page (88), to visually locate the actual clause 23.2.1 table ----
print("\n\n=== FULL TEXT OF ALL CHUNKS ON PAGES 36-39 (TOC says clause 23.2 is on page 37) ===")
near_toc = [c for c in chunks if c["page_num"] in (36, 37, 38, 39)]
near_toc.sort(key=lambda c: c["page_num"])
for c in near_toc:
    print(f"\n[{c['chunk_id']}] page={c['page_num']}")
    print(repr(c["text"]))

print("\n\n=== FULL TEXT OF ALL CHUNKS ON PAGES 49-52 (corrected for +13 printed/metadata offset) ===")
corrected = [c for c in chunks if c["page_num"] in (49, 50, 51, 52)]
corrected.sort(key=lambda c: c["page_num"])
for c in corrected:
    print(f"\n[{c['chunk_id']}] page={c['page_num']}")
    print(repr(c["text"]))