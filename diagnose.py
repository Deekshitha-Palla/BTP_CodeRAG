"""
Run this from C:\\Users\\DELL\\Downloads\\trail (same folder as claude.py).
Dumps everything needed to diagnose the formwork + exposure table issues
in one shot: which chunks exist, whether they were excluded/kept, and
where they rank for each query.
"""
import pickle
from claude import generate_embeddings, bm25_search, reciprocal_rank_fusion

with open("./chunks_cache.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks in cache: {len(chunks)}")
table_chunks = [c for c in chunks if c.get("extraction_method") == "table"]
print(f"Total 'table' chunks: {len(table_chunks)}")

# ---- 1. Find all table chunks on the formwork page (adjust if needed) ----
print("\n=== TABLE CHUNKS ON PAGE 38 (formwork) ===")
page_38_tables = [c for c in table_chunks if c["page_num"] == 38]
for c in page_38_tables:
    print(f"\n[{c['chunk_id']}]")
    print(repr(c["text"]))

# ---- 2. Find all table chunks mentioning the exposure table ----
print("\n=== TABLE CHUNKS MENTIONING 'Exposure' AND 'Mild' ===")
exposure_tables = [c for c in table_chunks if "Exposure" in c["text"] and "Mild" in c["text"]]
for c in exposure_tables:
    print(f"\n[{c['chunk_id']}] page {c['page_num']}")
    print(repr(c["text"]))

if not exposure_tables:
    print("  NONE FOUND — the exposure table isn't in the corpus as a table chunk at all.")

# ---- 3. Check for scrambled prose duplicates of each ----
prose_chunks = [c for c in chunks if c.get("extraction_method") != "table"]
print("\n=== PROSE CHUNKS ON PAGE 38 (checking for scrambled duplicate) ===")
page_38_prose = [c for c in prose_chunks if c["page_num"] == 38]
for c in page_38_prose:
    print(f"\n[{c['chunk_id']}] method={c.get('extraction_method')}")
    print(repr(c["text"][:300]))

# ---- 4. Retrieval rank check for both queries ----
def rank_of(target_id, ranked_list):
    return ranked_list.index(target_id) + 1 if target_id in ranked_list else "NOT IN RESULTS"

formwork_query = "What is the minimum period specified for striking formwork for the soffit of slabs?"
exposure_query = "What are the nominal concrete cover requirements for Mild, Severe, and Extreme exposure conditions?"

for label, query in [("FORMWORK", formwork_query), ("EXPOSURE", exposure_query)]:
    print(f"\n=== RETRIEVAL RANKS FOR: {label} ===")
    dense = generate_embeddings(chunks, query)
    sparse = bm25_search(chunks, query)
    fused = reciprocal_rank_fusion([dense, sparse])

    print("Top 5 fused results:")
    lookup = {c["chunk_id"]: c for c in chunks}
    for i, cid in enumerate(fused[:5], start=1):
        c = lookup[cid]
        method = c.get("extraction_method")
        print(f"  {i}. [{cid}] page={c['page_num']} method={method}")
        print(f"     {c['text'][:150]}")