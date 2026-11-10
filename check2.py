import chromadb
c = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("is_codes")
for s in ["satisfactory evidence", "adequacy", "Special", "8.2.7", "Curing", "curing"]:
    r = c.get(where_document={"$contains": s}, include=["documents"])
    print(f"== {s}: {len(r['ids'])} chunks")
    for d in r["documents"][:2]:
        print("   ", d[:160].replace("\n", " "))
