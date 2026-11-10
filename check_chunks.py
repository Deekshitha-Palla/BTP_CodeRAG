import chromadb
c = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("is_codes")
for s in ["41.3.1", "8.2.1", "Special System", "Special system", "shells, folded plates"]:
    r = c.get(where_document={"$contains": s}, include=["documents"])
    print(f"== {s}: {len(r['ids'])} chunks")
    for d in r["documents"][:2]:
        print("   ", d[:160].replace("\n", " "))
