import csv
import chromadb
from GENERATION import (load_corpus, build_bm25_index, hybrid_retrieve, rerank,
                        Filters, CHROMA_DIR, COLLECTION_NAME, MIN_RERANK_SCORE)

rows = {r['ID']: r for r in csv.DictReader(open('IS456_RAG_Test_Suite.csv', encoding='utf-8'))}
IDS = ['E02', 'EM10', 'M05', 'X01', 'X02', 'X03', 'E07']

collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
corpus = load_corpus(collection)
bm25_index = build_bm25_index(corpus)

first = True
for i in IDS:
    q = rows[i]['Test Query']
    rr = rerank(q, hybrid_retrieve(q, corpus, bm25_index, collection, Filters()))
    top = rr[0] if rr else {}
    if first:
        print("chunk keys:", list(top.keys()))
        first = False
    score = top.get('rerank_score')
    refuse = score is None or score < MIN_RERANK_SCORE
    print(f"{i}  top_score={score}  threshold={MIN_RERANK_SCORE}  would_refuse={refuse}")
    print("   Q:", q[:90])
