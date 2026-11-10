import csv, chromadb
from GENERATION import (load_corpus, build_bm25_index, hybrid_retrieve, rerank,
                        generate_answer, Filters, CHROMA_DIR, COLLECTION_NAME)

rows = {r['ID']: r for r in csv.DictReader(open('IS456_RAG_Test_Suite.csv', encoding='utf-8'))}
collection = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION_NAME)
corpus = load_corpus(collection)
bm25 = build_bm25_index(corpus)

for i in ['E02', 'M05', 'EM10']:
    q = rows[i]['Test Query']
    rr = rerank(q, hybrid_retrieve(q, corpus, bm25, collection, Filters()))
    print('=====', i, q)
    for c in rr[:2]:
        print(round(c['rerank_score'], 3), c['metadata'])
        print('   ', c['text'][:250].replace('\n', ' '))
    print('ANSWER:', generate_answer(q, rr)[:300])
