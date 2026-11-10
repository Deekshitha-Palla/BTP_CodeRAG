import ollama
import chromadb
from rank_bm25 import BM25Okapi
import re


client=chromadb.Client()
collection=client.create_collection("trail_collection")

docs = [
    "The allowable bending stress for A992 steel is 50 ksi.",
    "A992 steel has a yield strength around 50 ksi, commonly used in wide-flange beams.",
    "Concrete compressive strength is typically tested at 28 days curing.",
    "I had pasta for lunch today.",
]
ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]

embeddings=[ollama.embeddings(model="nomic-embed-text",prompt=q)["embedding"] for q in docs]
collection.add(ids=ids,embeddings=embeddings,documents=docs)

query=ollama.embeddings(model="nomic-embed-text",prompt="What is the yield strength of steel used in beams?")["embedding"]
q_embed=collection.query(query_embeddings=[query],n_results=4)

def tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_docs = [tokenize(d) for d in docs]
bm25 = BM25Okapi(tokenized_docs)

query = "What is the yield strength of steel used in beams?"
scores = bm25.get_scores(tokenize(query))

dense_ranking=[]
for id in q_embed["ids"][0]:
    dense_ranking.append(id)

bm25_ranking = [doc_id for doc_id, score in sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)]

def reciprocal_rank_fusion(ranking_lists,k=60):
    score_i={}
    for list in ranking_lists:
        for rank,id in enumerate(list):
            score_i[id]=score_i.get(id,0.0)+(1.0/(k+rank+1))
    fused_ranks=sorted(score_i.items(),key=lambda x:x[1],reverse=True)
    return fused_ranks

fused=reciprocal_rank_fusion([dense_ranking,bm25_ranking])

for item,score in fused:
    print(f"{item}:{score:.5f}")
