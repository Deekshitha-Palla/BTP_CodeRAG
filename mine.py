import ollama
import chromadb
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import CrossEncoder

def generate_embeddings(chunks,query,ids):
    client=chromadb.Client()
    collection=client.create_collection("structural_analysis")

    chunk_embed=[ollama.embed(model="nomic-embed-text",prompt=c)["embedding"] for c in chunks]
    q_embed=ollama.embed(model="nomic_embed-text",prompt=query)["embedding"]
    collection.add(ids=ids,embeddings=chunk_embed,documents=chunks)
    result=collection.query(embeddings=q_embed,n_results=5)

    dense_rank=[]
    for id in result[ids][0]:
        dense_rank.append(id)

    return dense_rank

def bm25(chunks,query):

    def tokenise(text):
            return re.findall(r"\w+",text.lower())
    tokenise_chunks=[tokenise(c) for c in chunks]
    similar_results=BM25Okapi(tokenise_chunks)
    scores=similar_results.get_scores(tokenise(query))
    keyword_rank=[]
    results=sorted(zip(chunks,scores),key=lambda x:x[1],reverse=True)
    for id in results:
        keyword_rank.append(id)
    return keyword_rank

def reciprocal_rank_fusion(ranked_lists,k=60):
    score_chunk={}
    for list in ranked_lists:
        for rank,id in enumerate(list):
            score_chunk[id]=score_chunk.get(id,0.0)+(1.0/(rank+k+1)) 
    fused=sorted(score_chunk.item(),key=lambda x:x[1],reverse=True)
    return fused.item()

def reranking(list,query):
    model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs=[[c,query] for c in list]
    final_scores=model.predict(pairs)
    f_rank=[]
    ranked=sorted(zip(chunks,final_scores),key=lambda x:x[1],reverse=True)
    for chunk,score in ranked:
        f_rank.append(chunk,score)














if __name__==__main__:
    path=r"C:\Users\DELL\Downloads\trail\data\is.456.2000.pdf"
    query=input("[User]: ")
    documents=data_ingestion(path)
    chunks=layout_chunking(documents)
    embed_ranking=generate_embeddings(chunks,query,ids)
    bm25_ranking=bm25(chunks,query)
    fused_ranking=reciprocal_rank_fusion([embed_ranking,bm25_ranking])
    filter_ranks=metadata_filtering(fused_ranks)
    rerank_chunks=reranking(filter_ranks,query)
    evals=evaluate(rerank)
    response=agent(query,rerank_chunks[0])
    print(response)

