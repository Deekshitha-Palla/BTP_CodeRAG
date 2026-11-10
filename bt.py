import ollama
import os
from langchain_community.document_loaders import PyPdfLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import union
import chromadb


def data_ingest(dir_path):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=[]
    for file in os.walk(dir_path):
        pdf_loader=PyPdfLoader(file)
        pages=pdf_loader.load()
        chunks.append(text_splitter.split_documents(pages))
    return chunks

def embeddings(chunks:union[list,str],ids):
    client=chromadb.Client()
    collection=client.create_collection("structural")

    embed=[ollama.embed(model="nomic-embed-text",input=d)["embeddings"] for d in chunks]
    collection.add(ids=ids,embeddings=embed,documents=chunks)












if __name__==__main__:
    dir_path=r"C:\Users\DELL\Downloads\trail\data"
    chunks=data_ingest(dir_path)
    embeddings(chunks)
    print("Enter quit or exit to terminate the session\n")
    while(True):
        user_query=input("[User]: ")
        if user_query.lower()=="exit" or user_query.lower()=="quit":
            print("Had a great time, looking forward to see you soon!")
            break
        embeddings(user_query)
        bm25_ranks=bm25_ranking(chunks,user_query)
        fused_ranked_docs=reciprocal_rank_fusion([bm25_ranks,dense_ranks])
        filtered_docs=metadata_filtering(fused_ranked_docs)
        final_docs=reranking(filtered_docs)
        response=agent(final_docs)
        print(f"[Assistant]: {response}")
