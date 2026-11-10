import ollama
import chromadb
# chromadb is a opensource database of ollama


client=chromadb.Client()
collection=client.create_collection("trail_collection")
# name of the collection is trail_collection

docs = [
    "The allowable bending stress for A992 steel is 50 ksi.",
    "A992 steel has a yield strength around 50 ksi, commonly used in wide-flange beams.",
    "Concrete compressive strength is typically tested at 28 days curing.",
    "I had pasta for lunch today.",
]
ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]

embeddings=[ollama.embeddings(model="nomic-embed-text",prompt=q)["embedding"] for q in docs]
collection.add(ids=ids,embeddings=embeddings,documents=docs)
# All docs,ids,embeddings are lists.
# here ollama.embeddings() creates a json body- dictionary in which model,prompt,embedding,start time,time elapsed,prompt_count etc get stored 
# however we are only concerned about ollama.embeddings()["embedding"] is extracted
query=ollama.embeddings(model="nomic-embed-text",prompt="What is the yield strength of steel used in beams?")["embedding"]
q_embed=collection.query(query_embeddings=[query],n_results=4)
# similarly embed the query.
# print("====The similar context results are=====\n")
# i=0
# for doc,dis in zip(q_embed["documents"][0],q_embed["distances"][0]):
#     i=i+1
#     print(f"{i}. {doc} with {dis:.4f}")
#here zip makes our life easy by binding the documents and distances lists(basically 2 different lists into a single one)
# also [0] after ["documents"] is because the chromadb's collection.query is capable of taking multiple queries so it can form list of lists
# in our case as there is a single query, access from the top most(0) list else you will get a typeError at .4f as list.4f is meaningless.