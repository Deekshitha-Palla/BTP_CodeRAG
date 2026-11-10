import ollama
import numpy as np

EMBED_MODEL="nomic-embed-text"
def get_embedding(text):
    response=ollama.embeddings(model=EMBED_MODEL,prompt=text)
    return np.array(response["embedding"])

def cosine_similarity(a,b):
    cos_similarity=(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_similarity

sent1 = "The allowable bending stress for A992 steel is 50 ksi."
sent2 = "A992 steel has a yield strength around 50 ksi."
sent3 = "I had pasta for lunch today."

e1, e2, e3 = get_embedding(sent1), get_embedding(sent2), get_embedding(sent3)

print("Related sentences similarity:", cosine_similarity(e1, e2))
print("Unrelated sentence similarity:", cosine_similarity(e1, e3))