from rank_bm25 import BM25Okapi
import re

def tokenize(text):
    return re.findall(r"\w+", text.lower())

docs = [
    "The allowable bending stress for A992 steel is 50 ksi.",
    "A992 steel has a yield strength around 50 ksi, commonly used in wide-flange beams.",
    "Concrete compressive strength is typically tested at 28 days curing.",
    "I had pasta for lunch today.",
]

tokenized_docs = [tokenize(d) for d in docs]
bm25 = BM25Okapi(tokenized_docs)

query = "What is the yield strength of steel used in beams?"
scores = bm25.get_scores(tokenize(query))

print("Query:", query)
print("\nBM25 scores (higher = more relevant):")
for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True):
    print(f"  ({score:.4f}) {doc}")