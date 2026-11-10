from sentence_transformers import CrossEncoder
model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


query = "What is the yield strength of steel used in beams?"
chunks = [
    "The allowable bending stress for A992 steel is 50 ksi.",
    "A992 steel has a yield strength around 50 ksi, commonly used in wide-flange beams.",
    "Concrete compressive strength is typically tested at 28 days curing.",
    "I had pasta for lunch today.",
]

q_chunk_pairs=[[query,c] for c in chunks]
scores=model.predict(q_chunk_pairs)

ranked=sorted(zip(chunks,scores),key=lambda x:x[1],reverse=True)
for text,score in ranked:
    print(f"{text} with {score:.4f}")