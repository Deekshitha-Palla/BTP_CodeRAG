import json
from collections import Counter
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
c = client.get_or_create_collection("is_codes")
existing = c.get(include=["metadatas"])["metadatas"]
print("existing content_type values:", Counter(m.get("content_type") for m in existing))

KIND_TO_TYPE = {"table": "table", "picture": "figure"}
recs = [json.loads(l) for l in open("vlm_records.jsonl")]
ids, metas = [], []
for r in recs:
    if r["truncated"]:
        continue
    ids.append("vlm-" + r["file"][:-4])
    metas.append({"source": "IS456.pdf", "page": r["page"], "clause": "unknown",
                  "content_type": KIND_TO_TYPE[r["kind"]], "low_confidence": False})
c.update(ids=ids, metadatas=metas)
print("updated", len(ids), "VLM chunks; collection now has", c.count())
