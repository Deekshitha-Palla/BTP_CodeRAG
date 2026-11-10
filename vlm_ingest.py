git remote -vimport json, os, sys
import ollama
import chromadb

CROPS = sys.argv[1]
MODEL = "gemma4:26b"
EMBED = "nomic-embed-text"
OUT = "vlm_records.jsonl"
TABLE_PROMPT = ("Transcribe this table from IS 456:2000 into markdown. Copy every number exactly as printed, "
                "including the row and column headers. If a cell is unreadable, write [UNREADABLE]. "
                "Output only the table.")
FIG_PROMPT = ("Describe this figure from IS 456:2000 for a structural engineer: its title, axis labels, "
              "curves or regions, and any values printed on it. Do not estimate values that are not printed.")

manifest = json.load(open(os.path.join(CROPS, "manifest.json")))
todo = [e for e in manifest if not e["near_top_of_page"]]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["file"] for l in open(OUT)}
print(f"{len(todo)} crops to process, {len(done)} already done", flush=True)

with open(OUT, "a") as out:
    for e in todo:
        if e["file"] in done:
            continue
        prompt = TABLE_PROMPT if e["kind"] == "table" else FIG_PROMPT
        img = open(os.path.join(CROPS, e["file"]), "rb").read()
        resp = ollama.chat(model=MODEL,
                           messages=[{"role": "user", "content": prompt, "images": [img]}],
                           options={"temperature": 0, "seed": 42, "num_predict": 8192}, think=False)
        text = resp["message"]["content"].strip()
        rec = {"file": e["file"], "page": e["page"], "kind": e["kind"], "caption": e["caption"],
               "truncated": resp.get("done_reason") == "length", "text": text}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(e["file"], len(text), "TRUNCATED" if rec["truncated"] else "", flush=True)

records = [json.loads(l) for l in open(OUT)]
bad = [r["file"] for r in records if r["truncated"]]
print("truncated (excluded, need manual check):", bad, flush=True)
records = [r for r in records if not r["truncated"]]

chunks = []
for r in records:
    header = f"IS 456:2000, page {r['page']}, {r['kind']}"
    if r["caption"]:
        header += f": {r['caption']}"
    chunks.append({"id": "vlm-" + r["file"][:-4], "text": header + "\n" + r["text"],
                   "page": r["page"], "kind": r["kind"], "caption": r["caption"]})

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("is_codes")
docs = [c["text"] for c in chunks]
embs = ollama.embed(model=EMBED, input=docs)["embeddings"]
collection.upsert(ids=[c["id"] for c in chunks], documents=docs, embeddings=embs,
                  metadatas=[{"page": c["page"], "source": "vlm", "kind": c["kind"],
                              "caption": c["caption"]} for c in chunks])
print("added", len(chunks), "chunks; collection now has", collection.count(), flush=True)
