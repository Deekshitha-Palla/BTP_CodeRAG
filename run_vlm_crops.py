import json, os, sys
import ollama

CROPS = sys.argv[1]
WANT = set(sys.argv[2:])
MODEL = "gemma4:26b"
TABLE_PROMPT = ("Transcribe this table from IS 456:2000 into markdown. Copy every number exactly as printed, "
                "including the row and column headers. If a cell is unreadable, write [UNREADABLE]. "
                "Output only the table.")
FIG_PROMPT = ("Describe this figure from IS 456:2000 for a structural engineer: its title, axis labels, "
              "curves or regions, and any values printed on it. Do not estimate values that are not printed.")

manifest = json.load(open(os.path.join(CROPS, "manifest.json")))
out = []
for e in manifest:
    if e["file"] not in WANT:
        continue
    prompt = TABLE_PROMPT if e["kind"] == "table" else FIG_PROMPT
    img = open(os.path.join(CROPS, e["file"]), "rb").read()
    resp = ollama.chat(model=MODEL,
                       messages=[{"role": "user", "content": prompt, "images": [img]}],
                       options={"temperature": 0, "seed": 42})
    rec = dict(e)
    rec["vlm_text"] = resp["message"]["content"].strip()
    out.append(rec)
    print(f"== {e['file']}\n{rec['vlm_text'][:600]}\n")

json.dump(out, open("vlm_crops_test.json", "w"), indent=2, ensure_ascii=False)
