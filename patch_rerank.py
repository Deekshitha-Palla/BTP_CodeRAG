p = 'GENERATION.py'
lines = open(p, encoding='utf-8').read().split('\n')

start = next(i for i, l in enumerate(lines) if l.startswith('def rerank('))
end = len(lines)
for j in range(start + 1, len(lines)):
    if lines[j] and not lines[j][0].isspace():
        end = j
        break

new_block = r'''AMENDMENT_PENALTY = 0.5
AMENDMENT_HEADER_RE = re.compile(r'^\W*(IS\s*456[^\n]{0,120}AMENDMENT|AMENDMENT|Amendment)', re.I)

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
    request = RerankRequest(query=query, passages=passages)
    results = _ranker.rerank(request)  # sorted by score desc

    by_id = {c["id"]: c for c in candidates}
    is_amendment_query = 'amendment' in query.lower()
    scored = []
    for r in results:
        chunk = by_id[r["id"]]
        score = r["score"]
        if not is_amendment_query and AMENDMENT_HEADER_RE.match(chunk["text"]):
            score -= AMENDMENT_PENALTY
        chunk["rerank_score"] = score
        scored.append(chunk)
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:RERANK_TOP_N]
'''.split('\n')

lines[start:end] = new_block
if not any(l.startswith('import re') for l in lines):
    lines.insert(0, 'import re')

text = '\n'.join(lines)
assert "'AMENDMENT' not in" not in text, 'old filter still present'
open(p, 'w', encoding='utf-8').write(text)
print(f'patched: replaced lines {start+1}-{end}')
