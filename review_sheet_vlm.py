import csv, json
d = json.load(open('test_suite_checkpoint_with_vlm.json'))
rows = {r['ID']: r for r in csv.DictReader(open('IS456_RAG_Test_Suite.csv', encoding='utf-8-sig'))}
for qid, v in d.items():
    if v['result'] != 'REVIEW':
        continue
    r = rows.get(qid, {})
    print('=' * 90)
    print(qid, '|', r.get('Difficulty', ''))
    print('QUESTION:', r.get('Test Query', ''))
    print('EXPECTED:', r.get('Expected Retrieval / Answer Anchor', ''))
    print('GOLD LOCATION:', r.get('Gold Evidence / Location', ''))
    print('ACTUAL:', v['answer'].strip())
    print()
