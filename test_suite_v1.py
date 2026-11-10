# """
# IS-Code RAG — Automated Test Suite
# =====================================
# Runs every question you've been manually typing, in one batch, against
# whatever's currently in chroma_db. No typing, no waiting between questions.

# Grading is automatic ONLY where a question has a clear, checkable answer
# (expect_keywords). Formula/figure questions where correctness needs a human
# to actually read the math are marked REVIEW, not auto-passed/failed — don't
# trust a green checkmark on those; read the printed answer yourself.

# Usage:
#     python test_suite.py                  # run everything
#     python test_suite.py --category table # run one category only
#     python test_suite.py --verbose        # print full answers, not just verdicts
#     python test_suite.py --known-issues   # only re-run cases flagged as previously broken

# Output: prints a summary table, and saves full results to
# test_results_<timestamp>.json for later comparison across ingestion changes.
# """

# import argparse
# import json
# from datetime import datetime

# from GENERATION import load_corpus, build_bm25_index, hybrid_retrieve, rerank, generate_answer, Filters
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"

# # ---------------------------------------------------------------------------
# # Test cases — compiled from every question asked across your manual testing.
# # expect_keywords: list of groups; each group is "alt1|alt2" (any alt counts).
# #   match_mode "all"  -> every group must be found somewhere in the answer
# #   match_mode "any"  -> at least one group must be found
# #   [] / None         -> no auto-grading; verdict is always REVIEW
# # known_issue: True marks cases that were previously confirmed broken, so you
# #   can quickly check "did my last fix actually land?" with --known-issues.
# # ---------------------------------------------------------------------------
# TEST_CASES = [
#     # --- Simple direct facts ---
#     {"id": "TC01", "category": "simple_fact",
#      "question": "What is the characteristic compressive strength of M30 concrete at 28 days?",
#      "expect_keywords": ["30"], "match_mode": "all"},
#     {"id": "TC02", "category": "simple_fact",
#      "question": "What are the five environmental exposure conditions used to classify the general environment of concrete?",
#      "expect_keywords": ["mild", "moderate", "severe", "very severe", "extreme"], "match_mode": "all"},
#     {"id": "TC03", "category": "simple_fact",
#      "question": "What is the maximum cement content that should normally not be exceeded unless special consideration is given?",
#      "expect_keywords": ["450"], "match_mode": "all"},
#     {"id": "TC04", "category": "simple_fact",
#      "question": "What is the approximate total shrinkage strain that may be taken for design in the absence of test data?",
#      "expect_keywords": ["0.0003|0.000 3|3 x 10"], "match_mode": "any"},
#     {"id": "TC05", "category": "simple_fact",
#      "question": "What is the maximum compressive strain in concrete in bending at the ultimate limit state?",
#      "expect_keywords": ["0.0035|0.003 5"], "match_mode": "all"},
#     {"id": "TC06", "category": "simple_fact",
#      "question": "What is the minimum number of longitudinal bars required in rectangular and circular columns?",
#      "expect_keywords": ["four|4", "six|6"], "match_mode": "all"},
#     {"id": "TC07", "category": "simple_fact",
#      "question": "What minimum percentage of reinforcement should be provided in either direction of a slab when high-strength deformed bars are used?",
#      "expect_keywords": ["0.12"], "match_mode": "all"},
#     {"id": "TC08", "category": "simple_fact",
#      "question": "What restrictions are specified for the diameter of reinforcing bars in slabs?",
#      "expect_keywords": ["one-eighth|1/8"], "match_mode": "all"},

#     # --- Table lookups ---
#     {"id": "TC09", "category": "table",
#      "question": "According to Table 16, what nominal concrete cover is required for severe exposure?",
#      "expect_keywords": ["45"], "match_mode": "all"},
#     {"id": "TC10", "category": "table",
#      "question": "List the nominal concrete covers for mild, moderate, severe, very severe and extreme exposure conditions.",
#      "expect_keywords": ["20", "30", "45", "50", "75"], "match_mode": "all"},
#     {"id": "TC11", "category": "table",
#      "question": "List the characteristic compressive strengths corresponding to M25, M35, M50 and M60 concrete.",
#      "expect_keywords": ["25", "35", "50", "60"], "match_mode": "all"},
#     {"id": "TC12", "category": "table",
#      "question": "What ranges of slump are suggested for low, medium and high degrees of workability?",
#      "expect_keywords": ["25-75|25 to 75", "100-150|100 to 150"], "match_mode": "all"},
#     {"id": "TC13", "category": "table",
#      "question": "According to Table 4, for sulphate Class 5 conditions, what minimum cement content and maximum free water-cement ratio are specified?",
#      "expect_keywords": ["400", "0.40|0.4"], "match_mode": "all"},
#     {"id": "TC14", "category": "table", "known_issue": True,
#      "question": "According to Table 16A, what nominal cover is required for a simply supported slab for 2 hours of fire resistance?",
#      "expect_keywords": ["35"], "match_mode": "all",
#      "note": "Known bug: pipeline has answered 40 (the beam column) instead of 35 (slab column)."},
#     {"id": "TC15", "category": "table",
#      "question": "According to Table 16A, what nominal cover is required for a simply supported beam for 2 hours of fire resistance?",
#      "expect_keywords": ["40"], "match_mode": "all"},
#     {"id": "TC16", "category": "table",
#      "question": "For 3 hours of fire resistance, compare the nominal cover requirements of simply supported beams, continuous beams, simply supported slabs and continuous slabs.",
#      "expect_keywords": [], "note": "Compound multi-cell lookup — read the answer manually."},

#     # --- Notes attached to tables (previously the ListItem-drop bug) ---
#     {"id": "TC17", "category": "notes", "known_issue": True,
#      "question": "According to Note 3 of Table 16, when is a 5 mm reduction in nominal cover permitted?",
#      "expect_keywords": ["m35|M 35", "5 mm|5mm"], "match_mode": "all",
#      "note": "Was silently dropped (ListItem not captured) — should now retrieve real note content."},
#     {"id": "TC18", "category": "notes", "known_issue": True,
#      "question": "For mild exposure, can the nominal cover be reduced when the main reinforcement bars are no larger than 12 mm? If so, by how much?",
#      "expect_keywords": ["5 mm|5mm"], "match_mode": "all"},
#     {"id": "TC19", "category": "notes", "known_issue": True,
#      "question": "Can the actual concrete cover normally deviate from the required nominal cover? State the permitted deviation given in the document.",
#      "expect_keywords": ["10"], "match_mode": "any"},

#     # --- Formulas (OCR-fragile — auto-grading can't catch a wrong number here) ---
#     {"id": "TC20", "category": "formula_risk",
#      "question": "According to Section 6.2.2, what formula is given for estimating the flexural strength of concrete from its characteristic cube compressive strength?",
#      "expect_keywords": [],
#      "note": "Correct: fc = 0.7*sqrt(fck). OCR drops the sqrt sign — watch for 0.7*fck (wrong) vs 0.7*sqrt(fck) (right)."},
#     {"id": "TC21", "category": "formula_risk",
#      "question": "According to Section 6.2.3.1, what expression is used for the short-term static modulus of elasticity of concrete?",
#      "expect_keywords": [], "note": "Correct: Ec = 5000*sqrt(fck). Watch for garbled/incomplete formulas."},
#     {"id": "TC22", "category": "formula_risk",
#      "question": "Estimate the flexural strength of M25 concrete using the relationship given in the document.",
#      "expect_keywords": [], "note": "Correct answer is 3.5 N/mm^2 (0.7*sqrt(25)). A confidently-stated 17.5 means the sqrt was silently dropped."},
#     {"id": "TC23", "category": "formula_risk",
#      "question": "Estimate the short-term static modulus of elasticity of M25 concrete using the expression given in the document.",
#      "expect_keywords": [], "note": "Correct answer is 25000 N/mm^2 (5000*sqrt(25))."},
#     {"id": "TC24", "category": "formula_risk",
#      "question": "State the minimum shear reinforcement expression given for beams and define all the terms appearing in it.",
#      "expect_keywords": ["0.4", "0.87"], "match_mode": "all",
#      "note": "Check the actual inequality (Asv/(b*sv) >= 0.4/(0.87*fy)) is stated, not just the variable definitions."},

#     # --- Calculations ---
#     {"id": "TC25", "category": "calculation",
#      "question": "A beam has a web area of 300,000 mm² and a web depth of 900 mm. Determine the minimum total area of side-face reinforcement required.",
#      "expect_keywords": ["300"], "match_mode": "all"},
#     {"id": "TC26", "category": "calculation",
#      "question": "A beam has a web depth of 800 mm, web thickness of 250 mm and web area of 240,000 mm². Determine whether side-face reinforcement is required; if required, calculate its minimum total area, state how it should be distributed, and determine the maximum permissible spacing.",
#      "expect_keywords": ["240", "250"], "match_mode": "all"},
#     {"id": "TC27", "category": "calculation",
#      "question": "For a two-way simply supported slab of shorter span 3.5 m reinforced with Fe415 high-strength deformed bars, determine the modified span-to-depth ratio given by the document.",
#      "expect_keywords": ["28"], "match_mode": "all"},
#     {"id": "TC28", "category": "calculation", "known_issue": True,
#      "question": "Compare the modified span-to-depth ratios for simply supported and continuous two-way slabs when Fe415 reinforcement is used.",
#      "expect_keywords": ["28", "32"], "match_mode": "all",
#      "note": "Was previously answered as 28 for BOTH (wrong) — continuous should be 40*0.8=32."},
#     {"id": "TC29", "category": "calculation",
#      "question": "A simply supported two-way slab has a shorter span of 3.5 m and uses Fe415 reinforcement. Using the deflection guidance in the document, estimate the corresponding effective depth.",
#      "expect_keywords": ["125"], "match_mode": "any",
#      "note": "This is actually computable (3500mm / 28) — previously refused as 'could not find'. Check if it now does the division."},
#     {"id": "TC30", "category": "calculation", "known_issue": True,
#      "question": "A concrete member is under severe exposure and uses M35 concrete. Determine the applicable nominal cover after using the reduction permitted by the document.",
#      "expect_keywords": ["40"], "match_mode": "any",
#      "note": "Correct once Note 3 is retrievable and applied: 45mm - 5mm = 40mm."},
#     {"id": "TC31", "category": "calculation",
#      "question": "Compare the nominal cover required for severe and very severe exposure. If M40 concrete is used and the permitted reduction is applicable, determine the resulting nominal cover for each.",
#      "expect_keywords": [], "note": "Compound — check both base values (45/50) and whether the M35+ reduction gets applied to both."},

#     # --- Figures (caption-only in v1 — no VLM description) ---
#     {"id": "TC32", "category": "figure",
#      "question": "Which graph in the document relates percentage tension reinforcement to a modification factor, and what quantity is represented by the different curves?",
#      "expect_keywords": ["fig. 4|figure 4|fig 4"], "match_mode": "any"},
#     {"id": "TC33", "category": "figure", "known_issue": True,
#      "question": "What relationship is represented by the graph for compression reinforcement?",
#      "expect_keywords": ["fig. 5|figure 5|fig 5"], "match_mode": "any",
#      "note": "Was previously misattributed to Fig. 21 (wrong) instead of Fig. 5."},
#     {"id": "TC34", "category": "figure",
#      "question": "What types of reinforcement stress-strain curves are illustrated in the document?",
#      "expect_keywords": ["fig. 21|figure 21", "fig. 23|figure 23"], "match_mode": "all"},
#     {"id": "TC35", "category": "figure",
#      "question": "What does Figure 4 represent, and what variables are shown on its horizontal and vertical axes?",
#      "expect_keywords": [], "note": "Axis labels need VLM description, not just a caption — expected to still be incomplete in v1."},

#     # --- Paraphrase / robustness checks (same fact, different wording) ---
#     {"id": "TC36", "category": "paraphrase",
#      "question": "At what beam web depth does side-face reinforcement become necessary?",
#      "expect_keywords": ["750"], "match_mode": "all"},
#     {"id": "TC37", "category": "paraphrase",
#      "question": "I have a very deep beam. At what point do I need additional reinforcement along its sides, and how should that reinforcement be arranged?",
#      "expect_keywords": ["750"], "match_mode": "all"},
#     {"id": "TC38", "category": "paraphrase",
#      "question": "How does the code account for the use of Fe415 bars when applying the simplified deflection guidance for short two-way slabs?",
#      "expect_keywords": ["0.8"], "match_mode": "all"},

#     # --- Adversarial / out-of-scope (correct behavior IS refusal) ---
#     {"id": "TC39", "category": "adversarial",
#      "question": "If I know the concrete grade but don't have a direct tensile test result, how can I approximately determine its resistance to cracking in bending?",
#      "expect_keywords": ["could not find"], "match_mode": "any",
#      "note": "This phrasing doesn't match source wording well — refusal is acceptable here, not a failure."},
# ]


# def check_answer(answer: str, expect_keywords, match_mode: str = "all"):
#     if not expect_keywords:
#         return "REVIEW"
#     answer_lower = answer.lower()
#     hits = []
#     for group in expect_keywords:
#         alts = [a.strip().lower() for a in group.split("|")]
#         hits.append(any(alt in answer_lower for alt in alts))
#     passed = all(hits) if match_mode == "all" else any(hits)
#     return "PASS" if passed else "FAIL"


# def run_suite(cases, verbose=False):
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty — run ingest.py first.")
#         return []
#     bm25_index = build_bm25_index(corpus)
#     print(f"Loaded {len(corpus['ids'])} chunks. Running {len(cases)} test case(s)...\n")

#     results = []
#     for case in cases:
#         candidates = hybrid_retrieve(case["question"], corpus, bm25_index, collection, Filters())
#         reranked = rerank(case["question"], candidates)
#         answer = generate_answer(case["question"], reranked)
#         verdict = check_answer(answer, case.get("expect_keywords"), case.get("match_mode", "all"))

#         result = {
#             "id": case["id"], "category": case["category"], "question": case["question"],
#             "answer": answer, "verdict": verdict, "known_issue": case.get("known_issue", False),
#             "note": case.get("note", ""),
#             "top_sources": [f"{c['metadata']['source']} p{c['metadata']['page']} "
#                              f"cl.{c['metadata']['clause']} ({c['metadata']['content_type']})"
#                              for c in reranked[:3]],
#         }
#         results.append(result)

#         flag = {"PASS": "✅", "FAIL": "❌", "REVIEW": "❔"}[verdict]
#         print(f"{flag} {case['id']} [{case['category']}] {case['question'][:70]}")
#         if verbose or verdict in ("FAIL", "REVIEW"):
#             print(f"    -> {answer}")
#             if case.get("note"):
#                 print(f"    (note: {case['note']})")
#         print()

#     return results


# def print_summary(results):
#     print("=" * 70)
#     print("SUMMARY")
#     by_category = {}
#     for r in results:
#         by_category.setdefault(r["category"], {"PASS": 0, "FAIL": 0, "REVIEW": 0})
#         by_category[r["category"]][r["verdict"]] += 1

#     for cat, counts in sorted(by_category.items()):
#         print(f"  {cat:15s} PASS={counts['PASS']:2d}  FAIL={counts['FAIL']:2d}  REVIEW={counts['REVIEW']:2d}")

#     total_pass = sum(r["verdict"] == "PASS" for r in results)
#     total_fail = sum(r["verdict"] == "FAIL" for r in results)
#     total_review = sum(r["verdict"] == "REVIEW" for r in results)
#     print(f"\n  TOTAL: {total_pass} pass / {total_fail} fail / {total_review} need manual review "
#           f"(out of {len(results)})")

#     failed_known_issues = [r for r in results if r["verdict"] == "FAIL" and r["known_issue"]]
#     fixed_known_issues = [r for r in results if r["verdict"] == "PASS" and r["known_issue"]]
#     if fixed_known_issues:
#         print(f"\n  Known issues now FIXED: {', '.join(r['id'] for r in fixed_known_issues)}")
#     if failed_known_issues:
#         print(f"  Known issues STILL BROKEN: {', '.join(r['id'] for r in failed_known_issues)}")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--category", help="only run cases in this category")
#     parser.add_argument("--known-issues", action="store_true", help="only run cases flagged known_issue=True")
#     parser.add_argument("--verbose", action="store_true", help="print full answers for every case, not just failures")
#     args = parser.parse_args()

#     cases = TEST_CASES
#     if args.category:
#         cases = [c for c in cases if c["category"] == args.category]
#     if args.known_issues:
#         cases = [c for c in cases if c.get("known_issue")]

#     results = run_suite(cases, verbose=args.verbose)
#     if results:
#         print_summary(results)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         out_path = f"test_results_{timestamp}.json"
#         with open(out_path, "w") as f:
#             json.dump(results, f, indent=2)
#         print(f"\nFull results saved to {out_path}")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Blind CSV Test Runner
# =======================================
# Runs every question from a CSV test suite (columns: ID, Difficulty, Question
# Type, Test Query, Gold Evidence / Location, Expected Retrieval / Answer
# Anchor, Primary RAG Failure Mode) against the pipeline -- but first strips
# any explicit Table/Clause/Figure/Annex/Section NUMBER from the question text,
# so retrieval has to actually find the right place in the document instead of
# being told where to look. This tests retrieval accuracy, not instruction-
# following.

# The Gold Evidence / Expected Answer columns are NOT sent to the model --
# they're kept only in the output file for you to grade against by eye.

# Usage:
#     python run_csv_suite.py                          # default filename below
#     python run_csv_suite.py --csv my_tests.csv
#     python run_csv_suite.py --limit 10                # quick smoke test first
# """

# import argparse
# import csv
# import re
# from datetime import datetime

# from GENERATION import load_corpus, build_bm25_index, hybrid_retrieve, rerank, generate_answer, Filters
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# DEFAULT_CSV = "IS456_RAG_Test_Suite.csv"

# # ---------------------------------------------------------------------------
# # Blinding: strip explicit Table/Clause/Figure/Annex/Section numbers from the
# # query text so the question can't be answered by following an instruction
# # instead of by actually retrieving the right content. Order matters little
# # here since the patterns target distinct keywords.
# # ---------------------------------------------------------------------------
# _RANGE = r"(\s*(?:,|and|to|\u2013|-)\s*\d+[A-Za-z]?)*"
# BLIND_PATTERNS = [
#     (re.compile(rf"\bTables?\s+\d+[A-Za-z]?{_RANGE}\b", re.IGNORECASE), "the relevant table"),
#     (re.compile(rf"\bFigs?\.?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bFigures?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bClauses?\s+\d+(?:\.\d+)*{_RANGE}\b", re.IGNORECASE), "the relevant clause"),
#     (re.compile(r"\bAnnex(?:es)?\s+[A-Z](\s*(?:,|and)\s*[A-Z])*\b", re.IGNORECASE), "the relevant annex"),
#     (re.compile(r"\bSections?\s+\d+(?:\.\d+)*\b", re.IGNORECASE), "the relevant section"),
# ]


# def blind(text: str) -> str:
#     for pattern, replacement in BLIND_PATTERNS:
#         text = pattern.sub(replacement, text)
#     return re.sub(r"\s+", " ", text).strip()


# def run_suite(rows, verbose=False):
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty -- run ingest.py first.")
#         return []
#     bm25_index = build_bm25_index(corpus)
#     print(f"Loaded {len(corpus['ids'])} chunks. Running {len(rows)} question(s) blind (numbers stripped)...\n")

#     results = []
#     for i, row in enumerate(rows, 1):
#         original_q = row["Test Query"].strip()
#         blind_q = blind(original_q)

#         candidates = hybrid_retrieve(blind_q, corpus, bm25_index, collection, Filters())
#         reranked = rerank(blind_q, candidates)
#         answer = generate_answer(blind_q, reranked)

#         result = {
#             "id": row["ID"],
#             "difficulty": row["Difficulty"],
#             "question_type": row["Question Type"],
#             "original_query": original_q,
#             "blind_query": blind_q,
#             "gold_evidence": row["Gold Evidence / Location"],
#             "expected_answer": row["Expected Retrieval / Answer Anchor"],
#             "failure_mode": row["Primary RAG Failure Mode"],
#             "actual_answer": answer,
#             "retrieved_sources": " | ".join(
#                 f"{c['metadata']['source']} p{c['metadata']['page']} cl.{c['metadata']['clause']} ({c['metadata']['content_type']})"
#                 for c in reranked
#             ),
#         }
#         results.append(result)

#         print(f"[{i}/{len(rows)}] {row['ID']} ({row['Difficulty']})")
#         print(f"  blind query: {blind_q}")
#         if verbose:
#             print(f"  gold: {row['Expected Retrieval / Answer Anchor']}")
#             print(f"  actual: {answer}")
#         print()

#     return results


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", default=DEFAULT_CSV, help="path to the test suite CSV")
#     parser.add_argument("--limit", type=int, default=None, help="only run the first N rows (smoke test)")
#     parser.add_argument("--verbose", action="store_true", help="print gold + actual answer for every row")
#     args = parser.parse_args()

#     with open(args.csv, newline="", encoding="utf-8-sig") as f:
#         rows = list(csv.DictReader(f))
#     if args.limit:
#         rows = rows[: args.limit]

#     results = run_suite(rows, verbose=args.verbose)
#     if not results:
#         return

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_path = f"blind_test_results_{timestamp}.csv"
#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
#         writer.writeheader()
#         writer.writerows(results)

#     print(f"\nDone. {len(results)} question(s) run.")
#     print(f"Full side-by-side results (original query, blinded query, gold, actual answer, "
#           f"retrieved sources) saved to {out_path} -- open it in a spreadsheet to grade by eye.")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Live Preliminary Accuracy Check
# ==================================================
# Self-contained: the 70 test queries + gold answers are embedded below (pulled
# straight from your graded blind_test_results CSV), so this needs nothing but
# a running Ollama + populated Chroma collection. Runs every query through the
# real pipeline (same blind()/hybrid_retrieve/rerank/generate_answer as
# run_csv_suite.py) and prints a tick or cross immediately after each one, so
# you can watch it live and get a number before your presentation.

# IMPORTANT — read before you quote this number to anyone:
# Grading here is an automated heuristic (checks whether the expected numbers
# and keywords show up in the model's answer text). It is NOT a substitute for
# the row-by-row human read you'd do against the CSV -- it WILL misgrade a few
# rows (a correct answer phrased differently than expected_answer, or a wrong
# answer that happens to contain the right numbers in passing). Treat the
# final score as a fast, approximate "are we in the right ballpark" gut-check
# for your slides, not a certified accuracy figure. Three-way grading:
#   PASS   - looks correct (or correctly refused a trap question)
#   FAIL   - looks wrong (or refused when it should have answered)
#   REVIEW - heuristic isn't confident either way -- eyeball this one yourself

# Usage:
#     python accuracy_check.py               # full run, live output
#     python accuracy_check.py --limit 10    # quick smoke test first
# """
# import argparse
# import re

# from GENERATION import load_corpus, build_bm25_index, hybrid_retrieve, rerank, generate_answer, Filters
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"

# # ---------------------------------------------------------------------------
# # The 70 questions, extracted from blind_test_results_20260922_123615.csv
# # ---------------------------------------------------------------------------
# QUESTIONS = [
#     {
#         "id": "E01",
#         "difficulty": "Easy",
#         "query": "What is the scope of IS 456:2000? As a site structural engineer, what type of concrete structures does this code generally cover?",
#         "gold_evidence": "Clause 1, p.11",
#         "expected_answer": "Scope; plain and reinforced concrete; special structures covered by respective standards"
#     },
#     {
#         "id": "E02",
#         "difficulty": "Easy",
#         "query": "What is meant by a plain concrete structure in IS 456?",
#         "gold_evidence": "Clause 1.1.1, p.11",
#         "expected_answer": "Reinforcement, if provided, is ignored for determining strength"
#     },
#     {
#         "id": "E03",
#         "difficulty": "Easy",
#         "query": "In a beam design calculation, what do the symbols d, D, Ast and Ld represent in IS 456?",
#         "gold_evidence": "Clause 4, pp.11–12",
#         "expected_answer": "Effective depth; overall depth; tension steel area; development length"
#     },
#     {
#         "id": "E04",
#         "difficulty": "Easy",
#         "query": "Which grades of ordinary Portland cement are explicitly listed in IS 456?",
#         "gold_evidence": "Clause 5.1, p.13",
#         "expected_answer": "33, 43 and 53 grade OPC"
#     },
#     {
#         "id": "E05",
#         "difficulty": "Easy",
#         "query": "What is the recommended nominal maximum coarse aggregate size for most concrete work?",
#         "gold_evidence": "Clause 5.3.3, p.14",
#         "expected_answer": "20 mm is suitable for most work"
#     },
#     {
#         "id": "E06",
#         "difficulty": "Easy",
#         "query": "What is the minimum pH value specified for water used in concrete?",
#         "gold_evidence": "Clause 5.4.2, p.14",
#         "expected_answer": "pH not less than 6"
#     },
#     {
#         "id": "E07",
#         "difficulty": "Easy",
#         "query": "According to Table 1, what is the permissible chloride limit for reinforced concrete mixing water?",
#         "gold_evidence": "Table 1, p.15",
#         "expected_answer": "2000 mg/L as Cl"
#     },
#     {
#         "id": "E08",
#         "difficulty": "Easy",
#         "query": "What is the minimum concrete grade specified for reinforced concrete work in IS 456?",
#         "gold_evidence": "Foreword / Clause 6.1, pp.2,15",
#         "expected_answer": "M20"
#     },
#     {
#         "id": "E09",
#         "difficulty": "Easy",
#         "query": "As a site engineer, where does IS 456 say the required workability of concrete should come from?",
#         "gold_evidence": "Clause 7, p.16",
#         "expected_answer": "Mix proportions should provide adequate workability for placing conditions and available compaction method"
#     },
#     {
#         "id": "E10",
#         "difficulty": "Easy",
#         "query": "Why does IS 456 emphasize curing when discussing durability?",
#         "gold_evidence": "Clause 8.2.1, p.17",
#         "expected_answer": "Adequate curing avoids harmful effects of early moisture loss"
#     },
#     {
#         "id": "EM01",
#         "difficulty": "Easy-Medium",
#         "query": "For a reinforced concrete member exposed to 'severe' conditions, what nominal cover does Table 16 require?",
#         "gold_evidence": "Table 16, p.47",
#         "expected_answer": "45 mm"
#     },
#     {
#         "id": "EM02",
#         "difficulty": "Easy-Medium",
#         "query": "For 'extreme' exposure, what nominal concrete cover is required according to Table 16?",
#         "gold_evidence": "Table 16, p.47",
#         "expected_answer": "75 mm"
#     },
#     {
#         "id": "EM03",
#         "difficulty": "Easy-Medium",
#         "query": "What are the five environmental exposure categories listed in Table 3?",
#         "gold_evidence": "Table 3, p.18",
#         "expected_answer": "Mild, moderate, severe, very severe, extreme"
#     },
#     {
#         "id": "EM04",
#         "difficulty": "Easy-Medium",
#         "query": "For M30 concrete, what standard deviation is assumed in Table 8?",
#         "gold_evidence": "Table 8, p.23",
#         "expected_answer": "Use the M30 row from Table 8"
#     },
#     {
#         "id": "EM05",
#         "difficulty": "Easy-Medium",
#         "query": "What are the permissible limits for organic and inorganic solids in mixing water according to Table 1?",
#         "gold_evidence": "Table 1, p.15",
#         "expected_answer": "200 mg/L organic; 3000 mg/L inorganic"
#     },
#     {
#         "id": "EM06",
#         "difficulty": "Easy-Medium",
#         "query": "A site engineer asks why 20 mm aggregate is generally preferred. What restriction in Clause 5.3.3 links aggregate size to member thickness and reinforcement congestion?",
#         "gold_evidence": "Clause 5.3.3, p.14",
#         "expected_answer": "Nominal maximum size not greater than one-fourth of minimum member thickness, subject to proper placing around reinforcement; smaller sizes may be needed for thin/congested sections"
#     },
#     {
#         "id": "EM07",
#         "difficulty": "Easy-Medium",
#         "query": "If I use deformed bars instead of plain bars, how does the design bond stress change? Also tell me which clause provides the base bond stresses.",
#         "gold_evidence": "Clause 26.2.1.1, p.43",
#         "expected_answer": "Deformed-bar values are 60% higher; base values are given for plain bars"
#     },
#     {
#         "id": "EM08",
#         "difficulty": "Easy-Medium",
#         "query": "A concrete mix uses two admixtures simultaneously. What does the code require before accepting that combination?",
#         "gold_evidence": "Clause 5.5.6, p.16",
#         "expected_answer": "Obtain data on interaction and ensure compatibility"
#     },
#     {
#         "id": "EM09",
#         "difficulty": "Easy-Medium",
#         "query": "What is the maximum cement content mentioned in the durability provisions, and what risks are cited if it is exceeded?",
#         "gold_evidence": "Clause 8.2.4.2, p.19",
#         "expected_answer": "450 kg/m³ excluding fly ash/GGBS; increased cracking/thermal cracking/alkali-silica-reaction risk unless specially considered"
#     },
#     {
#         "id": "EM10",
#         "difficulty": "Easy-Medium",
#         "query": "For a 2.0 m span-to-width ratio, which coefficient should I retrieve from Table 26 for an interior panel's positive mid-span moment?",
#         "gold_evidence": "Annex D, Table 26, p.91",
#         "expected_answer": "0.049 for the short-span coefficient row at l_y/l_x = 2.0"
#     },
#     {
#         "id": "M01",
#         "difficulty": "Medium",
#         "query": "Look at Fig. 1 for fire resistance. If I know the required fire-resistance period and member type, how is the figure intended to be used?",
#         "gold_evidence": "Fig. 1, p.34",
#         "expected_answer": "It relates minimum member dimensions and cover to fire-resistance periods/member types"
#     },
#     {
#         "id": "M02",
#         "difficulty": "Medium",
#         "query": "In Fig. 2, where should the critical section for shear normally be located relative to the face of a support when the special condition of Clause 22.6.2.1 applies?",
#         "gold_evidence": "Fig. 2 + Clause 22.6.2.1, p.37",
#         "expected_answer": "A section less than d from the support may use the shear at distance d when the reaction introduces compression"
#     },
#     {
#         "id": "M03",
#         "difficulty": "Medium",
#         "query": "For a fixed dead-load-plus-imposed-load case, what coefficient is used for the support moment at an interior support in Table 12?",
#         "gold_evidence": "Table 12, p.36",
#         "expected_answer": "1/12"
#     },
#     {
#         "id": "M04",
#         "difficulty": "Medium",
#         "query": "For an unfixed imposed load, what is the shear coefficient at an interior support according to Table 13?",
#         "gold_evidence": "Table 13, p.36",
#         "expected_answer": "0.6"
#     },
#     {
#         "id": "M05",
#         "difficulty": "Medium",
#         "query": "What is the purpose of Fig. 4, 'Modification Factor for Tension Reinforcement', in beam/slab deflection control?",
#         "gold_evidence": "Fig. 4, p.38",
#         "expected_answer": "It provides the modification factor used with the basic span-to-effective-depth relationship for tension reinforcement"
#     },
#     {
#         "id": "M06",
#         "difficulty": "Medium",
#         "query": "What does Fig. 5 provide, and when would I use it in a flexural-member deflection check?",
#         "gold_evidence": "Fig. 5, p.39",
#         "expected_answer": "Modification factor for compression reinforcement in deflection control"
#     },
#     {
#         "id": "M07",
#         "difficulty": "Medium",
#         "query": "What does Fig. 6 show for flanged beams?",
#         "gold_evidence": "Fig. 6, p.39",
#         "expected_answer": "Reduction factors for span-to-effective-depth ratios of flanged beams"
#     },
#     {
#         "id": "M08",
#         "difficulty": "Medium",
#         "query": "A beam is 800 mm deep. Does IS 456 require side-face reinforcement, and what minimum percentage is specified?",
#         "gold_evidence": "Clause 26.5.1.3, p.47",
#         "expected_answer": "Yes; when web depth exceeds 750 mm, total side-face reinforcement ≥0.1% of web area, distributed on two faces"
#     },
#     {
#         "id": "M09",
#         "difficulty": "Medium",
#         "query": "What is the maximum spacing of vertical shear stirrups specified in Clause 26.5.1.5?",
#         "gold_evidence": "Clause 26.5.1.5, p.47",
#         "expected_answer": "0.75d"
#     },
#     {
#         "id": "M10",
#         "difficulty": "Medium",
#         "query": "A beam has 16 mm bars in mild exposure. What reduction in nominal cover is permitted by the note to Table 16?",
#         "gold_evidence": "Table 16 Note 1, p.47",
#         "expected_answer": "5 mm reduction for main reinforcement up to 12 mm diameter only; therefore 16 mm does not qualify"
#     },
#     {
#         "id": "MH01",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete with plain tension bars, what is the design bond stress? If the bars are deformed bars conforming to IS 1786, what value should be used?",
#         "gold_evidence": "Clause 26.2.1.1, p.43",
#         "expected_answer": "Plain bar: 1.4 N/mm²; deformed-bar value increased by 60% = 2.24 N/mm²"
#     },
#     {
#         "id": "MH02",
#         "difficulty": "Medium-Hard",
#         "query": "A beam uses three bundled bars in contact. By what percentage should the development length of each bar be increased?",
#         "gold_evidence": "Clause 26.2.1.2, p.43",
#         "expected_answer": "20%"
#     },
#     {
#         "id": "MH03",
#         "difficulty": "Medium-Hard",
#         "query": "A plain tension bar has a 45° bend. What anchorage value can that bend provide, and what is the maximum allowed anchorage value from bends?",
#         "gold_evidence": "Clause 26.2.2.1, p.43",
#         "expected_answer": "4φ per 45° bend, subject to maximum 16φ"
#     },
#     {
#         "id": "MH04",
#         "difficulty": "Medium-Hard",
#         "query": "For a secondary stirrup bent through 135°, what continuation beyond the bend is required for complete development according to Clause 26.2.2.4?",
#         "gold_evidence": "Clause 26.2.2.4, p.43",
#         "expected_answer": "At least six bar diameters"
#     },
#     {
#         "id": "MH05",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete and a longitudinal tension steel ratio of 1.0%, what is the design shear strength τc from Table 19?",
#         "gold_evidence": "Table 19, p.73",
#         "expected_answer": "0.64 N/mm²"
#     },
#     {
#         "id": "MH06",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete at a steel percentage of 1.0%, compare τc from Table 19 with the maximum shear stress τc,max from Table 20. What design check does the code imply?",
#         "gold_evidence": "Tables 19–20, pp.73–74",
#         "expected_answer": "τc = 0.64 N/mm²; τc,max must be checked separately from Table 20; nominal shear stress must not exceed τc,max"
#     },
#     {
#         "id": "MH07",
#         "difficulty": "Medium-Hard",
#         "query": "If a section is within 2d of a support, what enhancement to shear strength can be considered, and what condition must hold at the support face?",
#         "gold_evidence": "Clause 40.5.1, p.74",
#         "expected_answer": "Enhanced shear strength may be considered; shear stress at the support face must remain below Table 20 limits"
#     },
#     {
#         "id": "MH08",
#         "difficulty": "Medium-Hard",
#         "query": "For a beam subjected to shear V and torsional moment T, how does IS 456 define equivalent shear for torsion design?",
#         "gold_evidence": "Clause 41.3.1, p.75",
#         "expected_answer": "Ve = Vu + 1.6T/b"
#     },
#     {
#         "id": "MH09",
#         "difficulty": "Medium-Hard",
#         "query": "For limit-state collapse, what partial safety factors are specified for concrete and steel material strengths?",
#         "gold_evidence": "Clause 36.4.2.1, p.68",
#         "expected_answer": "1.5 for concrete and 1.15 for steel"
#     },
#     {
#         "id": "MH10",
#         "difficulty": "Medium-Hard",
#         "query": "For the DL + IL + WL load combination, what partial safety factors does Table 18 specify for DL, IL and WL at the collapse limit state, and what are the corresponding serviceability values?",
#         "gold_evidence": "Table 18, p.68",
#         "expected_answer": "For DL+IL+WL at collapse: γf = 1.2 for DL, 1.2 for IL and 0.8 for WL; for serviceability: 1.0 for DL, 0.8 for IL and 0.8 for WL."
#     },
#     {
#         "id": "H01",
#         "difficulty": "Hard",
#         "query": "Use Fig. 21 to explain the assumed concrete stress-strain relationship used for limit-state flexural design. What are the key strain/stress values visible in the figure?",
#         "gold_evidence": "Fig. 21 + Clause 38.1, p.69",
#         "expected_answer": "Parabolic/rectangular idealization; maximum concrete compression strain 0.0035; design compression stress based on 0.67fck/γm"
#     },
#     {
#         "id": "H02",
#         "difficulty": "Hard",
#         "query": "In Fig. 22, what do 0.36fck xu and 0.42xu represent in the concrete stress block?",
#         "gold_evidence": "Fig. 22 + Clause 38.1, p.69",
#         "expected_answer": "Design stress-block intensity/resultant-related parameters and depth of compressive force from compression face"
#     },
#     {
#         "id": "H03",
#         "difficulty": "Hard",
#         "query": "For reinforcement stress-strain behaviour, which figure does IS 456 refer to and what partial safety factor is used for steel in flexural limit-state design?",
#         "gold_evidence": "Fig. 23 + Clause 38.1, p.69",
#         "expected_answer": "Fig. 23; γm = 1.15 for steel"
#     },
#     {
#         "id": "H04",
#         "difficulty": "Hard",
#         "query": "A structural engineer wants to use a core test to assess doubtful concrete. What acceptance percentages does Clause 17.4.3 give for the average equivalent cube strength and individual core strength?",
#         "gold_evidence": "Clause 17.4.3, p.31",
#         "expected_answer": "Average equivalent cube strength ≥85% of specified cube strength; no individual core <75%"
#     },
#     {
#         "id": "H05",
#         "difficulty": "Hard",
#         "query": "Clause 17.6.2 specifies a load test on a flexural member. What load is applied, for how long, and what recovery is expected after removing the imposed load?",
#         "gold_evidence": "Clause 17.6.2–17.6.3, p.31",
#         "expected_answer": "Dead load + 1.25× imposed load for 24 h; at least 75% recovery within 24 h after imposed load removal"
#     },
#     {
#         "id": "H06",
#         "difficulty": "Hard",
#         "query": "A flat slab has an opening near a column. How should the critical shear perimeter be treated according to Fig. 15?",
#         "gold_evidence": "Fig. 15 + Clause 31.6, p.59",
#         "expected_answer": "The critical section is modified depending on opening location; certain opening portions may be treated/subtracted or as a free edge"
#     },
#     {
#         "id": "H07",
#         "difficulty": "Hard",
#         "query": "In Fig. 13, what geometric distance is used to locate the critical section around a column/drop for flat-slab shear?",
#         "gold_evidence": "Fig. 13 + Clause 31.6, p.58",
#         "expected_answer": "d/2 offsets from the support/column face are shown"
#     },
#     {
#         "id": "H08",
#         "difficulty": "Hard",
#         "query": "What does Fig. 14 illustrate for flat slabs with free edges or free corners?",
#         "gold_evidence": "Fig. 14 + Clause 31.6, p.59",
#         "expected_answer": "How free edges/corners affect the critical shear section"
#     },
#     {
#         "id": "H09",
#         "difficulty": "Hard",
#         "query": "How would you use Fig. 26 together with Table 28 to estimate effective length of a column in a framed structure?",
#         "gold_evidence": "Annex E, Fig. 26 + Table 28, pp.92–94",
#         "expected_answer": "Use restraint/sway condition and end restraints to obtain effective-length ratio; Table 28 gives idealized end-condition values"
#     },
#     {
#         "id": "H10",
#         "difficulty": "Hard",
#         "query": "What does Fig. 27 represent, and why is it relevant when the frame has lateral sway?",
#         "gold_evidence": "Annex E, Fig. 27, p.93",
#         "expected_answer": "Effective-length ratios for a column in a frame without restraint against sway"
#     },
#     {
#         "id": "A01",
#         "difficulty": "Advanced",
#         "query": "I am designing a 250 mm thick RC slab in a very severe exposure condition. What durability information do I need to retrieve before fixing concrete grade, water-cement ratio, minimum cement content and cover?",
#         "gold_evidence": "Tables 3, 5, 16, p.18 and pp.20,47",
#         "expected_answer": "Exposure category must be mapped to durability requirements across concrete quality and cover tables"
#     },
#     {
#         "id": "A02",
#         "difficulty": "Advanced",
#         "query": "A beam is heavily reinforced with small clear spacing. How should I select nominal maximum aggregate size using Clause 5.3.3.1?",
#         "gold_evidence": "Clause 5.3.3.1, p.14",
#         "expected_answer": "Usually 5 mm less than minimum clear distance between main bars or 5 mm less than minimum cover, whichever is smaller"
#     },
#     {
#         "id": "A03",
#         "difficulty": "Advanced",
#         "query": "A retaining structure uses concrete exposed to sulphates. Which table should I consult, and what inputs from the exposure/environment should I identify before selecting the concrete mix?",
#         "gold_evidence": "Table 4 + Clause 8.2.4, p.19",
#         "expected_answer": "Table 4; sulphate exposure condition, sulphate concentration/type, cement type, water-cement ratio and concrete requirements"
#     },
#     {
#         "id": "A04",
#         "difficulty": "Advanced",
#         "query": "A beam carries torsion plus shear. Explain the retrieval chain from Clause 41.3.1 to Tables 19 and 20 and then to Clause 41.4 for reinforcement design.",
#         "gold_evidence": "Clauses 41.3–41.4 + Tables 19–20, pp.75–76",
#         "expected_answer": "Compute equivalent shear; compare with τc and τc,max; if required provide longitudinal and transverse torsion reinforcement"
#     },
#     {
#         "id": "A05",
#         "difficulty": "Advanced",
#         "query": "A continuous beam designer wants to redistribute moments. What are the main restrictions in Clause 37.1.1, including the 70% condition and the maximum reduction rule?",
#         "gold_evidence": "Clause 37.1.1, pp.68–69",
#         "expected_answer": "Equilibrium maintained; resistance not below 70% of elastic moment; reduction constraints including not more than 30% of numerically largest elastic moment, subject to stated conditions"
#     },
#     {
#         "id": "A06",
#         "difficulty": "Advanced",
#         "query": "For a column frame, how do I determine whether the column can be treated as no-sway using the stability index Q?",
#         "gold_evidence": "Annex E.2, p.92",
#         "expected_answer": "Q ≤ 0.04 → no-sway; otherwise sway"
#     },
#     {
#         "id": "A07",
#         "difficulty": "Advanced",
#         "query": "A column has an effective-length ratio above the threshold mentioned in Annex E. How does that affect the choice between the simplified and more exact approach?",
#         "gold_evidence": "Annex E, pp.92–94",
#         "expected_answer": "Use the appropriate effective-length relationships/figures; the source distinguishes normal idealized usage and more exact analysis"
#     },
#     {
#         "id": "A08",
#         "difficulty": "Advanced",
#         "query": "If I need to calculate design surface crack width, what variables does Annex F require, and what is the main crack-width expression?",
#         "gold_evidence": "Annex F, p.95",
#         "expected_answer": "acr, cmin, εm, h, x and the specified expression for design surface crack width"
#     },
#     {
#         "id": "A09",
#         "difficulty": "Advanced",
#         "query": "How does Annex C define total deflection in terms of short-term and long-term components, and what long-term effects are separately considered?",
#         "gold_evidence": "Annex C, pp.88–89",
#         "expected_answer": "Total deflection = short-term + long-term; long-term components include shrinkage and creep"
#     },
#     {
#         "id": "A10",
#         "difficulty": "Advanced",
#         "query": "For a rectangular slab supported on four sides, how do I choose the bending-moment coefficients from Annex D based on panel type and l_y/l_x?",
#         "gold_evidence": "Annex D, Tables 26–27, pp.90–91",
#         "expected_answer": "Identify panel/support condition, determine aspect ratio, then retrieve short- and long-span coefficients from the appropriate table"
#     },
#     {
#         "id": "X01",
#         "difficulty": "Advanced",
#         "query": "What is the minimum concrete grade for prestressed concrete according to IS 456:2000?",
#         "gold_evidence": "Not directly supported; scope excludes special systems not covered by this code",
#         "expected_answer": "Should state that the uploaded source does not provide a direct prestressed-concrete minimum-grade rule rather than inventing one"
#     },
#     {
#         "id": "X02",
#         "difficulty": "Advanced",
#         "query": "What is the seismic zone factor Z for Delhi according to this document?",
#         "gold_evidence": "Not supported by this document; earthquake forces are referenced but seismic zoning values are not given here",
#         "expected_answer": "Should not fabricate a value"
#     },
#     {
#         "id": "X03",
#         "difficulty": "Advanced",
#         "query": "According to Table 19, what is the permissible shear stress for M10 concrete?",
#         "gold_evidence": "Table 19 starts at M15; M10 is not a listed grade",
#         "expected_answer": "Should identify that M10 is absent from Table 19 instead of extrapolating"
#     },
#     {
#         "id": "X04",
#         "difficulty": "Advanced",
#         "query": "If a flat slab has shear stress exactly above 1.5τc, can I simply increase the reinforcement indefinitely?",
#         "gold_evidence": "Clause 31.6.3.2, p.59",
#         "expected_answer": "No; if shear stress exceeds 1.5τc, the flat slab shall be redesigned"
#     },
#     {
#         "id": "X05",
#         "difficulty": "Advanced",
#         "query": "Can I use sea water for reinforced concrete because the code permits it for plain concrete?",
#         "gold_evidence": "Clause 5.4.3, p.14",
#         "expected_answer": "No; the stated unavoidable-use exception is for plain concrete with no embedded steel"
#     },
#     {
#         "id": "X06",
#         "difficulty": "Advanced",
#         "query": "The note says '20 mm aggregate is suitable for most work', so is 20 mm always permitted in a heavily reinforced beam?",
#         "gold_evidence": "Clause 5.3.3.1, p.14",
#         "expected_answer": "No; heavily reinforced members may require a smaller nominal maximum aggregate size based on clear spacing/cover"
#     },
#     {
#         "id": "X07",
#         "difficulty": "Advanced",
#         "query": "Does a 45° bend always provide 4φ anchorage even if that exceeds the maximum bend anchorage permitted?",
#         "gold_evidence": "Clause 26.2.2.1, p.43",
#         "expected_answer": "No; 4φ per 45° bend is subject to the 16φ maximum"
#     },
#     {
#         "id": "X08",
#         "difficulty": "Advanced",
#         "query": "If Q = 0.04 exactly, should the column be classified as sway?",
#         "gold_evidence": "Annex E.2, p.92",
#         "expected_answer": "No; Q ≤ 0.04 is treated as no-sway"
#     },
#     {
#         "id": "X09",
#         "difficulty": "Advanced",
#         "query": "Does IS 456 require the use of only design-mix concrete and prohibit nominal mixes?",
#         "gold_evidence": "Clause 9, pp.21–22",
#         "expected_answer": "No; the contents explicitly include both design mix and nominal mix concrete"
#     },
#     {
#         "id": "X10",
#         "difficulty": "Advanced",
#         "query": "Can I use a special concrete structural system not covered by IS 456 without any additional evidence?",
#         "gold_evidence": "Clause 19, p.3",
#         "expected_answer": "The code permits special systems not covered only when satisfactory evidence of adequacy and safety is produced by analysis, test, or both"
#     }
# ]

# # ---------------------------------------------------------------------------
# # Blinding — identical to run_csv_suite.py, kept here so this file has no
# # dependency on it and stays comparable to your last graded run.
# # ---------------------------------------------------------------------------
# _RANGE = r"(\s*(?:,|and|to|\u2013|-)\s*\d+[A-Za-z]?)*"
# BLIND_PATTERNS = [
#     (re.compile(rf"\bTables?\s+\d+[A-Za-z]?{_RANGE}\b", re.IGNORECASE), "the relevant table"),
#     (re.compile(rf"\bFigs?\.?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bFigures?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bClauses?\s+\d+(?:\.\d+)*{_RANGE}\b", re.IGNORECASE), "the relevant clause"),
#     (re.compile(r"\bAnnex(?:es)?\s+[A-Z](\s*(?:,|and)\s*[A-Z])*\b", re.IGNORECASE), "the relevant annex"),
#     (re.compile(r"\bSections?\s+\d+(?:\.\d+)*\b", re.IGNORECASE), "the relevant section"),
# ]


# def blind(text: str) -> str:
#     for pattern, replacement in BLIND_PATTERNS:
#         text = pattern.sub(replacement, text)
#     return re.sub(r"\s+", " ", text).strip()


# # ---------------------------------------------------------------------------
# # Grading heuristic
# # ---------------------------------------------------------------------------
# TRAP_MARKERS = ("not supported", "not directly supported", "not listed", "not given",
#                 "not covered", "not specified", "excludes", "does not")
# STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "into", "value",
#              "values", "when", "which", "what", "where", "condition", "conditions",
#              "according", "then", "than", "also", "used", "use", "using"}
# NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# def is_refusal(answer: str) -> bool:
#     return answer.strip().lower().startswith("i could not find")


# def grade(q: dict, answer: str) -> str:
#     expected = q["expected_answer"]
#     expected_lower = expected.lower()
#     is_trap = any(marker in expected_lower for marker in TRAP_MARKERS)
#     refused = is_refusal(answer)

#     if is_trap:
#         return "PASS" if refused else "FAIL"
#     if refused:
#         return "FAIL"

#     answer_lower = answer.lower()
#     expected_numbers = NUMBER_RE.findall(expected)
#     expected_keywords = [w.strip(".,;:()").lower() for w in re.split(r"\s+", expected)
#                           if len(w) >= 4 and w.strip(".,;:()").lower() not in STOPWORDS]
#     expected_keywords = list(dict.fromkeys(expected_keywords))  # dedupe, keep order

#     num_score = None
#     if expected_numbers:
#         hits = sum(1 for n in expected_numbers if n in answer)
#         num_score = hits / len(expected_numbers)

#     kw_score = None
#     if expected_keywords:
#         hits = sum(1 for w in expected_keywords if w in answer_lower)
#         kw_score = hits / len(expected_keywords)

#     if num_score is not None and kw_score is not None:
#         score = 0.7 * num_score + 0.3 * kw_score
#     elif num_score is not None:
#         score = num_score
#     elif kw_score is not None:
#         score = kw_score
#     else:
#         return "REVIEW"

#     if score >= 0.6:
#         return "PASS"
#     if score < 0.3:
#         return "FAIL"
#     return "REVIEW"


# MARK = {"PASS": "\033[92m PASS\033[0m", "FAIL": "\033[91m FAIL\033[0m", "REVIEW": "\033[93m REVIEW\033[0m"}


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
#     args = parser.parse_args()

#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty — run ingest.py first.")
#         return
#     bm25_index = build_bm25_index(corpus)

#     questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS
#     print(f"Loaded {len(corpus['ids'])} chunks. Running {len(questions)} question(s) live...\n")

#     tally = {"PASS": 0, "FAIL": 0, "REVIEW": 0}
#     by_difficulty = {}

#     for i, q in enumerate(questions, 1):
#         blind_q = blind(q["query"])
#         candidates = hybrid_retrieve(blind_q, corpus, bm25_index, collection, Filters())
#         reranked = rerank(blind_q, candidates)
#         answer = generate_answer(blind_q, reranked)

#         result = grade(q, answer)
#         tally[result] += 1
#         diff = q["difficulty"]
#         by_difficulty.setdefault(diff, {"PASS": 0, "FAIL": 0, "REVIEW": 0})
#         by_difficulty[diff][result] += 1

#         print(f"[{i}/{len(questions)}] {q['id']} ({diff}) \u2014 {MARK[result]}")
#         print(f"    Q: {q['query']}")
#         print(f"    expected : {q['expected_answer']}")
#         shown = answer[:200] + ("..." if len(answer) > 200 else "")
#         print(f"    actual   : {shown}")
#         print()

#     total = sum(tally.values())
#     print("=" * 60)
#     print(f"PASS: {tally['PASS']}   FAIL: {tally['FAIL']}   REVIEW: {tally['REVIEW']}   (of {total})")
#     print(f"Preliminary accuracy (PASS / total): {tally['PASS'] / total:.0%}")
#     print(f"Best case if all REVIEW turn out correct: {(tally['PASS'] + tally['REVIEW']) / total:.0%}")
#     print()
#     print("By difficulty:")
#     for diff, counts in by_difficulty.items():
#         t = sum(counts.values())
#         print(f"  {diff:14s} PASS {counts['PASS']:2d}  FAIL {counts['FAIL']:2d}  REVIEW {counts['REVIEW']:2d}  (of {t})")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Live Preliminary Accuracy Check
# ==================================================
# Self-contained: the 70 test queries + gold answers are embedded below (pulled
# straight from your graded blind_test_results CSV), so this needs nothing but
# a running Ollama + populated Chroma collection. Runs every query through the
# real pipeline (same blind()/hybrid_retrieve/rerank/generate_answer as
# run_csv_suite.py) and prints a tick or cross immediately after each one, so
# you can watch it live and get a number before your presentation.

# IMPORTANT — read before you quote this number to anyone:
# Grading here is an automated heuristic (checks whether the expected numbers
# and keywords show up in the model's answer text). It is NOT a substitute for
# the row-by-row human read you'd do against the CSV -- it WILL misgrade a few
# rows (a correct answer phrased differently than expected_answer, or a wrong
# answer that happens to contain the right numbers in passing). Treat the
# final score as a fast, approximate "are we in the right ballpark" gut-check
# for your slides, not a certified accuracy figure. Three-way grading:
#   PASS   - looks correct (or correctly refused a trap question)
#   FAIL   - looks wrong (or refused when it should have answered)
#   REVIEW - heuristic isn't confident either way -- eyeball this one yourself

# Usage:
#     python accuracy_check.py               # full run, live output
#     python accuracy_check.py --limit 10    # quick smoke test first
# """
# import argparse
# import json
# import os
# import re

# from GENERATION import load_corpus, build_bm25_index, hybrid_retrieve, rerank, generate_answer, Filters
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# CHECKPOINT_PATH = "./test_suite_checkpoint.json"

# # ---------------------------------------------------------------------------
# # The 70 questions, extracted from blind_test_results_20260922_123615.csv
# # ---------------------------------------------------------------------------
# QUESTIONS = [
#     {
#         "id": "E01",
#         "difficulty": "Easy",
#         "query": "What is the scope of IS 456:2000? As a site structural engineer, what type of concrete structures does this code generally cover?",
#         "gold_evidence": "Clause 1, p.11",
#         "expected_answer": "Scope; plain and reinforced concrete; special structures covered by respective standards"
#     },
#     {
#         "id": "E02",
#         "difficulty": "Easy",
#         "query": "What is meant by a plain concrete structure in IS 456?",
#         "gold_evidence": "Clause 1.1.1, p.11",
#         "expected_answer": "Reinforcement, if provided, is ignored for determining strength"
#     },
#     {
#         "id": "E03",
#         "difficulty": "Easy",
#         "query": "In a beam design calculation, what do the symbols d, D, Ast and Ld represent in IS 456?",
#         "gold_evidence": "Clause 4, pp.11–12",
#         "expected_answer": "Effective depth; overall depth; tension steel area; development length"
#     },
#     {
#         "id": "E04",
#         "difficulty": "Easy",
#         "query": "Which grades of ordinary Portland cement are explicitly listed in IS 456?",
#         "gold_evidence": "Clause 5.1, p.13",
#         "expected_answer": "33, 43 and 53 grade OPC"
#     },
#     {
#         "id": "E05",
#         "difficulty": "Easy",
#         "query": "What is the recommended nominal maximum coarse aggregate size for most concrete work?",
#         "gold_evidence": "Clause 5.3.3, p.14",
#         "expected_answer": "20 mm is suitable for most work"
#     },
#     {
#         "id": "E06",
#         "difficulty": "Easy",
#         "query": "What is the minimum pH value specified for water used in concrete?",
#         "gold_evidence": "Clause 5.4.2, p.14",
#         "expected_answer": "pH not less than 6"
#     },
#     {
#         "id": "E07",
#         "difficulty": "Easy",
#         "query": "what is the permissible chloride limit for reinforced concrete mixing water?",
#         "gold_evidence": "Table 1, p.15",
#         "expected_answer": "2000 mg/L as Cl"
#     },
#     {
#         "id": "E08",
#         "difficulty": "Easy",
#         "query": "What is the minimum concrete grade specified for reinforced concrete work in IS 456?",
#         "gold_evidence": "Foreword / Clause 6.1, pp.2,15",
#         "expected_answer": "M20"
#     },
#     {
#         "id": "E09",
#         "difficulty": "Easy",
#         "query": "As a site engineer, where does IS 456 say the required workability of concrete should come from?",
#         "gold_evidence": "Clause 7, p.16",
#         "expected_answer": "Mix proportions should provide adequate workability for placing conditions and available compaction method"
#     },
#     {
#         "id": "E10",
#         "difficulty": "Easy",
#         "query": "Why does IS 456 emphasize curing when discussing durability?",
#         "gold_evidence": "Clause 8.2.1, p.17",
#         "expected_answer": "Adequate curing avoids harmful effects of early moisture loss"
#     },
#     {
#         "id": "EM01",
#         "difficulty": "Easy-Medium",
#         "query": "For a reinforced concrete member exposed to 'severe' conditions, what nominal cover does it require?",
#         "gold_evidence": "Table 16, p.47",
#         "expected_answer": "45 mm"
#     },
#     {
#         "id": "EM02",
#         "difficulty": "Easy-Medium",
#         "query": "For 'extreme' exposure, what nominal concrete cover is required?",
#         "gold_evidence": "Table 16, p.47",
#         "expected_answer": "75 mm"
#     },
#     {
#         "id": "EM03",
#         "difficulty": "Easy-Medium",
#         "query": "What are the five environmental exposure categories?",
#         "gold_evidence": "Table 3, p.18",
#         "expected_answer": "Mild, moderate, severe, very severe, extreme"
#     },
#     {
#         "id": "EM04",
#         "difficulty": "Easy-Medium",
#         "query": "For M30 concrete, what standard deviation is assumed?",
#         "gold_evidence": "Table 8, p.23",
#         "expected_answer": "Use the M30 row from Table 8"
#     },
#     {
#         "id": "EM05",
#         "difficulty": "Easy-Medium",
#         "query": "What are the permissible limits for organic and inorganic solids in mixing water",
#         "gold_evidence": "Table 1, p.15",
#         "expected_answer": "200 mg/L organic; 3000 mg/L inorganic"
#     },
#     {
#         "id": "EM06",
#         "difficulty": "Easy-Medium",
#         "query": "A site engineer asks why 20 mm aggregate is generally preferred. What restriction links aggregate size to member thickness and reinforcement congestion?",
#         "gold_evidence": "Clause 5.3.3, p.14",
#         "expected_answer": "Nominal maximum size not greater than one-fourth of minimum member thickness, subject to proper placing around reinforcement; smaller sizes may be needed for thin/congested sections"
#     },
#     {
#         "id": "EM07",
#         "difficulty": "Easy-Medium",
#         "query": "If I use deformed bars instead of plain bars, how does the design bond stress change? Also tell me which clause provides the base bond stresses.",
#         "gold_evidence": "Clause 26.2.1.1, p.43",
#         "expected_answer": "Deformed-bar values are 60% higher; base values are given for plain bars"
#     },
#     {
#         "id": "EM08",
#         "difficulty": "Easy-Medium",
#         "query": "A concrete mix uses two admixtures simultaneously. What does the code require before accepting that combination?",
#         "gold_evidence": "Clause 5.5.6, p.16",
#         "expected_answer": "Obtain data on interaction and ensure compatibility"
#     },
#     {
#         "id": "EM09",
#         "difficulty": "Easy-Medium",
#         "query": "What is the maximum cement content mentioned in the durability provisions, and what risks are cited if it is exceeded?",
#         "gold_evidence": "Clause 8.2.4.2, p.19",
#         "expected_answer": "450 kg/m³ excluding fly ash/GGBS; increased cracking/thermal cracking/alkali-silica-reaction risk unless specially considered"
#     },
#     {
#         "id": "EM10",
#         "difficulty": "Easy-Medium",
#         "query": "For a 2.0 m span-to-width ratio, which coefficient should I retrieve from Table 26 for an interior panel's positive mid-span moment?",
#         "gold_evidence": "Annex D, Table 26, p.91",
#         "expected_answer": "0.049 for the short-span coefficient row at l_y/l_x = 2.0"
#     },
#     {
#         "id": "M01",
#         "difficulty": "Medium",
#         "query": "If I know the required fire-resistance period and member type, how is the figure intended to be used?",
#         "gold_evidence": "Fig. 1, p.34",
#         "expected_answer": "It relates minimum member dimensions and cover to fire-resistance periods/member types"
#     },
#     {
#         "id": "M02",
#         "difficulty": "Medium",
#         "query": "where should the critical section for shear normally be located relative to the face of a support when the special condition of Clause 22.6.2.1 applies?",
#         "gold_evidence": "Fig. 2 + Clause 22.6.2.1, p.37",
#         "expected_answer": "A section less than d from the support may use the shear at distance d when the reaction introduces compression"
#     },
#     {
#         "id": "M03",
#         "difficulty": "Medium",
#         "query": "For a fixed dead-load-plus-imposed-load case, what coefficient is used for the support moment at an interior support?",
#         "gold_evidence": "Table 12, p.36",
#         "expected_answer": "1/12"
#     },
#     {
#         "id": "M04",
#         "difficulty": "Medium",
#         "query": "For an unfixed imposed load, what is the shear coefficient at an interior support?",
#         "gold_evidence": "Table 13, p.36",
#         "expected_answer": "0.6"
#     },
#     {
#         "id": "M05",
#         "difficulty": "Medium",
#         "query": "What is the purpose of 'Modification Factor for Tension Reinforcement', in beam/slab deflection control?",
#         "gold_evidence": "Fig. 4, p.38",
#         "expected_answer": "It provides the modification factor used with the basic span-to-effective-depth relationship for tension reinforcement"
#     },
#     {
#         "id": "M06",
#         "difficulty": "Medium",
#         "query": "What does Fig. 5 provide, and when would I use it in a flexural-member deflection check?",
#         "gold_evidence": "Fig. 5, p.39",
#         "expected_answer": "Modification factor for compression reinforcement in deflection control"
#     },
#     {
#         "id": "M07",
#         "difficulty": "Medium",
#         "query": "What does Fig. 6 show for flanged beams?",
#         "gold_evidence": "Fig. 6, p.39",
#         "expected_answer": "Reduction factors for span-to-effective-depth ratios of flanged beams"
#     },
#     {
#         "id": "M08",
#         "difficulty": "Medium",
#         "query": "A beam is 800 mm deep. Does IS 456 require side-face reinforcement, and what minimum percentage is specified?",
#         "gold_evidence": "Clause 26.5.1.3, p.47",
#         "expected_answer": "Yes; when web depth exceeds 750 mm, total side-face reinforcement ≥0.1% of web area, distributed on two faces"
#     },
#     {
#         "id": "M09",
#         "difficulty": "Medium",
#         "query": "What is the maximum spacing of vertical shear stirrups specified?",
#         "gold_evidence": "Clause 26.5.1.5, p.47",
#         "expected_answer": "0.75d"
#     },
#     {
#         "id": "M10",
#         "difficulty": "Medium",
#         "query": "A beam has 16 mm bars in mild exposure. What reduction in nominal cover is permitted?",
#         "gold_evidence": "Table 16 Note 1, p.47",
#         "expected_answer": "5 mm reduction for main reinforcement up to 12 mm diameter only; therefore 16 mm does not qualify"
#     },
#     {
#         "id": "MH01",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete with plain tension bars, what is the design bond stress? If the bars are deformed bars conforming to IS 1786, what value should be used?",
#         "gold_evidence": "Clause 26.2.1.1, p.43",
#         "expected_answer": "Plain bar: 1.4 N/mm²; deformed-bar value increased by 60% = 2.24 N/mm²"
#     },
#     {
#         "id": "MH02",
#         "difficulty": "Medium-Hard",
#         "query": "A beam uses three bundled bars in contact. By what percentage should the development length of each bar be increased?",
#         "gold_evidence": "Clause 26.2.1.2, p.43",
#         "expected_answer": "20%"
#     },
#     {
#         "id": "MH03",
#         "difficulty": "Medium-Hard",
#         "query": "A plain tension bar has a 45° bend. What anchorage value can that bend provide, and what is the maximum allowed anchorage value from bends?",
#         "gold_evidence": "Clause 26.2.2.1, p.43",
#         "expected_answer": "4φ per 45° bend, subject to maximum 16φ"
#     },
#     {
#         "id": "MH04",
#         "difficulty": "Medium-Hard",
#         "query": "For a secondary stirrup bent through 135°, what continuation beyond the bend is required for complete development?",
#         "gold_evidence": "Clause 26.2.2.4, p.43",
#         "expected_answer": "At least six bar diameters"
#     },
#     {
#         "id": "MH05",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete and a longitudinal tension steel ratio of 1.0%, what is the design shear strength τc?",
#         "gold_evidence": "Table 19, p.73",
#         "expected_answer": "0.64 N/mm²"
#     },
#     {
#         "id": "MH06",
#         "difficulty": "Medium-Hard",
#         "query": "For M25 concrete at a steel percentage of 1.0%, compare τc from Table 19 with the maximum shear stress τc,max from Table 20. What design check does the code imply?",
#         "gold_evidence": "Tables 19–20, pp.73–74",
#         "expected_answer": "τc = 0.64 N/mm²; τc,max must be checked separately from Table 20; nominal shear stress must not exceed τc,max"
#     },
#     {
#         "id": "MH07",
#         "difficulty": "Medium-Hard",
#         "query": "If a section is within 2d of a support, what enhancement to shear strength can be considered, and what condition must hold at the support face?",
#         "gold_evidence": "Clause 40.5.1, p.74",
#         "expected_answer": "Enhanced shear strength may be considered; shear stress at the support face must remain below Table 20 limits"
#     },
#     {
#         "id": "MH08",
#         "difficulty": "Medium-Hard",
#         "query": "For a beam subjected to shear V and torsional moment T, how does IS 456 define equivalent shear for torsion design?",
#         "gold_evidence": "Clause 41.3.1, p.75",
#         "expected_answer": "Ve = Vu + 1.6T/b"
#     },
#     {
#         "id": "MH09",
#         "difficulty": "Medium-Hard",
#         "query": "For limit-state collapse, what partial safety factors are specified for concrete and steel material strengths?",
#         "gold_evidence": "Clause 36.4.2.1, p.68",
#         "expected_answer": "1.5 for concrete and 1.15 for steel"
#     },
#     {
#         "id": "MH10",
#         "difficulty": "Medium-Hard",
#         "query": "For the DL + IL + WL load combination, what partial safety factors does Table 18 specify for DL, IL and WL at the collapse limit state, and what are the corresponding serviceability values?",
#         "gold_evidence": "Table 18, p.68",
#         "expected_answer": "For DL+IL+WL at collapse: γf = 1.2 for DL, 1.2 for IL and 0.8 for WL; for serviceability: 1.0 for DL, 0.8 for IL and 0.8 for WL."
#     },
#     {
#         "id": "H01",
#         "difficulty": "Hard",
#         "query": "explain the assumed concrete stress-strain relationship used for limit-state flexural design. What are the key strain/stress values visible in the figure?",
#         "gold_evidence": "Fig. 21 + Clause 38.1, p.69",
#         "expected_answer": "Parabolic/rectangular idealization; maximum concrete compression strain 0.0035; design compression stress based on 0.67fck/γm"
#     },
#     {
#         "id": "H02",
#         "difficulty": "Hard",
#         "query": "what do 0.36fck xu and 0.42xu represent in the concrete stress block?",
#         "gold_evidence": "Fig. 22 + Clause 38.1, p.69",
#         "expected_answer": "Design stress-block intensity/resultant-related parameters and depth of compressive force from compression face"
#     },
#     {
#         "id": "H03",
#         "difficulty": "Hard",
#         "query": "For reinforcement stress-strain behaviour, which figure does IS 456 refer to and what partial safety factor is used for steel in flexural limit-state design?",
#         "gold_evidence": "Fig. 23 + Clause 38.1, p.69",
#         "expected_answer": "Fig. 23; γm = 1.15 for steel"
#     },
#     {
#         "id": "H04",
#         "difficulty": "Hard",
#         "query": "A structural engineer wants to use a core test to assess doubtful concrete. What acceptance percentages given for the average equivalent cube strength and individual core strength?",
#         "gold_evidence": "Clause 17.4.3, p.31",
#         "expected_answer": "Average equivalent cube strength ≥85% of specified cube strength; no individual core <75%"
#     },
#     {
#         "id": "H05",
#         "difficulty": "Hard",
#         "query": "specifies a load test on a flexural member. What load is applied, for how long, and what recovery is expected after removing the imposed load?",
#         "gold_evidence": "Clause 17.6.2–17.6.3, p.31",
#         "expected_answer": "Dead load + 1.25× imposed load for 24 h; at least 75% recovery within 24 h after imposed load removal"
#     },
#     {
#         "id": "H06",
#         "difficulty": "Hard",
#         "query": "A flat slab has an opening near a column. How should the critical shear perimeter be treated?",
#         "gold_evidence": "Fig. 15 + Clause 31.6, p.59",
#         "expected_answer": "The critical section is modified depending on opening location; certain opening portions may be treated/subtracted or as a free edge"
#     },
#     {
#         "id": "H07",
#         "difficulty": "Hard",
#         "query": "what geometric distance is used to locate the critical section around a column/drop for flat-slab shear?",
#         "gold_evidence": "Fig. 13 + Clause 31.6, p.58",
#         "expected_answer": "d/2 offsets from the support/column face are shown"
#     },
#     {
#         "id": "H08",
#         "difficulty": "Hard",
#         "query": "What does Fig. 14 illustrate for flat slabs with free edges or free corners?",
#         "gold_evidence": "Fig. 14 + Clause 31.6, p.59",
#         "expected_answer": "How free edges/corners affect the critical shear section"
#     },
#     {
#         "id": "H09",
#         "difficulty": "Hard",
#         "query": "How would you use Fig. 26 together with Table 28 to estimate effective length of a column in a framed structure?",
#         "gold_evidence": "Annex E, Fig. 26 + Table 28, pp.92–94",
#         "expected_answer": "Use restraint/sway condition and end restraints to obtain effective-length ratio; Table 28 gives idealized end-condition values"
#     },
#     {
#         "id": "H10",
#         "difficulty": "Hard",
#         "query": "What does Fig. 27 represent, and why is it relevant when the frame has lateral sway?",
#         "gold_evidence": "Annex E, Fig. 27, p.93",
#         "expected_answer": "Effective-length ratios for a column in a frame without restraint against sway"
#     },
#     {
#         "id": "A01",
#         "difficulty": "Advanced",
#         "query": "I am designing a 250 mm thick RC slab in a very severe exposure condition. What durability information do I need to retrieve before fixing concrete grade, water-cement ratio, minimum cement content and cover?",
#         "gold_evidence": "Tables 3, 5, 16, p.18 and pp.20,47",
#         "expected_answer": "Exposure category must be mapped to durability requirements across concrete quality and cover tables"
#     },
#     {
#         "id": "A02",
#         "difficulty": "Advanced",
#         "query": "A beam is heavily reinforced with small clear spacing. How should I select nominal maximum aggregate size?",
#         "gold_evidence": "Clause 5.3.3.1, p.14",
#         "expected_answer": "Usually 5 mm less than minimum clear distance between main bars or 5 mm less than minimum cover, whichever is smaller"
#     },
#     {
#         "id": "A03",
#         "difficulty": "Advanced",
#         "query": "A retaining structure uses concrete exposed to sulphates. Which table should I consult, and what inputs from the exposure/environment should I identify before selecting the concrete mix?",
#         "gold_evidence": "Table 4 + Clause 8.2.4, p.19",
#         "expected_answer": "Table 4; sulphate exposure condition, sulphate concentration/type, cement type, water-cement ratio and concrete requirements"
#     },
#     {
#         "id": "A04",
#         "difficulty": "Advanced",
#         "query": "A beam carries torsion plus shear. Explain the retrieval chain from Clause 41.3.1 to Tables 19 and 20 and then to Clause 41.4 for reinforcement design.",
#         "gold_evidence": "Clauses 41.3–41.4 + Tables 19–20, pp.75–76",
#         "expected_answer": "Compute equivalent shear; compare with τc and τc,max; if required provide longitudinal and transverse torsion reinforcement"
#     },
#     {
#         "id": "A05",
#         "difficulty": "Advanced",
#         "query": "A continuous beam designer wants to redistribute moments. What are the main restrictions, including the 70% condition and the maximum reduction rule?",
#         "gold_evidence": "Clause 37.1.1, pp.68–69",
#         "expected_answer": "Equilibrium maintained; resistance not below 70% of elastic moment; reduction constraints including not more than 30% of numerically largest elastic moment, subject to stated conditions"
#     },
#     {
#         "id": "A06",
#         "difficulty": "Advanced",
#         "query": "For a column frame, how do I determine whether the column can be treated as no-sway using the stability index Q?",
#         "gold_evidence": "Annex E.2, p.92",
#         "expected_answer": "Q ≤ 0.04 → no-sway; otherwise sway"
#     },
#     {
#         "id": "A07",
#         "difficulty": "Advanced",
#         "query": "A column has an effective-length ratio above the threshold mentioned in Annex E. How does that affect the choice between the simplified and more exact approach?",
#         "gold_evidence": "Annex E, pp.92–94",
#         "expected_answer": "Use the appropriate effective-length relationships/figures; the source distinguishes normal idealized usage and more exact analysis"
#     },
#     {
#         "id": "A08",
#         "difficulty": "Advanced",
#         "query": "If I need to calculate design surface crack width, what variables does Annex F require, and what is the main crack-width expression?",
#         "gold_evidence": "Annex F, p.95",
#         "expected_answer": "acr, cmin, εm, h, x and the specified expression for design surface crack width"
#     },
#     {
#         "id": "A09",
#         "difficulty": "Advanced",
#         "query": "How does Annex C define total deflection in terms of short-term and long-term components, and what long-term effects are separately considered?",
#         "gold_evidence": "Annex C, pp.88–89",
#         "expected_answer": "Total deflection = short-term + long-term; long-term components include shrinkage and creep"
#     },
#     {
#         "id": "A10",
#         "difficulty": "Advanced",
#         "query": "For a rectangular slab supported on four sides, how do I choose the bending-moment coefficients from Annex D based on panel type and l_y/l_x?",
#         "gold_evidence": "Annex D, Tables 26–27, pp.90–91",
#         "expected_answer": "Identify panel/support condition, determine aspect ratio, then retrieve short- and long-span coefficients from the appropriate table"
#     },
#     {
#         "id": "X01",
#         "difficulty": "Advanced",
#         "query": "What is the minimum concrete grade for prestressed concrete according to IS 456:2000?",
#         "gold_evidence": "Not directly supported; scope excludes special systems not covered by this code",
#         "expected_answer": "Should state that the uploaded source does not provide a direct prestressed-concrete minimum-grade rule rather than inventing one"
#     },
#     {
#         "id": "X02",
#         "difficulty": "Advanced",
#         "query": "What is the seismic zone factor Z for Delhi according to this document?",
#         "gold_evidence": "Not supported by this document; earthquake forces are referenced but seismic zoning values are not given here",
#         "expected_answer": "Should not fabricate a value"
#     },
#     {
#         "id": "X03",
#         "difficulty": "Advanced",
#         "query": "what is the permissible shear stress for M10 concrete?",
#         "gold_evidence": "Table 19 starts at M15; M10 is not a listed grade",
#         "expected_answer": "Should identify that M10 is absent from Table 19 instead of extrapolating"
#     },
#     {
#         "id": "X04",
#         "difficulty": "Advanced",
#         "query": "If a flat slab has shear stress exactly above 1.5τc, can I simply increase the reinforcement indefinitely?",
#         "gold_evidence": "Clause 31.6.3.2, p.59",
#         "expected_answer": "No; if shear stress exceeds 1.5τc, the flat slab shall be redesigned"
#     },
#     {
#         "id": "X05",
#         "difficulty": "Advanced",
#         "query": "Can I use sea water for reinforced concrete because the code permits it for plain concrete?",
#         "gold_evidence": "Clause 5.4.3, p.14",
#         "expected_answer": "No; the stated unavoidable-use exception is for plain concrete with no embedded steel"
#     },
#     {
#         "id": "X06",
#         "difficulty": "Advanced",
#         "query": "The note says '20 mm aggregate is suitable for most work', so is 20 mm always permitted in a heavily reinforced beam?",
#         "gold_evidence": "Clause 5.3.3.1, p.14",
#         "expected_answer": "No; heavily reinforced members may require a smaller nominal maximum aggregate size based on clear spacing/cover"
#     },
#     {
#         "id": "X07",
#         "difficulty": "Advanced",
#         "query": "Does a 45° bend always provide 4φ anchorage even if that exceeds the maximum bend anchorage permitted?",
#         "gold_evidence": "Clause 26.2.2.1, p.43",
#         "expected_answer": "No; 4φ per 45° bend is subject to the 16φ maximum"
#     },
#     {
#         "id": "X08",
#         "difficulty": "Advanced",
#         "query": "If Q = 0.04 exactly, should the column be classified as sway?",
#         "gold_evidence": "Annex E.2, p.92",
#         "expected_answer": "No; Q ≤ 0.04 is treated as no-sway"
#     },
#     {
#         "id": "X09",
#         "difficulty": "Advanced",
#         "query": "Does IS 456 require the use of only design-mix concrete and prohibit nominal mixes?",
#         "gold_evidence": "Clause 9, pp.21–22",
#         "expected_answer": "No; the contents explicitly include both design mix and nominal mix concrete"
#     },
#     {
#         "id": "X10",
#         "difficulty": "Advanced",
#         "query": "Can I use a special concrete structural system not covered by IS 456 without any additional evidence?",
#         "gold_evidence": "Clause 19, p.3",
#         "expected_answer": "The code permits special systems not covered only when satisfactory evidence of adequacy and safety is produced by analysis, test, or both"
#     }
# ]

# # ---------------------------------------------------------------------------
# # Blinding — identical to run_csv_suite.py, kept here so this file has no
# # dependency on it and stays comparable to your last graded run.
# # ---------------------------------------------------------------------------
# _RANGE = r"(\s*(?:,|and|to|\u2013|-)\s*\d+[A-Za-z]?)*"
# BLIND_PATTERNS = [
#     (re.compile(rf"\bTables?\s+\d+[A-Za-z]?{_RANGE}\b", re.IGNORECASE), "the relevant table"),
#     (re.compile(rf"\bFigs?\.?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bFigures?\s*\d+{_RANGE}\b", re.IGNORECASE), "the relevant figure"),
#     (re.compile(rf"\bClauses?\s+\d+(?:\.\d+)*{_RANGE}\b", re.IGNORECASE), "the relevant clause"),
#     (re.compile(r"\bAnnex(?:es)?\s+[A-Z](\s*(?:,|and)\s*[A-Z])*\b", re.IGNORECASE), "the relevant annex"),
#     (re.compile(r"\bSections?\s+\d+(?:\.\d+)*\b", re.IGNORECASE), "the relevant section"),
# ]


# def blind(text: str) -> str:
#     for pattern, replacement in BLIND_PATTERNS:
#         text = pattern.sub(replacement, text)
#     return re.sub(r"\s+", " ", text).strip()


# # ---------------------------------------------------------------------------
# # Grading heuristic
# # ---------------------------------------------------------------------------
# # Fix #9: broadened after checking this against the actual run -- the
# # original list missed X02 ("should not fabricate a value") and X03
# # ("absent from Table 19 ... instead of extrapolating"), both of which the
# # chatbot handled correctly (refused) but got graded FAIL anyway. Verified
# # by re-running this exact list against every expected_answer in the CSV.
# TRAP_MARKERS = ("not supported", "not directly supported", "not listed", "not given",
#                 "not covered", "not specified", "excludes", "does not",
#                 "should not fabricate", "should not invent", "absent from",
#                 "instead of extrapolating", "instead of inventing", "rather than inventing")
# STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "into", "value",
#              "values", "when", "which", "what", "where", "condition", "conditions",
#              "according", "then", "than", "also", "used", "use", "using"}
# NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# def is_refusal(answer: str) -> bool:
#     return answer.strip().lower().startswith("i could not find")


# def grade(q: dict, answer: str) -> str:
#     expected = q["expected_answer"]
#     expected_lower = expected.lower()
#     is_trap = any(marker in expected_lower for marker in TRAP_MARKERS)
#     refused = is_refusal(answer)

#     if is_trap:
#         return "PASS" if refused else "FAIL"
#     if refused:
#         return "FAIL"

#     answer_lower = answer.lower()
#     expected_numbers = NUMBER_RE.findall(expected)
#     expected_keywords = [w.strip(".,;:()").lower() for w in re.split(r"\s+", expected)
#                           if len(w) >= 4 and w.strip(".,;:()").lower() not in STOPWORDS]
#     expected_keywords = list(dict.fromkeys(expected_keywords))  # dedupe, keep order

#     num_score = None
#     if expected_numbers:
#         hits = sum(1 for n in expected_numbers if n in answer)
#         num_score = hits / len(expected_numbers)

#     kw_score = None
#     if expected_keywords:
#         hits = sum(1 for w in expected_keywords if w in answer_lower)
#         kw_score = hits / len(expected_keywords)

#     if num_score is not None and kw_score is not None:
#         score = 0.7 * num_score + 0.3 * kw_score
#     elif num_score is not None:
#         score = num_score
#     elif kw_score is not None:
#         score = kw_score
#     else:
#         return "REVIEW"

#     if score >= 0.6:
#         return "PASS"
#     if score < 0.3:
#         return "FAIL"
#     return "REVIEW"


# MARK = {"PASS": "\033[92m PASS\033[0m", "FAIL": "\033[91m FAIL\033[0m", "REVIEW": "\033[93m REVIEW\033[0m"}


# def load_checkpoint() -> dict:
#     """Returns {question_id: {"result": ..., "answer": ...}} already completed."""
#     if not os.path.exists(CHECKPOINT_PATH):
#         return {}
#     try:
#         with open(CHECKPOINT_PATH) as f:
#             return json.load(f)
#     except (json.JSONDecodeError, OSError):
#         print(f"  [warn] {CHECKPOINT_PATH} is unreadable/corrupt -- ignoring it and starting fresh.")
#         return {}


# def save_checkpoint(done: dict) -> None:
#     # Write to a temp file and rename, so a crash mid-write never corrupts the
#     # checkpoint already on disk (what you'd lose otherwise).
#     tmp_path = CHECKPOINT_PATH + ".tmp"
#     with open(tmp_path, "w") as f:
#         json.dump(done, f, indent=2)
#     os.replace(tmp_path, CHECKPOINT_PATH)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
#     parser.add_argument("--fresh", action="store_true",
#                          help="ignore any existing checkpoint and re-run every question from scratch")
#     args = parser.parse_args()

#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)
#     corpus = load_corpus(collection)
#     if not corpus["ids"]:
#         print("Collection is empty — run ingest.py first.")
#         return
#     bm25_index = build_bm25_index(corpus)

#     questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS

#     done = {} if args.fresh else load_checkpoint()
#     if args.fresh and os.path.exists(CHECKPOINT_PATH):
#         os.remove(CHECKPOINT_PATH)
#     already = sum(1 for q in questions if q["id"] in done)
#     if already:
#         print(f"Resuming from checkpoint: {already}/{len(questions)} question(s) already done "
#               f"(delete {CHECKPOINT_PATH}, or pass --fresh, to start over).")
#     print(f"Loaded {len(corpus['ids'])} chunks. Running {len(questions)} question(s) live...\n")

#     tally = {"PASS": 0, "FAIL": 0, "REVIEW": 0}
#     by_difficulty = {}

#     for i, q in enumerate(questions, 1):
#         diff = q["difficulty"]
#         by_difficulty.setdefault(diff, {"PASS": 0, "FAIL": 0, "REVIEW": 0})

#         if q["id"] in done:
#             result = done[q["id"]]["result"]
#             answer = done[q["id"]]["answer"]
#             tally[result] += 1
#             by_difficulty[diff][result] += 1
#             print(f"[{i}/{len(questions)}] {q['id']} ({diff}) — {MARK[result]} (from checkpoint)")
#             continue

#         blind_q = blind(q["query"])
#         candidates = hybrid_retrieve(blind_q, corpus, bm25_index, collection, Filters())
#         reranked = rerank(blind_q, candidates)
#         answer = generate_answer(blind_q, reranked)

#         result = grade(q, answer)
#         tally[result] += 1
#         by_difficulty[diff][result] += 1

#         done[q["id"]] = {"result": result, "answer": answer}
#         save_checkpoint(done)

#         print(f"[{i}/{len(questions)}] {q['id']} ({diff}) — {MARK[result]}")
#         print(f"    Q: {q['query']}")
#         print(f"    expected : {q['expected_answer']}")
#         shown = answer[:200] + ("..." if len(answer) > 200 else "")
#         print(f"    actual   : {shown}")
#         print()

#     total = sum(tally.values())
#     print("=" * 60)
#     print(f"PASS: {tally['PASS']}   FAIL: {tally['FAIL']}   REVIEW: {tally['REVIEW']}   (of {total})")
#     print(f"Preliminary accuracy (PASS / total): {tally['PASS'] / total:.0%}")
#     print(f"Best case if all REVIEW turn out correct: {(tally['PASS'] + tally['REVIEW']) / total:.0%}")
#     print()
#     print("By difficulty:")
#     for diff, counts in by_difficulty.items():
#         t = sum(counts.values())
#         print(f"  {diff:14s} PASS {counts['PASS']:2d}  FAIL {counts['FAIL']:2d}  REVIEW {counts['REVIEW']:2d}  (of {t})")


# if __name__ == "__main__":
#     main()
"""
IS-Code RAG — Live Preliminary Accuracy Check
==================================================
Self-contained: the 70 test queries + gold answers are embedded below (pulled
straight from your graded blind_test_results CSV), so this needs nothing but
a running Ollama + populated Chroma collection. Runs every query through the
real pipeline (hybrid_retrieve/rerank/generate_answer) and prints a tick or
cross immediately after each one, so you can watch it live and get a number
before your presentation.

NOT a blind test: queries are sent to the pipeline exactly as written below,
including any "Table N" / "Fig. N" / "Clause N" references they contain --
this measures retrieval power WITH those hints available, not without them.
(An earlier version of this script auto-stripped such references before
retrieval via a blind() step; that step has been removed on request.)

IMPORTANT — read before you quote this number to anyone:
Grading here is an automated heuristic (checks whether the expected numbers
and keywords show up in the model's answer text). It is NOT a substitute for
the row-by-row human read you'd do against the CSV -- it WILL misgrade a few
rows (a correct answer phrased differently than expected_answer, or a wrong
answer that happens to contain the right numbers in passing). Treat the
final score as a fast, approximate "are we in the right ballpark" gut-check
for your slides, not a certified accuracy figure. Three-way grading:
  PASS   - looks correct (or correctly refused a trap question)
  FAIL   - looks wrong (or refused when it should have answered)
  REVIEW - heuristic isn't confident either way -- eyeball this one yourself

Usage:
    python accuracy_check.py               # full run, live output
    python accuracy_check.py --limit 10    # quick smoke test first
"""
import argparse
import json
import os
import re

from GENERATION import load_corpus, build_bm25_index, hybrid_retrieve, rerank, generate_answer, Filters
import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"
CHECKPOINT_PATH = "./test_suite_checkpoint.json"

# ---------------------------------------------------------------------------
# The 70 questions, extracted from blind_test_results_20260922_123615.csv
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": "E01",
        "difficulty": "Easy",
        "query": "What is the scope of IS 456:2000? As a site structural engineer, what type of concrete structures does this code generally cover?",
        "gold_evidence": "Clause 1, p.11",
        "expected_answer": "Scope; plain and reinforced concrete; special structures covered by respective standards"
    },
    {
        "id": "E02",
        "difficulty": "Easy",
        "query": "What is meant by a plain concrete structure in IS 456?",
        "gold_evidence": "Clause 1.1.1, p.11",
        "expected_answer": "Reinforcement, if provided, is ignored for determining strength"
    },
    {
        "id": "E03",
        "difficulty": "Easy",
        "query": "In a beam design calculation, what do the symbols d, D, Ast and Ld represent in IS 456?",
        "gold_evidence": "Clause 4, pp.11–12",
        "expected_answer": "Effective depth; overall depth; tension steel area; development length"
    },
    {
        "id": "E04",
        "difficulty": "Easy",
        "query": "Which grades of ordinary Portland cement are explicitly listed in IS 456?",
        "gold_evidence": "Clause 5.1, p.13",
        "expected_answer": "33, 43 and 53 grade OPC"
    },
    {
        "id": "E05",
        "difficulty": "Easy",
        "query": "What is the recommended nominal maximum coarse aggregate size for most concrete work?",
        "gold_evidence": "Clause 5.3.3, p.14",
        "expected_answer": "20 mm is suitable for most work"
    },
    {
        "id": "E06",
        "difficulty": "Easy",
        "query": "What is the minimum pH value specified for water used in concrete?",
        "gold_evidence": "Clause 5.4.2, p.14",
        "expected_answer": "pH not less than 6"
    },
    {
        "id": "E07",
        "difficulty": "Easy",
        "query": "what is the permissible chloride limit for reinforced concrete mixing water?",
        "gold_evidence": "Table 1, p.15",
        "expected_answer": "500 mg/L as Cl"
    },
    {
        "id": "E08",
        "difficulty": "Easy",
        "query": "What is the minimum concrete grade specified for reinforced concrete work in IS 456?",
        "gold_evidence": "Foreword / Clause 6.1, pp.2,15",
        "expected_answer": "M20"
    },
    {
        "id": "E09",
        "difficulty": "Easy",
        "query": "As a site engineer, where does IS 456 say the required workability of concrete should come from?",
        "gold_evidence": "Clause 7, p.16",
        "expected_answer": "Mix proportions should provide adequate workability for placing conditions and available compaction method"
    },
    {
        "id": "E10",
        "difficulty": "Easy",
        "query": "Why does IS 456 emphasize curing when discussing durability?",
        "gold_evidence": "Clause 8.2.7, p.17 and Clause 13.5" ,
        "expected_answer": "Adequate curing avoids harmful effects of early moisture loss"
    },
    {
        "id": "EM01",
        "difficulty": "Easy-Medium",
        "query": "For a reinforced concrete member exposed to 'severe' conditions, what nominal cover does it require?",
        "gold_evidence": "Table 16, p.47",
        "expected_answer": "45 mm"
    },
    {
        "id": "EM02",
        "difficulty": "Easy-Medium",
        "query": "For 'extreme' exposure, what nominal concrete cover is required?",
        "gold_evidence": "Table 16, p.47",
        "expected_answer": "75 mm"
    },
    {
        "id": "EM03",
        "difficulty": "Easy-Medium",
        "query": "What are the five environmental exposure categories?",
        "gold_evidence": "Table 3, p.18",
        "expected_answer": "Mild, moderate, severe, very severe, extreme"
    },
    {
        "id": "EM04",
        "difficulty": "Easy-Medium",
        "query": "For M30 concrete, what standard deviation is assumed?",
        "gold_evidence": "Table 8, p.23",
        "expected_answer": "Use the M30 row from Table 8"
    },
    {
        "id": "EM05",
        "difficulty": "Easy-Medium",
        "query": "What are the permissible limits for organic and inorganic solids in mixing water",
        "gold_evidence": "Table 1, p.15",
        "expected_answer": "200 mg/L organic; 3000 mg/L inorganic"
    },
    {
        "id": "EM06",
        "difficulty": "Easy-Medium",
        "query": "A site engineer asks why 20 mm aggregate is generally preferred. What restriction links aggregate size to member thickness and reinforcement congestion?",
        "gold_evidence": "Clause 5.3.3, p.14",
        "expected_answer": "Nominal maximum size not greater than one-fourth of minimum member thickness, subject to proper placing around reinforcement; smaller sizes may be needed for thin/congested sections"
    },
    {
        "id": "EM07",
        "difficulty": "Easy-Medium",
        "query": "If I use deformed bars instead of plain bars, how does the design bond stress change? Also tell me which clause provides the base bond stresses.",
        "gold_evidence": "Clause 26.2.1.1, p.43",
        "expected_answer": "Deformed-bar values are 60% higher; base values are given for plain bars"
    },
    {
        "id": "EM08",
        "difficulty": "Easy-Medium",
        "query": "A concrete mix uses two admixtures simultaneously. What does the code require before accepting that combination?",
        "gold_evidence": "Clause 5.5.6, p.16",
        "expected_answer": "Obtain data on interaction and ensure compatibility"
    },
    {
        "id": "EM09",
        "difficulty": "Easy-Medium",
        "query": "What is the maximum cement content mentioned in the durability provisions, and what risks are cited if it is exceeded?",
        "gold_evidence": "Clause 8.2.4.2, p.19",
        "expected_answer": "450 kg/m³ excluding fly ash/GGBS; increased cracking/thermal cracking/alkali-silica-reaction risk unless specially considered"
    },
    {
        "id": "EM10",
        "difficulty": "Easy-Medium",
        "query": "For a 2.0 m span-to-width ratio, which coefficient should I retrieve from Table 26 for an interior panel's positive mid-span moment?",
        "gold_evidence": "Annex D, Table 26, p.91",
        "expected_answer": "0.049 for the short-span coefficient row at l_y/l_x = 2.0"
    },
    {
        "id": "M01",
        "difficulty": "Medium",
        "query": "If I know the required fire-resistance period and member type, how is the figure intended to be used?",
        "gold_evidence": "Fig. 1, p.34",
        "expected_answer": "It relates minimum member dimensions and cover to fire-resistance periods/member types"
    },
    {
        "id": "M02",
        "difficulty": "Medium",
        "query": "where should the critical section for shear normally be located relative to the face of a support when the special condition of Clause 22.6.2.1 applies?",
        "gold_evidence": "Fig. 2 + Clause 22.6.2.1, p.37",
        "expected_answer": "A section less than d from the support may use the shear at distance d when the reaction introduces compression"
    },
    {
        "id": "M03",
        "difficulty": "Medium",
        "query": "For a fixed dead-load-plus-imposed-load case, what coefficient is used for the support moment at an interior support?",
        "gold_evidence": "Table 12, p.36",
        "expected_answer": "1/12"
    },
    {
        "id": "M04",
        "difficulty": "Medium",
        "query": "For an unfixed imposed load, what is the shear coefficient at an interior support?",
        "gold_evidence": "Table 13, p.36",
        "expected_answer": "0.6"
    },
    {
        "id": "M05",
        "difficulty": "Medium",
        "query": "What is the purpose of 'Modification Factor for Tension Reinforcement', in beam/slab deflection control?",
        "gold_evidence": "Fig. 4, p.38",
        "expected_answer": "It provides the modification factor used with the basic span-to-effective-depth relationship for tension reinforcement"
    },
    {
        "id": "M06",
        "difficulty": "Medium",
        "query": "What does Fig. 5 provide, and when would I use it in a flexural-member deflection check?",
        "gold_evidence": "Fig. 5, p.39",
        "expected_answer": "Modification factor for compression reinforcement in deflection control"
    },
    {
        "id": "M07",
        "difficulty": "Medium",
        "query": "What does Fig. 6 show for flanged beams?",
        "gold_evidence": "Fig. 6, p.39",
        "expected_answer": "Reduction factors for span-to-effective-depth ratios of flanged beams"
    },
    {
        "id": "M08",
        "difficulty": "Medium",
        "query": "A beam is 800 mm deep. Does IS 456 require side-face reinforcement, and what minimum percentage is specified?",
        "gold_evidence": "Clause 26.5.1.3, p.47",
        "expected_answer": "Yes; when web depth exceeds 750 mm, total side-face reinforcement ≥0.1% of web area, distributed on two faces"
    },
    {
        "id": "M09",
        "difficulty": "Medium",
        "query": "What is the maximum spacing of vertical shear stirrups specified?",
        "gold_evidence": "Clause 26.5.1.5, p.47",
        "expected_answer": "0.75d"
    },
    {
        "id": "M10",
        "difficulty": "Medium",
        "query": "A beam has 16 mm bars in mild exposure. What reduction in nominal cover is permitted?",
        "gold_evidence": "Table 16 Note 1, p.47",
        "expected_answer": "5 mm reduction for main reinforcement up to 12 mm diameter only; therefore 16 mm does not qualify"
    },
    {
        "id": "MH01",
        "difficulty": "Medium-Hard",
        "query": "For M25 concrete with plain tension bars, what is the design bond stress? If the bars are deformed bars conforming to IS 1786, what value should be used?",
        "gold_evidence": "Clause 26.2.1.1, p.43",
        "expected_answer": "Plain bar: 1.4 N/mm²; deformed-bar value increased by 60% = 2.24 N/mm²"
    },
    {
        "id": "MH02",
        "difficulty": "Medium-Hard",
        "query": "A beam uses three bundled bars in contact. By what percentage should the development length of each bar be increased?",
        "gold_evidence": "Clause 26.2.1.2, p.43",
        "expected_answer": "20%"
    },
    {
        "id": "MH03",
        "difficulty": "Medium-Hard",
        "query": "A plain tension bar has a 45° bend. What anchorage value can that bend provide, and what is the maximum allowed anchorage value from bends?",
        "gold_evidence": "Clause 26.2.2.1, p.43",
        "expected_answer": "4φ per 45° bend, subject to maximum 16φ"
    },
    {
        "id": "MH04",
        "difficulty": "Medium-Hard",
        "query": "For a secondary stirrup bent through 135°, what continuation beyond the bend is required for complete development?",
        "gold_evidence": "Clause 26.2.2.4, p.43",
        "expected_answer": "At least six bar diameters"
    },
    {
        "id": "MH05",
        "difficulty": "Medium-Hard",
        "query": "For M25 concrete and a longitudinal tension steel ratio of 1.0%, what is the design shear strength τc?",
        "gold_evidence": "Table 19, p.73",
        "expected_answer": "0.64 N/mm²"
    },
    {
        "id": "MH06",
        "difficulty": "Medium-Hard",
        "query": "For M25 concrete at a steel percentage of 1.0%, compare τc from Table 19 with the maximum shear stress τc,max from Table 20. What design check does the code imply?",
        "gold_evidence": "Tables 19–20, pp.73–74",
        "expected_answer": "τc = 0.64 N/mm²; τc,max must be checked separately from Table 20; nominal shear stress must not exceed τc,max"
    },
    {
        "id": "MH07",
        "difficulty": "Medium-Hard",
        "query": "If a section is within 2d of a support, what enhancement to shear strength can be considered, and what condition must hold at the support face?",
        "gold_evidence": "Clause 40.5.1, p.74",
        "expected_answer": "Enhanced shear strength may be considered; shear stress at the support face must remain below Table 20 limits"
    },
    {
        "id": "MH08",
        "difficulty": "Medium-Hard",
        "query": "For a beam subjected to shear V and torsional moment T, how does IS 456 define equivalent shear for torsion design?",
        "gold_evidence": "Clause 41.3.1, p.75",
        "expected_answer": "Ve = Vu + 1.6T/b"
    },
    {
        "id": "MH09",
        "difficulty": "Medium-Hard",
        "query": "For limit-state collapse, what partial safety factors are specified for concrete and steel material strengths?",
        "gold_evidence": "Clause 36.4.2.1, p.68",
        "expected_answer": "1.5 for concrete and 1.15 for steel"
    },
    {
        "id": "MH10",
        "difficulty": "Medium-Hard",
        "query": "For the DL + IL + WL load combination, what partial safety factors does Table 18 specify for DL, IL and WL at the collapse limit state, and what are the corresponding serviceability values?",
        "gold_evidence": "Table 18, p.68",
        "expected_answer": "For DL+IL+WL at collapse: γf = 1.2 for DL, 1.2 for IL and 0.8 for WL; for serviceability: 1.0 for DL, 0.8 for IL and 0.8 for WL."
    },
    {
        "id": "H01",
        "difficulty": "Hard",
        "query": "explain the assumed concrete stress-strain relationship used for limit-state flexural design. What are the key strain/stress values visible in the figure?",
        "gold_evidence": "Fig. 21 + Clause 38.1, p.69",
        "expected_answer": "Parabolic/rectangular idealization; maximum concrete compression strain 0.0035; design compression stress based on 0.67fck/γm"
    },
    {
        "id": "H02",
        "difficulty": "Hard",
        "query": "what do 0.36fck xu and 0.42xu represent in the concrete stress block?",
        "gold_evidence": "Fig. 22 + Clause 38.1, p.69",
        "expected_answer": "Design stress-block intensity/resultant-related parameters and depth of compressive force from compression face"
    },
    {
        "id": "H03",
        "difficulty": "Hard",
        "query": "For reinforcement stress-strain behaviour, which figure does IS 456 refer to and what partial safety factor is used for steel in flexural limit-state design?",
        "gold_evidence": "Fig. 23 + Clause 38.1, p.69",
        "expected_answer": "Fig. 23; γm = 1.15 for steel"
    },
    {
        "id": "H04",
        "difficulty": "Hard",
        "query": "A structural engineer wants to use a core test to assess doubtful concrete. What acceptance percentages given for the average equivalent cube strength and individual core strength?",
        "gold_evidence": "Clause 17.4.3, p.31",
        "expected_answer": "Average equivalent cube strength ≥85% of specified cube strength; no individual core <75%"
    },
    {
        "id": "H05",
        "difficulty": "Hard",
        "query": "specifies a load test on a flexural member. What load is applied, for how long, and what recovery is expected after removing the imposed load?",
        "gold_evidence": "Clause 17.6.2–17.6.3, p.31",
        "expected_answer": "Dead load + 1.25× imposed load for 24 h; at least 75% recovery within 24 h after imposed load removal"
    },
    {
        "id": "H06",
        "difficulty": "Hard",
        "query": "A flat slab has an opening near a column. How should the critical shear perimeter be treated?",
        "gold_evidence": "Fig. 15 + Clause 31.6, p.59",
        "expected_answer": "The critical section is modified depending on opening location; certain opening portions may be treated/subtracted or as a free edge"
    },
    {
        "id": "H07",
        "difficulty": "Hard",
        "query": "what geometric distance is used to locate the critical section around a column/drop for flat-slab shear?",
        "gold_evidence": "Fig. 13 + Clause 31.6, p.58",
        "expected_answer": "d/2 offsets from the support/column face are shown"
    },
    {
        "id": "H08",
        "difficulty": "Hard",
        "query": "What does Fig. 14 illustrate for flat slabs with free edges or free corners?",
        "gold_evidence": "Fig. 14 + Clause 31.6, p.59",
        "expected_answer": "How free edges/corners affect the critical shear section"
    },
    {
        "id": "H09",
        "difficulty": "Hard",
        "query": "How would you use Fig. 26 together with Table 28 to estimate effective length of a column in a framed structure?",
        "gold_evidence": "Annex E, Fig. 26 + Table 28, pp.92–94",
        "expected_answer": "Use restraint/sway condition and end restraints to obtain effective-length ratio; Table 28 gives idealized end-condition values"
    },
    {
        "id": "H10",
        "difficulty": "Hard",
        "query": "What does Fig. 27 represent, and why is it relevant when the frame has lateral sway?",
        "gold_evidence": "Annex E, Fig. 27, p.93",
        "expected_answer": "Effective-length ratios for a column in a frame without restraint against sway"
    },
    {
        "id": "A01",
        "difficulty": "Advanced",
        "query": "I am designing a 250 mm thick RC slab in a very severe exposure condition. What durability information do I need to retrieve before fixing concrete grade, water-cement ratio, minimum cement content and cover?",
        "gold_evidence": "Tables 3, 5, 16, p.18 and pp.20,47",
        "expected_answer": "Exposure category must be mapped to durability requirements across concrete quality and cover tables"
    },
    {
        "id": "A02",
        "difficulty": "Advanced",
        "query": "A beam is heavily reinforced with small clear spacing. How should I select nominal maximum aggregate size?",
        "gold_evidence": "Clause 5.3.3.1, p.14",
        "expected_answer": "Usually 5 mm less than minimum clear distance between main bars or 5 mm less than minimum cover, whichever is smaller"
    },
    {
        "id": "A03",
        "difficulty": "Advanced",
        "query": "A retaining structure uses concrete exposed to sulphates. Which table should I consult, and what inputs from the exposure/environment should I identify before selecting the concrete mix?",
        "gold_evidence": "Table 4 + Clause 8.2.4, p.19",
        "expected_answer": "Table 4; sulphate exposure condition, sulphate concentration/type, cement type, water-cement ratio and concrete requirements"
    },
    {
        "id": "A04",
        "difficulty": "Advanced",
        "query": "A beam carries torsion plus shear. Explain the retrieval chain from Clause 41.3.1 to Tables 19 and 20 and then to Clause 41.4 for reinforcement design.",
        "gold_evidence": "Clauses 41.3–41.4 + Tables 19–20, pp.75–76",
        "expected_answer": "Compute equivalent shear; compare with τc and τc,max; if required provide longitudinal and transverse torsion reinforcement"
    },
    {
        "id": "A05",
        "difficulty": "Advanced",
        "query": "A continuous beam designer wants to redistribute moments. What are the main restrictions, including the 70% condition and the maximum reduction rule?",
        "gold_evidence": "Clause 37.1.1, pp.68–69",
        "expected_answer": "Equilibrium maintained; resistance not below 70% of elastic moment; reduction constraints including not more than 30% of numerically largest elastic moment, subject to stated conditions"
    },
    {
        "id": "A06",
        "difficulty": "Advanced",
        "query": "For a column frame, how do I determine whether the column can be treated as no-sway using the stability index Q?",
        "gold_evidence": "Annex E.2, p.92",
        "expected_answer": "Q ≤ 0.04 → no-sway; otherwise sway"
    },
    {
        "id": "A07",
        "difficulty": "Advanced",
        "query": "A column has an effective-length ratio above the threshold mentioned in Annex E. How does that affect the choice between the simplified and more exact approach?",
        "gold_evidence": "Annex E, pp.92–94",
        "expected_answer": "Use the appropriate effective-length relationships/figures; the source distinguishes normal idealized usage and more exact analysis"
    },
    {
        "id": "A08",
        "difficulty": "Advanced",
        "query": "If I need to calculate design surface crack width, what variables does Annex F require, and what is the main crack-width expression?",
        "gold_evidence": "Annex F, p.95",
        "expected_answer": "acr, cmin, εm, h, x and the specified expression for design surface crack width"
    },
    {
        "id": "A09",
        "difficulty": "Advanced",
        "query": "How does Annex C define total deflection in terms of short-term and long-term components, and what long-term effects are separately considered?",
        "gold_evidence": "Annex C, pp.88–89",
        "expected_answer": "Total deflection = short-term + long-term; long-term components include shrinkage and creep"
    },
    {
        "id": "A10",
        "difficulty": "Advanced",
        "query": "For a rectangular slab supported on four sides, how do I choose the bending-moment coefficients from Annex D based on panel type and l_y/l_x?",
        "gold_evidence": "Annex D, Tables 26–27, pp.90–91",
        "expected_answer": "Identify panel/support condition, determine aspect ratio, then retrieve short- and long-span coefficients from the appropriate table"
    },
    {
        "id": "X01",
        "difficulty": "Advanced",
        "query": "What is the minimum concrete grade for prestressed concrete according to IS 456:2000?",
        "gold_evidence": "Not directly supported; scope excludes special systems not covered by this code",
        "expected_answer": "Should state that the uploaded source does not provide a direct prestressed-concrete minimum-grade rule rather than inventing one"
    },
    {
        "id": "X02",
        "difficulty": "Advanced",
        "query": "What is the seismic zone factor Z for Delhi according to this document?",
        "gold_evidence": "Not supported by this document; earthquake forces are referenced but seismic zoning values are not given here",
        "expected_answer": "Should not fabricate a value"
    },
    {
        "id": "X03",
        "difficulty": "Advanced",
        "query": "what is the permissible shear stress for M10 concrete?",
        "gold_evidence": "Table 19 starts at M15; M10 is not a listed grade",
        "expected_answer": "Should identify that M10 is absent from Table 19 instead of extrapolating"
    },
    {
        "id": "X04",
        "difficulty": "Advanced",
        "query": "If a flat slab has shear stress exactly above 1.5τc, can I simply increase the reinforcement indefinitely?",
        "gold_evidence": "Clause 31.6.3.2, p.59",
        "expected_answer": "No; if shear stress exceeds 1.5τc, the flat slab shall be redesigned"
    },
    {
        "id": "X05",
        "difficulty": "Advanced",
        "query": "Can I use sea water for reinforced concrete because the code permits it for plain concrete?",
        "gold_evidence": "Clause 5.4.3, p.14",
        "expected_answer": "No; the stated unavoidable-use exception is for plain concrete with no embedded steel"
    },
    {
        "id": "X06",
        "difficulty": "Advanced",
        "query": "The note says '20 mm aggregate is suitable for most work', so is 20 mm always permitted in a heavily reinforced beam?",
        "gold_evidence": "Clause 5.3.3.1, p.14",
        "expected_answer": "No; heavily reinforced members may require a smaller nominal maximum aggregate size based on clear spacing/cover"
    },
    {
        "id": "X07",
        "difficulty": "Advanced",
        "query": "Does a 45° bend always provide 4φ anchorage even if that exceeds the maximum bend anchorage permitted?",
        "gold_evidence": "Clause 26.2.2.1, p.43",
        "expected_answer": "No; 4φ per 45° bend is subject to the 16φ maximum"
    },
    {
        "id": "X08",
        "difficulty": "Advanced",
        "query": "If Q = 0.04 exactly, should the column be classified as sway?",
        "gold_evidence": "Annex E.2, p.92",
        "expected_answer": "No; Q ≤ 0.04 is treated as no-sway"
    },
    {
        "id": "X09",
        "difficulty": "Advanced",
        "query": "Does IS 456 require the use of only design-mix concrete and prohibit nominal mixes?",
        "gold_evidence": "Clause 9, pp.21–22",
        "expected_answer": "No; the contents explicitly include both design mix and nominal mix concrete"
    },
    {
        "id": "X10",
        "difficulty": "Advanced",
        "query": "Can I use a special concrete structural system not covered by IS 456 without any additional evidence?",
        "gold_evidence": "Clause 19, p.3",
        "expected_answer": "The code permits special systems not covered only when satisfactory evidence of adequacy and safety is produced by analysis, test, or both"
    }
]

# ---------------------------------------------------------------------------
# Grading heuristic
# ---------------------------------------------------------------------------
# Fix #9: broadened after checking this against the actual run -- the
# original list missed X02 ("should not fabricate a value") and X03
# ("absent from Table 19 ... instead of extrapolating"), both of which the
# chatbot handled correctly (refused) but got graded FAIL anyway. Verified
# by re-running this exact list against every expected_answer in the CSV.
TRAP_MARKERS = ("not supported", "not directly supported", "not listed", "not given",
                "not covered", "not specified", "excludes", "does not",
                "should not fabricate", "should not invent", "absent from",
                "instead of extrapolating", "instead of inventing", "rather than inventing")
STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "into", "value",
             "values", "when", "which", "what", "where", "condition", "conditions",
             "according", "then", "than", "also", "used", "use", "using"}
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("i could not find")


def grade(q: dict, answer: str) -> str:
    expected = q["expected_answer"]
    expected_lower = expected.lower()
    is_trap = any(marker in expected_lower for marker in TRAP_MARKERS)
    refused = is_refusal(answer)

    if is_trap:
        return "PASS" if refused else "FAIL"
    if refused:
        return "FAIL"

    answer_lower = answer.lower()
    expected_numbers = NUMBER_RE.findall(expected)
    expected_keywords = [w.strip(".,;:()").lower() for w in re.split(r"\s+", expected)
                          if len(w) >= 4 and w.strip(".,;:()").lower() not in STOPWORDS]
    expected_keywords = list(dict.fromkeys(expected_keywords))  # dedupe, keep order

    num_score = None
    if expected_numbers:
        hits = sum(1 for n in expected_numbers if n in answer)
        num_score = hits / len(expected_numbers)

    kw_score = None
    if expected_keywords:
        hits = sum(1 for w in expected_keywords if w in answer_lower)
        kw_score = hits / len(expected_keywords)

    if num_score is not None and kw_score is not None:
        score = 0.7 * num_score + 0.3 * kw_score
    elif num_score is not None:
        score = num_score
    elif kw_score is not None:
        score = kw_score
    else:
        return "REVIEW"

    if score >= 0.6:
        return "PASS"
    if score < 0.3:
        return "FAIL"
    return "REVIEW"


MARK = {"PASS": "\033[92m PASS\033[0m", "FAIL": "\033[91m FAIL\033[0m", "REVIEW": "\033[93m REVIEW\033[0m"}


def load_checkpoint() -> dict:
    """Returns {question_id: {"result": ..., "answer": ...}} already completed."""
    if not os.path.exists(CHECKPOINT_PATH):
        return {}
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  [warn] {CHECKPOINT_PATH} is unreadable/corrupt -- ignoring it and starting fresh.")
        return {}


def save_checkpoint(done: dict) -> None:
    # Write to a temp file and rename, so a crash mid-write never corrupts the
    # checkpoint already on disk (what you'd lose otherwise).
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(done, f, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    parser.add_argument("--fresh", action="store_true",
                         help="ignore any existing checkpoint and re-run every question from scratch")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    corpus = load_corpus(collection)
    if not corpus["ids"]:
        print("Collection is empty — run ingest.py first.")
        return
    bm25_index = build_bm25_index(corpus)

    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS

    done = {} if args.fresh else load_checkpoint()
    if args.fresh and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    already = sum(1 for q in questions if q["id"] in done)
    if already:
        print(f"Resuming from checkpoint: {already}/{len(questions)} question(s) already done "
              f"(delete {CHECKPOINT_PATH}, or pass --fresh, to start over).")
    print(f"Loaded {len(corpus['ids'])} chunks. Running {len(questions)} question(s) live...\n")

    tally = {"PASS": 0, "FAIL": 0, "REVIEW": 0}
    by_difficulty = {}

    for i, q in enumerate(questions, 1):
        diff = q["difficulty"]
        by_difficulty.setdefault(diff, {"PASS": 0, "FAIL": 0, "REVIEW": 0})

        if q["id"] in done:
            result = done[q["id"]]["result"]
            answer = done[q["id"]]["answer"]
            tally[result] += 1
            by_difficulty[diff][result] += 1
            print(f"[{i}/{len(questions)}] {q['id']} ({diff}) — {MARK[result]} (from checkpoint)")
            continue

        candidates = hybrid_retrieve(q["query"], corpus, bm25_index, collection, Filters())
        reranked = rerank(q["query"], candidates)
        answer = generate_answer(q["query"], reranked)

        result = grade(q, answer)
        tally[result] += 1
        by_difficulty[diff][result] += 1

        done[q["id"]] = {"result": result, "answer": answer}
        save_checkpoint(done)

        print(f"[{i}/{len(questions)}] {q['id']} ({diff}) — {MARK[result]}")
        print(f"    Q: {q['query']}")
        print(f"    expected : {q['expected_answer']}")
        shown = answer[:200] + ("..." if len(answer) > 200 else "")
        print(f"    actual   : {shown}")
        print()

    total = sum(tally.values())
    print("=" * 60)
    print(f"PASS: {tally['PASS']}   FAIL: {tally['FAIL']}   REVIEW: {tally['REVIEW']}   (of {total})")
    print(f"Preliminary accuracy (PASS / total): {tally['PASS'] / total:.0%}")
    print(f"Best case if all REVIEW turn out correct: {(tally['PASS'] + tally['REVIEW']) / total:.0%}")
    print()
    print("By difficulty:")
    for diff, counts in by_difficulty.items():
        t = sum(counts.values())
        print(f"  {diff:14s} PASS {counts['PASS']:2d}  FAIL {counts['FAIL']:2d}  REVIEW {counts['REVIEW']:2d}  (of {t})")


if __name__ == "__main__":
    main()