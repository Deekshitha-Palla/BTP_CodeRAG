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

# GENERATION is imported inside main() so --regrade needs no Ollama/Chroma.


CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"
DEFAULT_CHECKPOINT_PATH = "./test_suite_checkpoint.json"

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
        "expected_answer": "Curing extends the hydration of cement, particularly in the surface zone, which reduces permeability and enhances durability"
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
        # Gold updated: Amendment No. 4 (2013) replaced 5.4.3 with "Sea water shall not be
        # used for mixing or curing of concrete" -- the plain-concrete exception is gone.
        # VERIFY against the amendment page of the PDF before quoting.
        "gold_evidence": "Clause 5.4.3 as modified by Amendment No. 4, p.14",
        "expected_answer": "No; sea water shall not be used for mixing or curing of concrete (Amendment No. 4); the original plain-concrete exception never applied to reinforced concrete"
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
# Grading
# ---------------------------------------------------------------------------
# 1. Questions listed in GRADING get explicit, hand-written rules (checked
#    against IS 456). These are the questions the generic heuristic misgraded.
# 2. Everything else uses the generic heuristic (numbers + keywords).
#
# GRADING keys:
#   must    : list of keyword groups; each group is "alt1|alt2" and ANY alt must
#             appear (case-insensitive). ALL groups must be satisfied.
#   say_no  : the answer must open with "No" (false-premise / trap questions
#             where the correct answer is a refusal of the premise).
#   trap    : correct behaviour is to abstain (the document has no answer).
GRADING = {
    "E07":  {"must": ["500"]},
    "E10":  {"must": ["curing", "hydration|permeab|moisture|drying"]},
    "EM06": {"must": ["one-fourth|1/4|one fourth|quarter", "reinforcement"]},
    "EM10": {"must": ["0.049"]},
    "M01":  {"must": ["cover", "dimension"]},
    "H06":  {"must": ["opening", "ineffective|subtract|deduct|free edge"]},
    "H08":  {"must": ["free edge|free corner", "critical section|critical perimeter"]},
    "A10":  {"must": ["table 26", "table 27", "l_y/l_x|ly/lx|aspect|ratio"]},
    "M05":  {"must": ["modification factor", "span|effective depth"]},
    "M10":  {"must": ["12", "no reduction|not permitted|not applicable|does not qualify|do not qualify"
                            "|not eligible|cannot|not allowed|does not apply|not be reduced"]},
    "MH08": {"must": ["1.6"]},
    "X01":  {"trap": True},
    "X02":  {"trap": True},
    "X03":  {"trap": True},
    "X04":  {"say_no": True, "must": ["redesign"]},
    "X05":  {"say_no": True, "must": ["sea water|sea-water|seawater"]},
    "X06":  {"say_no": True, "must": ["5 mm|5mm", "clear|cover"]},
    "X07":  {"say_no": True, "must": ["16"]},
    "X08":  {"say_no": True, "must": ["0.04"]},
    "X09":  {"say_no": True, "must": ["nominal mix"]},
    "X10":  {"say_no": True, "must": ["evidence|test|analysis"]},
}

# Generic heuristic: an expected answer containing one of these means "the
# correct behaviour is to abstain". NOTE: plain "does not" was removed -- it
# fired on M10 ("16 mm does not qualify") and failed a correct answer.
TRAP_MARKERS = ("not supported", "not directly supported", "not listed", "not given",
                "not covered", "not specified", "excludes", "does not provide",
                "should not fabricate", "should not invent", "absent from",
                "instead of extrapolating", "instead of inventing", "rather than inventing")
# Phrases that show the model abstained (a refusal does not have to be the
# exact canned sentence to be correct).
ABSTAIN_MARKERS = ("could not find", "not provided", "not specified", "does not specify",
                   "does not provide", "does not contain", "not listed", "not given",
                   "not available", "not included", "not covered", "no value", "not mentioned",
                   "does not list", "does not include", "not stated", "not found")
STOPWORDS = {"the", "and", "for", "with", "from", "that", "this", "into", "value",
             "values", "when", "which", "what", "where", "condition", "conditions",
             "according", "then", "than", "also", "used", "use", "using"}
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("i could not find")


def has_number(answer: str, n: str) -> bool:
    """Whole-number match: "5" must not match inside "45" or "0.5"."""
    return re.search(rf"(?<![\d.]){re.escape(n)}(?![\d]|\.\d)", answer) is not None


def _group_hit(group: str, answer_lower: str) -> bool:
    for alt in (a.strip().lower() for a in group.split("|")):
        if NUMBER_RE.fullmatch(alt):
            if has_number(answer_lower, alt):
                return True
        elif alt in answer_lower:
            return True
    return False


def _opens_with_no(answer: str) -> bool:
    return re.match(r"^[\W_]*no\b", answer.strip(), re.I) is not None


def grade_explicit(rule: dict, answer: str) -> str:
    answer_lower = answer.lower()
    if rule.get("trap"):
        if is_refusal(answer) or any(m in answer_lower for m in ABSTAIN_MARKERS):
            return "PASS"
        return "REVIEW"   # answered anyway -- a human must check it isn't invented
    if is_refusal(answer):
        return "FAIL"
    must_ok = all(_group_hit(g, answer_lower) for g in rule.get("must", []))
    if not must_ok:
        return "FAIL"
    if rule.get("say_no") and not _opens_with_no(answer):
        # Right content but doesn't open with "No" -- could still be correct
        # ("Based on the excerpts, no, ..."), so a human decides.
        return "REVIEW"
    return "PASS"


def grade(q: dict, answer: str) -> str:
    if answer.startswith("[GENERATION ERROR"):
        return "FAIL"
    rule = GRADING.get(q["id"])
    if rule:
        return grade_explicit(rule, answer)

    expected = q["expected_answer"]
    expected_lower = expected.lower()
    if any(marker in expected_lower for marker in TRAP_MARKERS):
        return grade_explicit({"trap": True}, answer)
    if is_refusal(answer):
        return "FAIL"

    answer_lower = answer.lower()
    expected_numbers = list(dict.fromkeys(NUMBER_RE.findall(expected)))
    expected_keywords = [w.strip(".,;:()").lower() for w in re.split(r"\s+", expected)
                         if len(w) >= 4 and w.strip(".,;:()").lower() not in STOPWORDS]
    expected_keywords = list(dict.fromkeys(expected_keywords))

    num_score = None
    if expected_numbers:
        num_score = sum(has_number(answer, n) for n in expected_numbers) / len(expected_numbers)
    kw_score = None
    if expected_keywords:
        kw_score = sum(w in answer_lower for w in expected_keywords) / len(expected_keywords)

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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> dict:
    """Returns {question_id: {"result": ..., "answer": ..., "sources": [...]}}."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"  [warn] {path} is unreadable/corrupt -- ignoring it and starting fresh.")
        return {}


def save_checkpoint(done: dict, path: str) -> None:
    # Write to a temp file and rename so a crash mid-write never corrupts it.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(done, f, indent=2)
    os.replace(tmp_path, path)


def print_summary(questions: list[dict], done: dict) -> None:
    tally = {"PASS": 0, "FAIL": 0, "REVIEW": 0}
    by_difficulty: dict[str, dict] = {}
    for q in questions:
        if q["id"] not in done:
            continue
        result = done[q["id"]]["result"]
        tally[result] += 1
        by_difficulty.setdefault(q["difficulty"], {"PASS": 0, "FAIL": 0, "REVIEW": 0})[result] += 1
    total = sum(tally.values())
    if not total:
        print("No results.")
        return
    print("=" * 60)
    print(f"PASS: {tally['PASS']}   FAIL: {tally['FAIL']}   REVIEW: {tally['REVIEW']}   (of {total})")
    print(f"Preliminary accuracy (PASS / total): {tally['PASS'] / total:.0%}")
    print(f"Best case if all REVIEW turn out correct: {(tally['PASS'] + tally['REVIEW']) / total:.0%}")
    print()
    print("By difficulty:")
    for diff, counts in by_difficulty.items():
        t = sum(counts.values())
        print(f"  {diff:14s} PASS {counts['PASS']:2d}  FAIL {counts['FAIL']:2d}  REVIEW {counts['REVIEW']:2d}  (of {t})")
    for label in ("FAIL", "REVIEW"):
        ids = [q["id"] for q in questions if done.get(q["id"], {}).get("result") == label]
        if ids:
            print(f"\n{label}: {', '.join(ids)}")


def regrade(questions: list[dict], path: str) -> None:
    """Re-grade saved answers with the CURRENT grader. No model, no GPU."""
    done = load_checkpoint(path)
    if not done:
        print(f"No checkpoint at {path}.")
        return
    changed = []
    for q in questions:
        if q["id"] not in done:
            continue
        old = done[q["id"]]["result"]
        new = grade(q, done[q["id"]]["answer"])
        if new != old:
            changed.append(f"  {q['id']:5s} {old:6s} -> {new}")
        done[q["id"]]["result"] = new
    save_checkpoint(done, path)
    print(f"Re-graded {path}. {len(changed)} verdict(s) changed:")
    print("\n".join(changed) if changed else "  (none)")
    print()
    print_summary(questions, done)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    parser.add_argument("--ids", default=None, help="question IDs to run, e.g. MH08+X10 (use + inside qsub -v)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing checkpoint and re-run every question from scratch")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH,
                        help="checkpoint file (use a new name per run to keep runs separate)")
    parser.add_argument("--regrade", action="store_true",
                        help="re-grade the answers in --checkpoint with the current grader; no model calls")
    args = parser.parse_args()

    questions = QUESTIONS[:args.limit] if args.limit else QUESTIONS
    if args.ids:
        wanted = {i for i in re.split(r"[,+\s]+", args.ids) if i}   # "+" works inside qsub -v
        questions = [q for q in questions if q["id"] in wanted]

    if args.regrade:
        regrade(questions, args.checkpoint)
        return

    # Imported here so --regrade works without Ollama/Chroma/flashrank.
    from GENERATION import Pipeline, generate_answer, describe_chunk
    pipe = Pipeline()

    if args.fresh and os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)
    done = {} if args.fresh else load_checkpoint(args.checkpoint)
    already = sum(1 for q in questions if q["id"] in done)
    if already:
        print(f"Resuming from {args.checkpoint}: {already}/{len(questions)} question(s) already done "
              f"(pass --fresh to start over).")
    print(f"Loaded {len(pipe.corpus['ids'])} chunks. Running {len(questions)} question(s) live...\n")

    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            print(f"[{i}/{len(questions)}] {q['id']} ({q['difficulty']}) — {MARK[done[q['id']]['result']]} (from checkpoint)")
            continue

        chunks = pipe.retrieve(q["query"])
        answer = generate_answer(q["query"], chunks)
        result = grade(q, answer)

        # Retrieved chunks are saved so a FAIL can be triaged later
        # (retrieval miss vs. generation error) without re-running anything.
        done[q["id"]] = {"result": result, "answer": answer,
                         "sources": [describe_chunk(c) for c in chunks]}
        save_checkpoint(done, args.checkpoint)

        print(f"[{i}/{len(questions)}] {q['id']} ({q['difficulty']}) — {MARK[result]}")
        print(f"    Q: {q['query']}")
        print(f"    expected : {q['expected_answer']}")
        shown = answer[:200] + ("..." if len(answer) > 200 else "")
        print(f"    actual   : {shown}")
        print(f"    top chunk: {done[q['id']]['sources'][0] if chunks else '(none)'}")
        print()

    print_summary(questions, done)


if __name__ == "__main__":
    main()
