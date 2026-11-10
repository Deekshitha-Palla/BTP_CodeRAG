# """
# Manual content injection — for the specific notes/tables that 4 rounds of
# debugging confirmed are NOT recoverable via automated Docling/OCR extraction
# (verified via inspect_store.py: zero hits anywhere in the corpus for their
# content, under any tag).

# This bypasses parsing entirely for just these entries. Add more to
# MANUAL_CHUNKS as you find other confirmed-missing content (e.g. Table 16A's
# "tance Beams" column confusion, Table 4's missing cells) — but only after
# the same level of confirmation you just did for these three notes; this is
# a scalpel for specific known gaps, not a general-purpose fallback.

# Usage:
#     python add_manual_content.py
# """

# import hashlib
# import ollama
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# # Hand-transcribed, verified against the source PDF directly.
# MANUAL_CHUNKS = [
#     {
#         "text": "NOTES to Table 16 (Nominal Cover to Meet Durability Requirements, "
#                 "Clause 26.4.2):\n"
#                 "1 For main reinforcement up to 12 mm diameter bar for mild exposure "
#                 "the nominal cover may be reduced by 5 mm.\n"
#                 "2 Unless specified otherwise, actual concrete cover should not "
#                 "deviate from the required nominal cover by +10 mm -0 mm.\n"
#                 "3 For exposure condition 'severe' and 'very severe', reduction of "
#                 "5 mm may be made, where concrete grade is M35 and above.",
#         "page": 9,
#         "content_type": "table_note",
#         "related_table": "Table 16 on IS456.pdf page 48",
#     },
#     {
#         # Transcribed directly from a clean scan (2026-09-12). Standard
#         # typeset text, high confidence.
#         "text": "Table 4 Requirements for Concrete Exposed to Sulphate Attack "
#                 "(Clauses 8.2.2.4 and 9.1.2)\n\n"
#                 "| SI No. | Class | Total SO3 in Soil (%) | SO3 in 2:1 Water:Soil Extract (g/l) | SO3 in Ground Water (g/l) | Type of Cement | Minimum Cement Content (kg/m3) | Maximum Free Water-Cement Ratio |\n"
#                 "|---|---|---|---|---|---|---|---|\n"
#                 "| i) | 1 | Traces (<0.2) | Less than 1.0 | Less than 0.3 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 280 | 0.55 |\n"
#                 "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 330 | 0.50 |\n"
#                 "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Supersulphated cement or sulphate resisting Portland cement | 310 | 0.50 |\n"
#                 "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Supersulphated cement or sulphate resisting Portland cement | 330 | 0.50 |\n"
#                 "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Portland pozzolana cement or Portland slag cement | 350 | 0.45 |\n"
#                 "| iv) | 4 | 1.0 to 2.0 | 3.1 to 5.0 | 2.5 to 5.0 | Supersulphated or sulphate resisting Portland cement | 370 | 0.45 |\n"
#                 "| v) | 5 | More than 2.0 | More than 5.0 | More than 5.0 | Sulphate resisting Portland cement or supersulphated cement with protective coatings | 400 | 0.40 |\n\n"
#                 "NOTES:\n"
#                 "1 Cement content given in this table is irrespective of grades of cement.\n"
#                 "2 Use of supersulphated cement is generally restricted where the prevailing temperature is above 40C.\n"
#                 "3 Supersulphated cement gives an acceptable life provided that the concrete is dense and prepared with a water-cement ratio of 0.4 or less, in mineral acids, down to pH 3.5.\n"
#                 "4 The cement contents given in col 6 of this table are the minimum recommended. For SO3 contents near the upper limit of any class, cement contents above these minimum are advised.\n"
#                 "5 For severe conditions, such as thin sections under hydrostatic pressure on one side only and sections partly immersed, considerations should be given to a further reduction of water-cement ratio.\n"
#                 "6 Portland slag cement conforming to IS 455 with slag content more than 50 percent exhibits better sulphate resisting properties.\n"
#                 "7 Where chloride is encountered along with sulphates in soil or ground water, ordinary Portland cement with C3A content from 5 to 8 percent shall be desirable to be used in concrete, instead of sulphate resisting cement.",
#         "page": 19,
#         "content_type": "table",
#         "related_table": "",
#     },
#     {
#         # Transcribed from a photographed page with small print/underlines --
#         # PLEASE double-check the h=1.5 and h=2 Ribs-Continuous cells against
#         # your original before fully trusting this one; those two were the
#         # hardest to read confidently.
#         "text": "Table 16A Nominal Cover to Meet Specified Period of Fire Resistance "
#                 "(Clauses 21.4 and 26.4.3 and Fig. 1)\n\n"
#                 "| Fire Resistance h (hours) | Beams - Simply supported (mm) | Beams - Continuous (mm) | Slabs - Simply supported (mm) | Slabs - Continuous (mm) | Ribs - Simply supported (mm) | Ribs - Continuous (mm) | Columns (mm) |\n"
#                 "|---|---|---|---|---|---|---|---|\n"
#                 "| 0.5 | 20 | 20 | 20 | 20 | 20 | 20 | 40 |\n"
#                 "| 1 | 20 | 20 | 20 | 25 | 20 | 20 | 40 |\n"
#                 "| 1.5 | 20 | 20 | 25 | 20 | 35 | 35 | 40 |\n"
#                 "| 2 | 40 | 30 | 35 | 25 | 45 | 35 | 40 |\n"
#                 "| 3 | 60 | 40 | 45 | 35 | 55 | 45 | 40 |\n"
#                 "| 4 | 70 | 50 | 55 | 45 | 65 | 55 | 40 |\n\n"
#                 "NOTES:\n"
#                 "1 The nominal covers given relate specifically to the minimum member dimensions given in Fig. 1.\n"
#                 "2 Cases that lie below the bold line require attention to the additional measures necessary to reduce the risks of spalling (see 21.3.1).",
#         "page": 9,
#         "content_type": "table",
#         "related_table": "",
#     },
#     # Add more verified entries here as you confirm them missing.
# ]


# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     ids, embeddings, documents, metadatas = [], [], [], []
#     for chunk in MANUAL_CHUNKS:
#         resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])
#         chunk_id = hashlib.sha256(
#             (f"manual::{chunk['text']}").encode()
#         ).hexdigest()[:16]

#         ids.append(chunk_id)
#         embeddings.append(resp["embedding"])
#         documents.append(chunk["text"])
#         metadatas.append({
#             "source": "IS456.pdf",
#             "page": chunk["page"],
#             "content_type": chunk["content_type"],
#             "clause": "unknown",
#             "related_table": chunk.get("related_table", ""),
#             "low_confidence": False,
#             "verified": True,  # marks this as hand-checked, not auto-extracted
#         })

#     collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"Added {len(ids)} manually verified chunk(s). "
#           f"Collection now has {collection.count()} total chunks.")


# if __name__ == "__main__":
#     main()
# """
# Manual content injection — for the specific notes/tables that 4 rounds of
# debugging confirmed are NOT recoverable via automated Docling/OCR extraction
# (verified via inspect_store.py: zero hits anywhere in the corpus for their
# content, under any tag).

# This bypasses parsing entirely for just these entries. Add more to
# MANUAL_CHUNKS as you find other confirmed-missing content (e.g. Table 16A's
# "tance Beams" column confusion, Table 4's missing cells) — but only after
# the same level of confirmation you just did for these three notes; this is
# a scalpel for specific known gaps, not a general-purpose fallback.

# Usage:
#     python add_manual_content.py
# """

# import hashlib
# import ollama
# import chromadb

# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# # Hand-transcribed, verified against the source PDF directly.
# MANUAL_CHUNKS = [
#     {
#         "text": "NOTES to Table 16 (Nominal Cover to Meet Durability Requirements, "
#                 "Clause 26.4.2):\n"
#                 "1 For main reinforcement up to 12 mm diameter bar for mild exposure "
#                 "the nominal cover may be reduced by 5 mm.\n"
#                 "2 Unless specified otherwise, actual concrete cover should not "
#                 "deviate from the required nominal cover by +10 mm -0 mm.\n"
#                 "3 For exposure condition 'severe' and 'very severe', reduction of "
#                 "5 mm may be made, where concrete grade is M35 and above.",
#         "page": 9,
#         "content_type": "table_note",
#         "table": "16",
#         "related_table": "Table 16 on sample.pdf page 9",
#     },
#     {
#         # Transcribed directly from a clean scan (2026-09-12). Standard
#         # typeset text, high confidence.
#         "text": "Table 4 Requirements for Concrete Exposed to Sulphate Attack "
#                 "(Clauses 8.2.2.4 and 9.1.2)\n\n"
#                 "| SI No. | Class | Total SO3 in Soil (%) | SO3 in 2:1 Water:Soil Extract (g/l) | SO3 in Ground Water (g/l) | Type of Cement | Minimum Cement Content (kg/m3) | Maximum Free Water-Cement Ratio |\n"
#                 "|---|---|---|---|---|---|---|---|\n"
#                 "| i) | 1 | Traces (<0.2) | Less than 1.0 | Less than 0.3 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 280 | 0.55 |\n"
#                 "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 330 | 0.50 |\n"
#                 "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Supersulphated cement or sulphate resisting Portland cement | 310 | 0.50 |\n"
#                 "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Supersulphated cement or sulphate resisting Portland cement | 330 | 0.50 |\n"
#                 "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Portland pozzolana cement or Portland slag cement | 350 | 0.45 |\n"
#                 "| iv) | 4 | 1.0 to 2.0 | 3.1 to 5.0 | 2.5 to 5.0 | Supersulphated or sulphate resisting Portland cement | 370 | 0.45 |\n"
#                 "| v) | 5 | More than 2.0 | More than 5.0 | More than 5.0 | Sulphate resisting Portland cement or supersulphated cement with protective coatings | 400 | 0.40 |\n\n"
#                 "NOTES:\n"
#                 "1 Cement content given in this table is irrespective of grades of cement.\n"
#                 "2 Use of supersulphated cement is generally restricted where the prevailing temperature is above 40C.\n"
#                 "3 Supersulphated cement gives an acceptable life provided that the concrete is dense and prepared with a water-cement ratio of 0.4 or less, in mineral acids, down to pH 3.5.\n"
#                 "4 The cement contents given in col 6 of this table are the minimum recommended. For SO3 contents near the upper limit of any class, cement contents above these minimum are advised.\n"
#                 "5 For severe conditions, such as thin sections under hydrostatic pressure on one side only and sections partly immersed, considerations should be given to a further reduction of water-cement ratio.\n"
#                 "6 Portland slag cement conforming to IS 455 with slag content more than 50 percent exhibits better sulphate resisting properties.\n"
#                 "7 Where chloride is encountered along with sulphates in soil or ground water, ordinary Portland cement with C3A content from 5 to 8 percent shall be desirable to be used in concrete, instead of sulphate resisting cement.",
#         "page": 19,
#         "content_type": "table",
#         "table": "4",
#         "related_table": "",
#     },
#     {
#         # Transcribed from a photographed page with small print/underlines --
#         # PLEASE double-check the h=1.5 and h=2 Ribs-Continuous cells against
#         # your original before fully trusting this one; those two were the
#         # hardest to read confidently.
#         "text": "Table 16A Nominal Cover to Meet Specified Period of Fire Resistance "
#                 "(Clauses 21.4 and 26.4.3 and Fig. 1)\n\n"
#                 "| Fire Resistance h (hours) | Beams - Simply supported (mm) | Beams - Continuous (mm) | Slabs - Simply supported (mm) | Slabs - Continuous (mm) | Ribs - Simply supported (mm) | Ribs - Continuous (mm) | Columns (mm) |\n"
#                 "|---|---|---|---|---|---|---|---|\n"
#                 "| 0.5 | 20 | 20 | 20 | 20 | 20 | 20 | 40 |\n"
#                 "| 1 | 20 | 20 | 20 | 25 | 20 | 20 | 40 |\n"
#                 "| 1.5 | 20 | 20 | 25 | 20 | 35 | 35 | 40 |\n"
#                 "| 2 | 40 | 30 | 35 | 25 | 45 | 35 | 40 |\n"
#                 "| 3 | 60 | 40 | 45 | 35 | 55 | 45 | 40 |\n"
#                 "| 4 | 70 | 50 | 55 | 45 | 65 | 55 | 40 |\n\n"
#                 "NOTES:\n"
#                 "1 The nominal covers given relate specifically to the minimum member dimensions given in Fig. 1.\n"
#                 "2 Cases that lie below the bold line require attention to the additional measures necessary to reduce the risks of spalling (see 21.3.1).",
#         "page": 9,
#         "content_type": "table",
#         "table": "16A",
#         "related_table": "",
#     },
#     {
#         # Fix #8: this is the table body that EM05 was missing -- the notes-
#         # only pattern from Table 16 below, but for solids in mixing water.
#         # Transcribed from the official IS 456:2000 text (BIS, via
#         # law.resource.org), cross-checked against the gold CSV: EM05 expects
#         # "200 mg/L organic; 3000 mg/L inorganic", E07 expects "2000 mg/L as
#         # Cl" for reinforced concrete -- both match row iv below exactly.
#         # NOTE: this is Table 1, not "Table 3" as an earlier draft called it
#         # -- Table 3 (Environmental Exposure Conditions) is a different table
#         # your bot already answers correctly (EM03 passed).
#         "text": "Table 1 Permissible Limit for Solids (Clause 5.4)\n\n"
#                 "| SI No. | Solid | Tested as per | Permissible Limit, Max |\n"
#                 "|---|---|---|---|\n"
#                 "| i) | Organic | IS 3025 (Part 18) | 200 mg/l |\n"
#                 "| ii) | Inorganic | IS 3025 (Part 18) | 3000 mg/l |\n"
#                 "| iii) | Sulphates (as SO3) | IS 3025 (Part 24) | 400 mg/l |\n"
#                 "| iv) | Chlorides (as Cl) | IS 3025 (Part 32) | 2000 mg/l for concrete not containing embedded steel and 500 mg/l for reinforced concrete work |\n"
#                 "| v) | Suspended matter | IS 3025 (Part 17) | 2000 mg/l |\n",
#         "page": 14,
#         "content_type": "table",
#         "table": "1",
#         "related_table": "",
#     },
#     {
#         # Fix #8: EM01/EM02's missing table body -- you'd only transcribed
#         # Table 16's NOTES (further up in this file), never the base
#         # exposure-vs-cover values themselves. Same source/cross-check
#         # method as above; EM01 expects "45 mm" (severe), EM02 expects
#         # "75 mm" (extreme) -- both match directly.
#         "text": "Table 16 Nominal Cover to Meet Durability Requirements (Clause 26.4.2)\n\n"
#                 "| Exposure | Nominal Concrete Cover in mm not Less Than |\n"
#                 "|---|---|\n"
#                 "| Mild | 20 |\n"
#                 "| Moderate | 30 |\n"
#                 "| Severe | 45 |\n"
#                 "| Very severe | 50 |\n"
#                 "| Extreme | 75 |\n\n"
#                 "NOTES:\n"
#                 "1 For main reinforcement up to 12 mm diameter bar for mild exposure the nominal cover may be reduced by 5 mm.\n"
#                 "2 Unless specified otherwise, actual concrete cover should not deviate from the required nominal cover by +10 mm -0 mm.\n"
#                 "3 For exposure condition 'severe' and 'very severe', reduction of 5 mm may be made, where concrete grade is M35 and above.",
#         "page": 9,  # matches the existing Table 16 NOTES entry above -- same physical page
#         "content_type": "table",
#         "table": "16",
#         "related_table": "",
#     },
#     {
#         # Fix #8: MH05/MH06/X03's missing table -- the clause pointing to
#         # Table 19 kept retrieving fine; the table itself never did. Cross-
#         # checked against MH05's gold value (M25 @ 1.0% steel = 0.64 N/mm2)
#         # and MH06's (same value, compared against Table 20 below).
#         "text": "Table 19 Design Shear Strength of Concrete, tau_c (N/mm2) (Clause 40.2.1)\n\n"
#                 "| 100 As/bd | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
#                 "|---|---|---|---|---|---|---|\n"
#                 "| <=0.15 | 0.28 | 0.28 | 0.29 | 0.29 | 0.29 | 0.30 |\n"
#                 "| 0.25 | 0.35 | 0.36 | 0.36 | 0.37 | 0.37 | 0.38 |\n"
#                 "| 0.50 | 0.46 | 0.48 | 0.49 | 0.50 | 0.50 | 0.51 |\n"
#                 "| 0.75 | 0.54 | 0.56 | 0.57 | 0.59 | 0.59 | 0.60 |\n"
#                 "| 1.00 | 0.60 | 0.62 | 0.64 | 0.66 | 0.67 | 0.68 |\n"
#                 "| 1.25 | 0.64 | 0.67 | 0.70 | 0.71 | 0.73 | 0.74 |\n"
#                 "| 1.50 | 0.68 | 0.72 | 0.74 | 0.76 | 0.78 | 0.79 |\n"
#                 "| 1.75 | 0.71 | 0.75 | 0.78 | 0.80 | 0.82 | 0.84 |\n"
#                 "| 2.00 | 0.71 | 0.79 | 0.82 | 0.84 | 0.86 | 0.88 |\n"
#                 "| 2.25 | 0.71 | 0.81 | 0.85 | 0.88 | 0.90 | 0.92 |\n"
#                 "| 2.50 | 0.71 | 0.82 | 0.88 | 0.91 | 0.93 | 0.95 |\n"
#                 "| 2.75 | 0.71 | 0.82 | 0.90 | 0.94 | 0.96 | 0.98 |\n"
#                 "| 3.00 and above | 0.71 | 0.82 | 0.92 | 0.96 | 0.99 | 1.01 |\n\n"
#                 "NOTE: 100 As/bd is the percentage of tension reinforcement.",
#         "page": 73,
#         "content_type": "table",
#         "table": "19",
#         "related_table": "",
#     },
#     {
#         # Fix #8: needed alongside Table 19 for the MH06/X04-style
#         # compare-against-the-max-limit questions (nominal shear stress
#         # must never exceed tau_c,max regardless of shear reinforcement).
#         "text": "Table 20 Maximum Shear Stress, tau_c,max (N/mm2) (Clauses 40.2.3, 40.5.1 and 41.3.1)\n\n"
#                 "| Concrete Grade | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
#                 "|---|---|---|---|---|---|---|\n"
#                 "| tau_c,max | 2.5 | 2.8 | 3.1 | 3.5 | 3.7 | 4.0 |\n",
#         "page": 73,
#         "content_type": "table",
#         "table": "20",
#         "related_table": "",
#     },
#     {
#         # Fix #8: MH01's missing table -- the clause (26.2.1.1) kept
#         # retrieving; these are the base plain-bar values it refers readers
#         # to. Cross-checked against MH01's gold value directly (M25 plain
#         # bar = 1.4 N/mm2; deformed-bar value = 1.4 x 1.6 = 2.24 N/mm2,
#         # per the +60% rule already answered correctly elsewhere as EM07).
#         "text": "Table 21 Design Bond Stress for Plain Bars in Tension, tau_bd (N/mm2) "
#                 "(Clause 26.2.1.1)\n\n"
#                 "| Concrete Grade | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
#                 "|---|---|---|---|---|---|---|\n"
#                 "| tau_bd | 1.0 | 1.2 | 1.4 | 1.5 | 1.7 | 1.9 |\n\n"
#                 "NOTE: For deformed bars conforming to IS 1786, these values shall be increased by 60 percent.",
#         "page": 44,
#         "content_type": "table",
#         "table": "21",
#         "related_table": "",
#     },
#     # Add more verified entries here as you confirm them missing.
#     # Fix #7: every entry above now carries its own "table" key -- when you
#     # add more, give each one its real table number (or omit the key/leave
#     # it "" for a genuinely tableless entry) so main() below can stop
#     # hardcoding every manual chunk's clause to "unknown".
# ]


# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     ids, embeddings, documents, metadatas = [], [], [], []
#     for chunk in MANUAL_CHUNKS:
#         resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])
#         chunk_id = hashlib.sha256(
#             (f"manual::{chunk['text']}").encode()
#         ).hexdigest()[:16]

#         ids.append(chunk_id)
#         embeddings.append(resp["embedding"])
#         documents.append(chunk["text"])
#         table_num = chunk.get("table") or None  # Fix #7
#         metadatas.append({
#             "source": "sample.pdf",
#             "page": chunk["page"],
#             "content_type": chunk["content_type"],
#             "clause": f"Table {table_num}" if table_num else "unknown",
#             "table": table_num or "",
#             "related_table": chunk.get("related_table", ""),
#             "low_confidence": False,
#             "verified": True,  # marks this as hand-checked, not auto-extracted
#         })

#     collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"Added {len(ids)} manually verified chunk(s). "
#           f"Collection now has {collection.count()} total chunks.")


# if __name__ == "__main__":
#     main()
"""
Manual content injection — for the specific notes/tables that 4 rounds of
debugging confirmed are NOT recoverable via automated Docling/OCR extraction
(verified via inspect_store.py: zero hits anywhere in the corpus for their
content, under any tag).

This bypasses parsing entirely for just these entries. Add more to
MANUAL_CHUNKS as you find other confirmed-missing content (e.g. Table 16A's
"tance Beams" column confusion, Table 4's missing cells) — but only after
the same level of confirmation you just did for these three notes; this is
a scalpel for specific known gaps, not a general-purpose fallback.

Usage:
    python add_manual_content.py
"""

import hashlib
import ollama
import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"
EMBED_MODEL = "nomic-embed-text"

# Hand-transcribed, verified against the source PDF directly.
MANUAL_CHUNKS = [
    {
        "text": "NOTES to Table 16 (Nominal Cover to Meet Durability Requirements, "
                "Clause 26.4.2):\n"
                "1 For main reinforcement up to 12 mm diameter bar for mild exposure "
                "the nominal cover may be reduced by 5 mm.\n"
                "2 Unless specified otherwise, actual concrete cover should not "
                "deviate from the required nominal cover by +10 mm -0 mm.\n"
                "3 For exposure condition 'severe' and 'very severe', reduction of "
                "5 mm may be made, where concrete grade is M35 and above.",
        "page": 9,
        "content_type": "table_note",
        "table": "16",
        "related_table": "Table 16 on sample.pdf page 9",
    },
    {
        # Transcribed directly from a clean scan (2026-09-12). Standard
        # typeset text, high confidence.
        "text": "Table 4 Requirements for Concrete Exposed to Sulphate Attack "
                "(Clauses 8.2.2.4 and 9.1.2)\n\n"
                "| SI No. | Class | Total SO3 in Soil (%) | SO3 in 2:1 Water:Soil Extract (g/l) | SO3 in Ground Water (g/l) | Type of Cement | Minimum Cement Content (kg/m3) | Maximum Free Water-Cement Ratio |\n"
                "|---|---|---|---|---|---|---|---|\n"
                "| i) | 1 | Traces (<0.2) | Less than 1.0 | Less than 0.3 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 280 | 0.55 |\n"
                "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Ordinary Portland cement or Portland slag cement or Portland pozzolana cement | 330 | 0.50 |\n"
                "| ii) | 2 | 0.2 to 0.5 | 1.0 to 1.9 | 0.3 to 1.2 | Supersulphated cement or sulphate resisting Portland cement | 310 | 0.50 |\n"
                "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Supersulphated cement or sulphate resisting Portland cement | 330 | 0.50 |\n"
                "| iii) | 3 | 0.5 to 1.0 | 1.9 to 3.1 | 1.2 to 2.5 | Portland pozzolana cement or Portland slag cement | 350 | 0.45 |\n"
                "| iv) | 4 | 1.0 to 2.0 | 3.1 to 5.0 | 2.5 to 5.0 | Supersulphated or sulphate resisting Portland cement | 370 | 0.45 |\n"
                "| v) | 5 | More than 2.0 | More than 5.0 | More than 5.0 | Sulphate resisting Portland cement or supersulphated cement with protective coatings | 400 | 0.40 |\n\n"
                "NOTES:\n"
                "1 Cement content given in this table is irrespective of grades of cement.\n"
                "2 Use of supersulphated cement is generally restricted where the prevailing temperature is above 40C.\n"
                "3 Supersulphated cement gives an acceptable life provided that the concrete is dense and prepared with a water-cement ratio of 0.4 or less, in mineral acids, down to pH 3.5.\n"
                "4 The cement contents given in col 6 of this table are the minimum recommended. For SO3 contents near the upper limit of any class, cement contents above these minimum are advised.\n"
                "5 For severe conditions, such as thin sections under hydrostatic pressure on one side only and sections partly immersed, considerations should be given to a further reduction of water-cement ratio.\n"
                "6 Portland slag cement conforming to IS 455 with slag content more than 50 percent exhibits better sulphate resisting properties.\n"
                "7 Where chloride is encountered along with sulphates in soil or ground water, ordinary Portland cement with C3A content from 5 to 8 percent shall be desirable to be used in concrete, instead of sulphate resisting cement.",
        "page": 19,
        "content_type": "table",
        "table": "4",
        "related_table": "",
    },
    {
        # Transcribed from a photographed page with small print/underlines --
        # PLEASE double-check the h=1.5 and h=2 Ribs-Continuous cells against
        # your original before fully trusting this one; those two were the
        # hardest to read confidently.
        "text": "Table 16A Nominal Cover to Meet Specified Period of Fire Resistance "
                "(Clauses 21.4 and 26.4.3 and Fig. 1)\n\n"
                "| Fire Resistance h (hours) | Beams - Simply supported (mm) | Beams - Continuous (mm) | Slabs - Simply supported (mm) | Slabs - Continuous (mm) | Ribs - Simply supported (mm) | Ribs - Continuous (mm) | Columns (mm) |\n"
                "|---|---|---|---|---|---|---|---|\n"
                "| 0.5 | 20 | 20 | 20 | 20 | 20 | 20 | 40 |\n"
                "| 1 | 20 | 20 | 20 | 25 | 20 | 20 | 40 |\n"
                "| 1.5 | 20 | 20 | 25 | 20 | 35 | 35 | 40 |\n"
                "| 2 | 40 | 30 | 35 | 25 | 45 | 35 | 40 |\n"
                "| 3 | 60 | 40 | 45 | 35 | 55 | 45 | 40 |\n"
                "| 4 | 70 | 50 | 55 | 45 | 65 | 55 | 40 |\n\n"
                "NOTES:\n"
                "1 The nominal covers given relate specifically to the minimum member dimensions given in Fig. 1.\n"
                "2 Cases that lie below the bold line require attention to the additional measures necessary to reduce the risks of spalling (see 21.3.1).",
        "page": 9,
        "content_type": "table",
        "table": "16A",
        "related_table": "",
    },
    {
        # Fix #8: this is the table body that EM05 was missing -- the notes-
        # only pattern from Table 16 below, but for solids in mixing water.
        # Transcribed from the official IS 456:2000 text (BIS, via
        # law.resource.org), cross-checked against the gold CSV: EM05 expects
        # "200 mg/L organic; 3000 mg/L inorganic", E07 expects "2000 mg/L as
        # Cl" for reinforced concrete -- both match row iv below exactly.
        # NOTE: this is Table 1, not "Table 3" as an earlier draft called it
        # -- Table 3 (Environmental Exposure Conditions) is a different table
        # your bot already answers correctly (EM03 passed).
        "text": "Table 1 Permissible Limit for Solids (Clause 5.4)\n\n"
                "| SI No. | Solid | Tested as per | Permissible Limit, Max |\n"
                "|---|---|---|---|\n"
                "| i) | Organic | IS 3025 (Part 18) | 200 mg/l |\n"
                "| ii) | Inorganic | IS 3025 (Part 18) | 3000 mg/l |\n"
                "| iii) | Sulphates (as SO3) | IS 3025 (Part 24) | 400 mg/l |\n"
                "| iv) | Chlorides (as Cl) | IS 3025 (Part 32) | 2000 mg/l for concrete not containing embedded steel and 500 mg/l for reinforced concrete work |\n"
                "| v) | Suspended matter | IS 3025 (Part 17) | 2000 mg/l |\n",
        "page": 14,
        "content_type": "table",
        "table": "1",
        "related_table": "",
    },
    {
        # Fix #8: EM01/EM02's missing table body -- you'd only transcribed
        # Table 16's NOTES (further up in this file), never the base
        # exposure-vs-cover values themselves. Same source/cross-check
        # method as above; EM01 expects "45 mm" (severe), EM02 expects
        # "75 mm" (extreme) -- both match directly.
        "text": "Table 16 Nominal Cover to Meet Durability Requirements (Clause 26.4.2)\n\n"
                "| Exposure | Nominal Concrete Cover in mm not Less Than |\n"
                "|---|---|\n"
                "| Mild | 20 |\n"
                "| Moderate | 30 |\n"
                "| Severe | 45 |\n"
                "| Very severe | 50 |\n"
                "| Extreme | 75 |\n\n"
                "NOTES:\n"
                "1 For main reinforcement up to 12 mm diameter bar for mild exposure the nominal cover may be reduced by 5 mm.\n"
                "2 Unless specified otherwise, actual concrete cover should not deviate from the required nominal cover by +10 mm -0 mm.\n"
                "3 For exposure condition 'severe' and 'very severe', reduction of 5 mm may be made, where concrete grade is M35 and above.",
        "page": 9,  # matches the existing Table 16 NOTES entry above -- same physical page
        "content_type": "table",
        "table": "16",
        "related_table": "",
    },
    {
        # Fix #8: MH05/MH06/X03's missing table -- the clause pointing to
        # Table 19 kept retrieving fine; the table itself never did. Cross-
        # checked against MH05's gold value (M25 @ 1.0% steel = 0.64 N/mm2)
        # and MH06's (same value, compared against Table 20 below).
        "text": "Table 19 Design Shear Strength of Concrete, tau_c (N/mm2) (Clause 40.2.1)\n\n"
                "| 100 As/bd | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
                "|---|---|---|---|---|---|---|\n"
                "| <=0.15 | 0.28 | 0.28 | 0.29 | 0.29 | 0.29 | 0.30 |\n"
                "| 0.25 | 0.35 | 0.36 | 0.36 | 0.37 | 0.37 | 0.38 |\n"
                "| 0.50 | 0.46 | 0.48 | 0.49 | 0.50 | 0.50 | 0.51 |\n"
                "| 0.75 | 0.54 | 0.56 | 0.57 | 0.59 | 0.59 | 0.60 |\n"
                "| 1.00 | 0.60 | 0.62 | 0.64 | 0.66 | 0.67 | 0.68 |\n"
                "| 1.25 | 0.64 | 0.67 | 0.70 | 0.71 | 0.73 | 0.74 |\n"
                "| 1.50 | 0.68 | 0.72 | 0.74 | 0.76 | 0.78 | 0.79 |\n"
                "| 1.75 | 0.71 | 0.75 | 0.78 | 0.80 | 0.82 | 0.84 |\n"
                "| 2.00 | 0.71 | 0.79 | 0.82 | 0.84 | 0.86 | 0.88 |\n"
                "| 2.25 | 0.71 | 0.81 | 0.85 | 0.88 | 0.90 | 0.92 |\n"
                "| 2.50 | 0.71 | 0.82 | 0.88 | 0.91 | 0.93 | 0.95 |\n"
                "| 2.75 | 0.71 | 0.82 | 0.90 | 0.94 | 0.96 | 0.98 |\n"
                "| 3.00 and above | 0.71 | 0.82 | 0.92 | 0.96 | 0.99 | 1.01 |\n\n"
                "NOTE: 100 As/bd is the percentage of tension reinforcement.",
        "page": 73,
        "content_type": "table",
        "table": "19",
        "related_table": "",
    },
    {
        # Fix #8: needed alongside Table 19 for the MH06/X04-style
        # compare-against-the-max-limit questions (nominal shear stress
        # must never exceed tau_c,max regardless of shear reinforcement).
        "text": "Table 20 Maximum Shear Stress, tau_c,max (N/mm2) (Clauses 40.2.3, 40.5.1 and 41.3.1)\n\n"
                "| Concrete Grade | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
                "|---|---|---|---|---|---|---|\n"
                "| tau_c,max | 2.5 | 2.8 | 3.1 | 3.5 | 3.7 | 4.0 |\n",
        "page": 73,
        "content_type": "table",
        "table": "20",
        "related_table": "",
    },
    {
        # Fix #8: MH01's missing table -- the clause (26.2.1.1) kept
        # retrieving; these are the base plain-bar values it refers readers
        # to. Cross-checked against MH01's gold value directly (M25 plain
        # bar = 1.4 N/mm2; deformed-bar value = 1.4 x 1.6 = 2.24 N/mm2,
        # per the +60% rule already answered correctly elsewhere as EM07).
        "text": "Table 21 Design Bond Stress for Plain Bars in Tension, tau_bd (N/mm2) "
                "(Clause 26.2.1.1)\n\n"
                "| Concrete Grade | M15 | M20 | M25 | M30 | M35 | M40 and above |\n"
                "|---|---|---|---|---|---|---|\n"
                "| tau_bd | 1.0 | 1.2 | 1.4 | 1.5 | 1.7 | 1.9 |\n\n"
                "NOTE: For deformed bars conforming to IS 1786, these values shall be increased by 60 percent.",
        "page": 44,
        "content_type": "table",
        "table": "21",
        "related_table": "",
    },
    {
        # Fig. 4 -- transcribed directly from the rendered PDF page (NOT
        # VLM-generated; the Qwen2-VL run on this exact figure fabricated
        # curve labels "0.58, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0" -- the real
        # labels, read directly off the chart, are the fs values below.
        "text": "Fig. 4 Modification Factor for Tension Reinforcement\n\n"
                "X-axis: Percentage Tension Reinforcement (0 to 3.0)\n"
                "Y-axis: Modification Factor (0.4 to 2.0)\n\n"
                "Five curves, each for a different service-load steel stress f_s "
                "(N/mm2): f_s = 120, f_s = 145, f_s = 190, f_s = 240, f_s = 290. "
                "For a fixed percentage of tension reinforcement, the modification "
                "factor is higher for lower f_s. All curves decrease as the "
                "percentage of tension reinforcement increases.\n\n"
                "f_s = 0.58 * f_y * (Area of cross-section of steel required / "
                "Area of cross-section of steel provided)",
        "page": 38,
        "content_type": "figure",
        "table": "Fig. 4",
        "related_table": "",
    },
    {
        # Fig. 5 -- same page family as Fig. 4, transcribed directly.
        "text": "Fig. 5 Modification Factor for Compression Reinforcement\n\n"
                "X-axis: Percentage Compression Reinforcement (0 to 3.00)\n"
                "Y-axis: Modification Factor (1.0 to 1.5)\n\n"
                "A single curve rising monotonically from a modification factor "
                "of 1.0 at 0 percent compression reinforcement up to 1.5 at 3.0 "
                "percent compression reinforcement -- unlike Fig. 4, there is "
                "only one curve (no dependence on f_s).",
        "page": 39,
        "content_type": "figure",
        "table": "Fig. 5",
        "related_table": "",
    },
    {
        # Fig. 6 -- same page as Fig. 5, transcribed directly, plus the
        # genuinely-present NOTES directly below it on the same page (the
        # VLM actually got this NOTES text right both times -- it's real
        # clause 23.2.1-adjacent content, just mis-attached to the wrong
        # figure in its output; placed here as its own correctly-labelled
        # chunk instead).
        "text": "Fig. 6 Reduction Factors for Ratios of Span to Effective Depth "
                "for Flanged Beams\n\n"
                "X-axis: Ratio of Web Width to Flange Width (0 to 1.0)\n"
                "Y-axis: Reduction Factor (0.70 to 1.00)\n\n"
                "A single curve starting near a reduction factor of 0.80 at a "
                "low web-to-flange width ratio and rising to 1.00 at a ratio of "
                "1.0 (i.e. a rectangular beam with no flange projection needs no "
                "reduction).\n\n"
                "NOTES (same page):\n"
                "1 For slabs spanning in two directions, the shorter of the two "
                "spans should be used for calculating the span to effective "
                "depth ratios.\n"
                "2 For two-way slabs of shorter spans (up to 3.5 m) with mild "
                "steel reinforcement, the span to overall depth ratios given "
                "below may generally be assumed to satisfy vertical deflection "
                "limits for loading class up to 3 kN/m2: Simply supported slabs "
                "= 35; Continuous slabs = 40. For high strength deformed bars "
                "of grade Fe 415, the values given above should be multiplied "
                "by 0.8.",
        "page": 39,
        "content_type": "figure",
        "table": "Fig. 6",
        "related_table": "",
    },
    {
        # Fig. 26 -- the Qwen2-VL run claimed the y-axis was the stability
        # index Q (confusing it with the unrelated E-2 formula elsewhere on
        # the same page) and invented an evenly-spaced 0.00-0.95 value list.
        # The real axes are beta_1/beta_2 and the real curve labels are
        # below, read directly off the chart.
        "text": "Fig. 26 Effective Length Ratios for a Column in a Frame with "
                "No Sway\n\n"
                "X-axis: beta_2, from 0 (FIXED) to 1.0 (HINGED)\n"
                "Y-axis: beta_1, from 0 (FIXED) to 1.0 (HINGED)\n\n"
                "A family of diagonal curves, each labelled with a value of the "
                "effective length ratio l_ef/l, running from 0.50 (at the "
                "fixed-fixed corner, beta_1=beta_2=0) up to 1.0 (at the "
                "hinged-hinged corner, beta_1=beta_2=1.0). Labelled curve values "
                "visible on the chart: 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, "
                "0.65, 0.675, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0.\n\n"
                "Usage: enter with beta_1 and beta_2 (the restraint stiffness "
                "ratios at each end of the column, from Note 2 below) and read "
                "off the effective length ratio l_ef/l from the nearest labelled "
                "curve, per Annex E-1.",
        "page": 92,
        "content_type": "figure",
        "table": "Fig. 26",
        "related_table": "",
    },
    {
        # Fig. 27 -- both Qwen2-VL attempts were wrong here (first invented
        # "0.95/0.85" as if there were only two curves; second produced an
        # incoherent description with no real values). Real curve labels
        # below, read directly off the chart -- note these are all >=1.0,
        # consistent with this being the sway (not-restrained) case vs.
        # Fig. 26's <=1.0 no-sway case.
        "text": "Fig. 27 Effective Length Ratios for a Column in a Frame "
                "Without Restraint Against Sway\n\n"
                "X-axis: beta_2, from 0 (FIXED) to 1.0 (HINGED)\n"
                "Y-axis: beta_1, from 0 (FIXED) to 1.0 (HINGED)\n\n"
                "A family of curves, each labelled with a value of the "
                "effective length ratio l_ef/l, running from 1.0 (at the "
                "fixed-fixed corner) up through much larger ratios toward the "
                "hinged-hinged corner. Labelled curve values visible on the "
                "chart: 1.0, 1.01, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, "
                "1.7, 1.9, 2.0, 2.5, 3.0, 5.0.\n\n"
                "Usage: same beta_1/beta_2 lookup as Fig. 26, but for the case "
                "where the column's ends are NOT restrained against relative "
                "lateral (sway) displacement -- per Annex E-1, these ratios "
                "should not be taken as less than 1.2 even where the chart "
                "would suggest a lower value very close to the fixed-fixed "
                "corner.",
        "page": 93,
        "content_type": "figure",
        "table": "Fig. 27",
        "related_table": "",
    },
    {
        # Annex E clause text (E-1 and E-3) -- this is the actual content
        # H09 and A07 were failing to retrieve (the chain from "more exact
        # analysis" via Fig. 26/27 vs "normal usage... Table 28"), not just
        # the figures themselves. E-2 (the Q stability-index formula) is
        # left out here since A06/X08 already pass on it -- it's already
        # being retrieved correctly, no need to duplicate it.
        # NOTE on E-1: the source PDF's printed text literally reads
        # "...given in Fig. 26 when relative displacement... is prevented
        # and in Fig. 26 when relative lateral displacement... is not
        # prevented" -- repeating "Fig. 26" both times. Given Fig. 26 is
        # captioned "with No Sway" and Fig. 27 "Without Restraint Against
        # Sway", the second instance is almost certainly a typo in this
        # reprint for "Fig. 27" -- corrected below. If you have a different
        # printing of IS 456:2000, it's worth a quick check against it.
        "text": "ANNEX E (Clause 25.2) EFFECTIVE LENGTH OF COLUMNS\n\n"
                "E-1 In the absence of more exact analysis, the effective "
                "length of columns in framed structures may be obtained from "
                "the ratio of effective length to unsupported length l_ef/l "
                "given in Fig. 26 when relative displacement of the ends of "
                "the column is prevented and in Fig. 27 when relative lateral "
                "displacement of the ends is not prevented. In the latter "
                "case, it is recommended that the effective length ratio "
                "l_ef/l may not be taken to be less than 1.2.\n\n"
                "E-3 For normal usage assuming idealized conditions, the "
                "effective length l_ef of a column in a given plane may be "
                "assessed on the basis of Table 28.\n\n"
                "NOTES:\n"
                "1 Figures 26 and 27 are reproduced from 'The Structural "
                "Engineer' No. 7, Volume 52, July 1974 by the permission of "
                "the Council of the Institution of Structural Engineers, UK.\n"
                "2 In Figs. 26 and 27, beta_1 and beta_2 are equal to "
                "(sum of Kc) / (sum of Kc + sum of Kb), where the summation is "
                "to be done for the members framing into a joint at top and "
                "bottom respectively, and Kc and Kb are the flexural stiffness "
                "for column and beam respectively.",
        "page": 92,
        "content_type": "text",
        "table": "Annex E",
        "related_table": "clause 25.2",
    },
    # Add more verified entries here as you confirm them missing.
    # Fix #7: every entry above now carries its own "table" key -- when you
    # add more, give each one its real table number (or omit the key/leave
    # it "" for a genuinely tableless entry) so main() below can stop
    # hardcoding every manual chunk's clause to "unknown".
]


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []
    for chunk in MANUAL_CHUNKS:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["text"])
        chunk_id = hashlib.sha256(
            (f"manual::{chunk['text']}").encode()
        ).hexdigest()[:16]

        ids.append(chunk_id)
        embeddings.append(resp["embedding"])
        documents.append(chunk["text"])
        table_num = chunk.get("table") or None  # Fix #7
        # Fix #10: table_num is either a bare number ("16", for Table
        # entries -- needs "Table " prepended to match ingest.py's
        # convention) or already a full label ("Fig. 4", for figure
        # entries -- used as-is, or "Table " would double up into
        # "Table Fig. 4"). A plain numeric-looking table_num gets the
        # prefix; anything already containing a label word doesn't.
        if table_num and not any(word in table_num for word in ("Fig.", "Annex")):
            clause = f"Table {table_num}"
        elif table_num:
            clause = table_num
        else:
            clause = "unknown"
        metadatas.append({
            "source": "sample.pdf",
            "page": chunk["page"],
            "content_type": chunk["content_type"],
            "clause": clause,
            "table": table_num or "",
            "related_table": chunk.get("related_table", ""),
            "low_confidence": False,
            "verified": True,  # marks this as hand-checked, not auto-extracted
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"Added {len(ids)} manually verified chunk(s). "
          f"Collection now has {collection.count()} total chunks.")


if __name__ == "__main__":
    main()