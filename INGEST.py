# """
# IS-Code RAG — Ingestion Pipeline (v1 / MVP)
# =============================================
# Parses PDFs (IS 456, textbooks, etc.) with Docling, chunks text along
# clause boundaries, keeps tables whole, embeds with nomic-embed-text
# via Ollama, and stores everything in a local ChromaDB collection.

# Deliberately NOT included in this v1 (add later, once this works):
#   - VLM figure captioning (needs GPU — your VM is CPU-only)
#   - Cross-page table merging
#   - SHA-256 cache/version gating (just re-run ingestion when files change)
#   - ColPali (architecturally incompatible with hybrid BM25 + metadata filtering)

# Install:
#     pip install docling chromadb ollama

# Usage:
#     Put PDFs in ./documents/, then:
#     python ingest.py
# """

# import re
# import hashlib
# from pathlib import Path

# import ollama
# import chromadb
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import TableItem, TextItem

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# # IS-code clause numbers look like: 6, 6.2, 6.2.1, 6.2.1.3 ...
# CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")

# SLIDING_WINDOW_WORDS = 500
# SLIDING_WINDOW_OVERLAP = 0.10  # 10%


# # ---------------------------------------------------------------------------
# # Text chunking — clause-aware, with sliding-window fallback
# # ---------------------------------------------------------------------------
# def chunk_text_by_clause(text: str, source: str, page: int) -> list[dict]:
#     """
#     Groups lines under their clause number (e.g. 6.2.1) into one chunk each.
#     If no clause headers are found anywhere in this block (e.g. a textbook
#     paragraph, not an IS code), falls back to a fixed-size sliding window.
#     """
#     lines = text.split("\n")
#     clause_chunks = []
#     current_clause = None
#     current_lines = []

#     def flush():
#         if current_lines:
#             clause_chunks.append({
#                 "text": "\n".join(current_lines).strip(),
#                 "clause": current_clause,
#             })

#     for line in lines:
#         match = CLAUSE_PATTERN.match(line)
#         if match:
#             flush()
#             current_clause = match.group(1)
#             current_lines = [line]
#         else:
#             current_lines.append(line)
#     flush()

#     found_any_clause = any(c["clause"] for c in clause_chunks)

#     if found_any_clause:
#         chunks = clause_chunks
#     else:
#         # Fallback: fixed-size sliding window with overlap
#         words = text.split()
#         step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
#         chunks = []
#         for i in range(0, len(words), step):
#             window = words[i:i + SLIDING_WINDOW_WORDS]
#             if not window:
#                 continue
#             chunks.append({"text": " ".join(window), "clause": None})
#             if i + SLIDING_WINDOW_WORDS >= len(words):
#                 break

#     out = []
#     for c in chunks:
#         if not c["text"].strip():
#             continue
#         out.append({
#             "text": c["text"],
#             "embed_text": c["text"],
#             "metadata": {
#                 "source": source,
#                 "page": page,
#                 "content_type": "text",
#                 "clause": c["clause"] or "unknown",
#             },
#         })
#     return out


# # ---------------------------------------------------------------------------
# # Table handling — keep full markdown, embed a cheap heuristic summary
# # ---------------------------------------------------------------------------
# def summarize_table_heuristically(markdown_table: str) -> str:
#     """Header row + first 2 data rows — cheap, no LLM call needed."""
#     rows = [r for r in markdown_table.split("\n") if r.strip()]
#     return "\n".join(rows[:4])  # header + separator + 2 data rows


# def process_table(table_item: TableItem, doc, source: str, page: int) -> dict:
#     markdown = table_item.export_to_markdown(doc)
#     summary = summarize_table_heuristically(markdown)
#     return {
#         "text": markdown,              # full table -> goes to the LLM
#         "embed_text": summary,         # short summary -> gets embedded
#         "metadata": {
#             "source": source,
#             "page": page,
#             "content_type": "table",
#             "clause": "unknown",
#         },
#     }


# # ---------------------------------------------------------------------------
# # Main per-document pipeline
# # ---------------------------------------------------------------------------
# def process_document(pdf_path: Path) -> list[dict]:
#     print(f"Parsing {pdf_path.name} ...")
#     converter = DocumentConverter()
#     result = converter.convert(str(pdf_path))
#     doc = result.document

#     chunks = []
#     text_buffer = []
#     buffer_page = None

#     def flush_text_buffer():
#         if text_buffer:
#             joined = "\n".join(text_buffer)
#             chunks.extend(chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0))
#             text_buffer.clear()

#     for item, _level in doc.iterate_items():
#         page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

#         if isinstance(item, TableItem):
#             flush_text_buffer()
#             try:
#                 chunks.append(process_table(item, doc, pdf_path.name, page_no))
#             except Exception as e:
#                 print(f"  [warn] table on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, TextItem):
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(item.text)
#             continue

#         # Pictures / other types: skipped in v1 (no GPU for captioning yet)

#     flush_text_buffer()
#     print(f"  -> {len(chunks)} chunks")
#     return chunks


# # ---------------------------------------------------------------------------
# # Embedding + storage
# # ---------------------------------------------------------------------------
# def embed_and_store(chunks: list[dict], collection):
#     ids, embeddings, documents, metadatas = [], [], [], []

#     for chunk in chunks:
#         try:
#             resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
#             vector = resp["embedding"]
#         except Exception as e:
#             print(f"  [warn] embedding failed, skipping chunk: {e}")
#             continue

#         chunk_id = hashlib.sha256(
#             (chunk["metadata"]["source"] + chunk["text"][:100] + str(chunk["metadata"]["page"])).encode()
#         ).hexdigest()[:16]

#         ids.append(chunk_id)
#         embeddings.append(vector)
#         documents.append(chunk["text"])       # full text/table -> stored for the LLM
#         metadatas.append(chunk["metadata"])

#     if ids:
#         collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"  -> stored {len(ids)} embedded chunks")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     pdfs = sorted(DOCS_DIR.glob("*.pdf"))
#     if not pdfs:
#         print(f"No PDFs found in {DOCS_DIR.resolve()} — add your trial PDF there first.")
#         return

#     for pdf_path in pdfs:
#         chunks = process_document(pdf_path)
#         embed_and_store(chunks, collection)

#     print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Ingestion Pipeline (v2)
# =============================================
# v2 changes, addressing the 6 failure modes found in trial-PDF testing:
#   1. Hierarchical table headers  — reconstructs multi-row headers (e.g.
#      Beams/Slabs/Ribs/Columns each split into Simply-supported/Continuous)
#      into unambiguous "Parent > Child" column labels, instead of trusting
#      TableFormer's flattened single-row markdown export.
#   2. Table notes                — "NOTES 1/2/3" blocks following a table
#      are now their own retrievable chunk, tagged and linked to that table,
#      instead of being silently absorbed or lost.
#   3. Formula/OCR confidence flag — text chunks with gibberish-looking
#      tokens (common OCR corruption pattern) get flagged low_confidence
#      so the LLM is told explicitly not to trust/reconstruct them.
#   4. Figure captions             — captions are now their own chunk (still
#      no VLM description — that still needs GPU — but "Fig. 5 ..." becomes
#      findable by name, which fixed most of the figure-query failures).
#   5. Duplicate table dedup       — identical tables (e.g. a dirty rescan +
#      a clean rescan of the same page) are only embedded once, so retrieval
#      mass doesn't get diluted across near-duplicate copies.
#   6. Clause sequence validation  — logs (doesn't block) suspicious jumps
#      in detected clause numbers, so you can spot-check OCR/reading-order
#      mistakes without manually reading every page at scale.

# CAVEAT: fix #1 depends on Docling's structured TableCell fields
# (start_row_offset_idx, column_header, etc.). Attribute names vary across
# Docling versions. It's wrapped in try/except and falls back to plain
# markdown export if it doesn't match your installed version — if you see
# the fallback warning firing, paste me the actual TableItem/TableData
# attributes from your version (e.g. `print(dir(table_item.data))`) and
# I'll adjust it to match, rather than guessing further.

# Install:
#     pip install docling chromadb ollama
# """

# import re
# import hashlib
# from pathlib import Path

# import ollama
# import chromadb
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import TableItem, TextItem, PictureItem

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")
# NOTE_PATTERN = re.compile(r"^\s*NOTES?\b", re.IGNORECASE)

# SLIDING_WINDOW_WORDS = 500
# SLIDING_WINDOW_OVERLAP = 0.10

# # Fix #3: crude OCR-gibberish detector — catches mixed-case+digit blobs
# # like "ooOjJc7" that showed up when a formula-heavy region OCR'd badly.
# # This is a heuristic, not a real OCR-confidence score (Tesseract/Docling
# # don't expose one at this granularity) — tune it against your own corpus
# # if it over/under-fires.
# GIBBERISH_TOKEN = re.compile(r"\b[a-z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\d[a-zA-Z0-9]*\b")


# def flag_low_confidence(text: str) -> bool:
#     return bool(GIBBERISH_TOKEN.search(text))


# # ---------------------------------------------------------------------------
# # Text chunking — clause-aware, note-aware, sliding-window fallback
# # ---------------------------------------------------------------------------
# def chunk_text_by_clause(text: str, source: str, page: int, table_context: str | None) -> list[dict]:
#     lines = text.split("\n")

#     # Fix #2: if this whole block is a NOTES block following a table,
#     # tag it distinctly and prepend the table's identity so it's
#     # retrievable by "notes on Table 16" style queries, not just by
#     # clause number (notes rarely have their own clause number).
#     if NOTE_PATTERN.match(text.strip()) and table_context:
#         note_text = text.strip()
#         return [{
#             "text": note_text,
#             "embed_text": f"Notes related to {table_context}:\n{note_text}",
#             "metadata": {
#                 "source": source, "page": page, "content_type": "table_note",
#                 "clause": "unknown", "related_table": table_context,
#                 "low_confidence": flag_low_confidence(note_text),
#             },
#         }]

#     clause_chunks = []
#     current_clause = None
#     current_lines = []

#     def flush():
#         if current_lines:
#             clause_chunks.append({"text": "\n".join(current_lines).strip(), "clause": current_clause})

#     for line in lines:
#         match = CLAUSE_PATTERN.match(line)
#         if match:
#             flush()
#             current_clause = match.group(1)
#             current_lines = [line]
#         else:
#             current_lines.append(line)
#     flush()

#     found_any_clause = any(c["clause"] for c in clause_chunks)

#     if found_any_clause:
#         chunks = clause_chunks
#     else:
#         words = text.split()
#         step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
#         chunks = []
#         for i in range(0, len(words), step):
#             window = words[i:i + SLIDING_WINDOW_WORDS]
#             if not window:
#                 continue
#             chunks.append({"text": " ".join(window), "clause": None})
#             if i + SLIDING_WINDOW_WORDS >= len(words):
#                 break

#     out = []
#     for c in chunks:
#         if not c["text"].strip():
#             continue
#         out.append({
#             "text": c["text"],
#             "embed_text": c["text"],
#             "metadata": {
#                 "source": source, "page": page, "content_type": "text",
#                 "clause": c["clause"] or "unknown",
#                 "low_confidence": flag_low_confidence(c["text"]),
#             },
#         })
#     return out


# def validate_clause_sequence(chunks: list[dict], source: str):
#     """Fix #6: logs suspicious jumps in clause numbers for manual spot-checking.
#     Doesn't block ingestion — at hundreds-of-pages scale you can't manually
#     read every page, so this narrows down where to actually look."""
#     prev = None
#     for c in chunks:
#         clause = c["metadata"].get("clause")
#         if not clause or clause == "unknown":
#             continue
#         try:
#             parts = tuple(int(x) for x in clause.split("."))
#         except ValueError:
#             continue
#         if prev is not None and parts[0] < prev[0] - 1:
#             print(f"  [validate] {source} p{c['metadata']['page']}: clause {clause} "
#                   f"comes after {'.'.join(map(str, prev))} — check reading order")
#         prev = parts


# # ---------------------------------------------------------------------------
# # Table handling
# # ---------------------------------------------------------------------------
# def export_table_hierarchical(table_item: TableItem, doc) -> str:
#     """Fix #1: reconstructs multi-row headers into 'Parent > Child' column
#     labels using Docling's structured cell data, instead of TableFormer's
#     flattened single-row markdown (which is what caused Beams/Slabs/Ribs/
#     Columns to get confused with each other)."""
#     try:
#         data = table_item.data
#         cells = data.table_cells
#         n_rows, n_cols = data.num_rows, data.num_cols

#         grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
#         header_rows = set()
#         for cell in cells:
#             for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
#                 for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
#                     grid[r][c] = cell.text
#             if getattr(cell, "column_header", False):
#                 header_rows.update(range(cell.start_row_offset_idx, cell.end_row_offset_idx))

#         if not header_rows:
#             raise ValueError("no column_header cells found in this Docling version's TableData")

#         header_idx = sorted(header_rows)
#         data_idx = [r for r in range(n_rows) if r not in header_rows]

#         col_labels = []
#         for c in range(n_cols):
#             parts = []
#             for r in header_idx:
#                 val = grid[r][c].strip()
#                 if val and (not parts or parts[-1] != val):
#                     parts.append(val)
#             col_labels.append(" > ".join(parts) if parts else f"col{c}")

#         lines = ["| " + " | ".join(col_labels) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
#         for r in data_idx:
#             lines.append("| " + " | ".join(grid[r]) + " |")
#         return "\n".join(lines)

#     except Exception as e:
#         print(f"  [warn] hierarchical table export failed ({e}) — falling back to plain markdown. "
#               f"If this fires a lot, share `dir(table_item.data)` from your Docling version.")
#         return table_item.export_to_markdown(doc)


# def summarize_table_heuristically(markdown_table: str) -> str:
#     rows = [r for r in markdown_table.split("\n") if r.strip()]
#     return "\n".join(rows[:4])


# def process_table(table_item: TableItem, doc, source: str, page: int, seen_table_hashes: set) -> dict | None:
#     markdown = export_table_hierarchical(table_item, doc)

#     # Fix #5: dedup identical/near-identical tables (e.g. your PDF's two
#     # scans of Table 2) so retrieval mass isn't split across copies.
#     normalized = re.sub(r"\s+", " ", markdown).strip().lower()
#     table_hash = hashlib.sha256(normalized.encode()).hexdigest()
#     if table_hash in seen_table_hashes:
#         print(f"  [dedup] skipping duplicate table on page {page} (matches an earlier table)")
#         return None
#     seen_table_hashes.add(table_hash)

#     summary = summarize_table_heuristically(markdown)
#     return {
#         "text": markdown,
#         "embed_text": summary,
#         "metadata": {
#             "source": source, "page": page, "content_type": "table",
#             "clause": "unknown", "low_confidence": flag_low_confidence(markdown),
#         },
#     }


# # ---------------------------------------------------------------------------
# # Figure captions (fix #4) — still no VLM description (needs GPU), but the
# # caption text alone makes "Fig. 5 ..." queries findable by name.
# # ---------------------------------------------------------------------------
# def process_picture(item: PictureItem, doc, source: str, page: int) -> dict | None:
#     caption = None
#     try:
#         caption = item.caption_text(doc)
#     except Exception:
#         try:
#             caption = " ".join(c.text for c in getattr(item, "captions", []) if hasattr(c, "text"))
#         except Exception:
#             pass

#     if not caption or not caption.strip():
#         return None

#     return {
#         "text": caption.strip(),
#         "embed_text": caption.strip(),
#         "metadata": {
#             "source": source, "page": page, "content_type": "figure_caption",
#             "clause": "unknown", "low_confidence": False,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Main per-document pipeline
# # ---------------------------------------------------------------------------
# def process_document(pdf_path: Path, seen_table_hashes: set) -> list[dict]:
#     print(f"Parsing {pdf_path.name} ...")
#     converter = DocumentConverter()
#     result = converter.convert(str(pdf_path))
#     doc = result.document

#     chunks = []
#     text_buffer = []
#     buffer_page = None
#     last_table_context = None  # Fix #2: what table was just processed, for note-linking

#     def flush_text_buffer():
#         if text_buffer:
#             joined = "\n".join(text_buffer)
#             new_chunks = chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0, last_table_context)
#             chunks.extend(new_chunks)
#             text_buffer.clear()

#     for item, _level in doc.iterate_items():
#         page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

#         if isinstance(item, TableItem):
#             flush_text_buffer()
#             try:
#                 table_chunk = process_table(item, doc, pdf_path.name, page_no, seen_table_hashes)
#                 if table_chunk:
#                     chunks.append(table_chunk)
#                     last_table_context = f"the table on {pdf_path.name} page {page_no}"
#             except Exception as e:
#                 print(f"  [warn] table on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, PictureItem):
#             flush_text_buffer()
#             try:
#                 pic_chunk = process_picture(item, doc, pdf_path.name, page_no)
#                 if pic_chunk:
#                     chunks.append(pic_chunk)
#             except Exception as e:
#                 print(f"  [warn] figure caption on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, TextItem):
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(item.text)
#             continue

#     flush_text_buffer()
#     validate_clause_sequence(chunks, pdf_path.name)
#     print(f"  -> {len(chunks)} chunks "
#           f"({sum(1 for c in chunks if c['metadata']['content_type'] == 'table')} tables, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'table_note')} table-notes, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'figure_caption')} captions, "
#           f"{sum(1 for c in chunks if c['metadata'].get('low_confidence'))} flagged low-confidence)")
#     return chunks


# # ---------------------------------------------------------------------------
# # Embedding + storage
# # ---------------------------------------------------------------------------
# def embed_and_store(chunks: list[dict], collection):
#     ids, embeddings, documents, metadatas = [], [], [], []
#     seen_ids_this_batch = {}  # base_hash -> count, so genuine duplicates get -1, -2... instead of colliding

#     for chunk in chunks:
#         try:
#             resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
#             vector = resp["embedding"]
#         except Exception as e:
#             print(f"  [warn] embedding failed, skipping chunk: {e}")
#             continue

#         # Hash the FULL text (not just a 100-char prefix) and include content_type,
#         # so a table and a caption on the same page can never collide, and two
#         # long chunks that only share their opening lines don't collide either.
#         base_hash = hashlib.sha256(
#             (chunk["metadata"]["source"] + chunk["metadata"]["content_type"]
#              + str(chunk["metadata"]["page"]) + chunk["text"]).encode()
#         ).hexdigest()[:16]

#         # Even with a full-text hash, two chunks can legitimately be byte-identical
#         # (e.g. a genuinely repeated caption) — disambiguate rather than crash.
#         n = seen_ids_this_batch.get(base_hash, 0)
#         seen_ids_this_batch[base_hash] = n + 1
#         chunk_id = f"{base_hash}-{n}" if n else base_hash

#         # Chroma metadata values must be str/int/float/bool — cast explicitly
#         metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
#                     for k, v in chunk["metadata"].items()}

#         ids.append(chunk_id)
#         embeddings.append(vector)
#         documents.append(chunk["text"])
#         metadatas.append(metadata)

#     if ids:
#         collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"  -> stored {len(ids)} embedded chunks")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     pdfs = sorted(DOCS_DIR.glob("*.pdf"))
#     if not pdfs:
#         print(f"No PDFs found in {DOCS_DIR.resolve()} — add your PDFs there first.")
#         return

#     seen_table_hashes = set()  # shared across all PDFs in this run — dedups cross-document too

#     for pdf_path in pdfs:
#         chunks = process_document(pdf_path, seen_table_hashes)
#         embed_and_store(chunks, collection)

#     print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Ingestion Pipeline (v2)
# =============================================
# v2 changes, addressing the 6 failure modes found in trial-PDF testing:
#   1. Hierarchical table headers  — reconstructs multi-row headers (e.g.
#      Beams/Slabs/Ribs/Columns each split into Simply-supported/Continuous)
#      into unambiguous "Parent > Child" column labels, instead of trusting
#      TableFormer's flattened single-row markdown export.
#   2. Table notes                — "NOTES 1/2/3" blocks following a table
#      are now their own retrievable chunk, tagged and linked to that table,
#      instead of being silently absorbed or lost.
#   3. Formula/OCR confidence flag — text chunks with gibberish-looking
#      tokens (common OCR corruption pattern) get flagged low_confidence
#      so the LLM is told explicitly not to trust/reconstruct them.
#   4. Figure captions             — captions are now their own chunk (still
#      no VLM description — that still needs GPU — but "Fig. 5 ..." becomes
#      findable by name, which fixed most of the figure-query failures).
#   5. Duplicate table dedup       — identical tables (e.g. a dirty rescan +
#      a clean rescan of the same page) are only embedded once, so retrieval
#      mass doesn't get diluted across near-duplicate copies.
#   6. Clause sequence validation  — logs (doesn't block) suspicious jumps
#      in detected clause numbers, so you can spot-check OCR/reading-order
#      mistakes without manually reading every page at scale.

# CAVEAT: fix #1 depends on Docling's structured TableCell fields
# (start_row_offset_idx, column_header, etc.). Attribute names vary across
# Docling versions. It's wrapped in try/except and falls back to plain
# markdown export if it doesn't match your installed version — if you see
# the fallback warning firing, paste me the actual TableItem/TableData
# attributes from your version (e.g. `print(dir(table_item.data))`) and
# I'll adjust it to match, rather than guessing further.

# Install:
#     pip install docling chromadb ollama
# """

# import re
# import hashlib
# from pathlib import Path

# import ollama
# import chromadb
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import TableItem, TextItem, PictureItem

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")
# NOTE_PATTERN = re.compile(r"^\s*NOTES?\b", re.IGNORECASE)

# SLIDING_WINDOW_WORDS = 500
# SLIDING_WINDOW_OVERLAP = 0.10

# # Fix #3: crude OCR-gibberish detector — catches mixed-case+digit blobs
# # like "ooOjJc7" that showed up when a formula-heavy region OCR'd badly.
# # This is a heuristic, not a real OCR-confidence score (Tesseract/Docling
# # don't expose one at this granularity) — tune it against your own corpus
# # if it over/under-fires.
# GIBBERISH_TOKEN = re.compile(r"\b[a-z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\d[a-zA-Z0-9]*\b")


# def flag_low_confidence(text: str) -> bool:
#     return bool(GIBBERISH_TOKEN.search(text))


# # ---------------------------------------------------------------------------
# # Text chunking — clause-aware, note-aware, sliding-window fallback
# # ---------------------------------------------------------------------------
# def chunk_text_by_clause(text: str, source: str, page: int, table_context: str | None) -> list[dict]:
#     lines = text.split("\n")

#     # Fix #2: if this whole block is a NOTES block following a table,
#     # tag it distinctly and prepend the table's identity so it's
#     # retrievable by "notes on Table 16" style queries, not just by
#     # clause number (notes rarely have their own clause number).
#     if NOTE_PATTERN.match(text.strip()) and table_context:
#         note_text = text.strip()
#         return [{
#             "text": note_text,
#             "embed_text": f"Notes related to {table_context}:\n{note_text}",
#             "metadata": {
#                 "source": source, "page": page, "content_type": "table_note",
#                 "clause": "unknown", "related_table": table_context,
#                 "low_confidence": flag_low_confidence(note_text),
#             },
#         }]

#     clause_chunks = []
#     current_clause = None
#     current_lines = []

#     def flush():
#         if current_lines:
#             clause_chunks.append({"text": "\n".join(current_lines).strip(), "clause": current_clause})

#     for line in lines:
#         match = CLAUSE_PATTERN.match(line)
#         if match:
#             flush()
#             current_clause = match.group(1)
#             current_lines = [line]
#         else:
#             current_lines.append(line)
#     flush()

#     found_any_clause = any(c["clause"] for c in clause_chunks)

#     if found_any_clause:
#         chunks = clause_chunks
#     else:
#         words = text.split()
#         step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
#         chunks = []
#         for i in range(0, len(words), step):
#             window = words[i:i + SLIDING_WINDOW_WORDS]
#             if not window:
#                 continue
#             chunks.append({"text": " ".join(window), "clause": None})
#             if i + SLIDING_WINDOW_WORDS >= len(words):
#                 break

#     out = []
#     for c in chunks:
#         if not c["text"].strip():
#             continue
#         out.append({
#             "text": c["text"],
#             "embed_text": c["text"],
#             "metadata": {
#                 "source": source, "page": page, "content_type": "text",
#                 "clause": c["clause"] or "unknown",
#                 "low_confidence": flag_low_confidence(c["text"]),
#             },
#         })
#     return out


# def validate_clause_sequence(chunks: list[dict], source: str):
#     """Fix #6: logs suspicious jumps in clause numbers for manual spot-checking.
#     Doesn't block ingestion — at hundreds-of-pages scale you can't manually
#     read every page, so this narrows down where to actually look."""
#     prev = None
#     for c in chunks:
#         clause = c["metadata"].get("clause")
#         if not clause or clause == "unknown":
#             continue
#         try:
#             parts = tuple(int(x) for x in clause.split("."))
#         except ValueError:
#             continue
#         if prev is not None and parts[0] < prev[0] - 1:
#             print(f"  [validate] {source} p{c['metadata']['page']}: clause {clause} "
#                   f"comes after {'.'.join(map(str, prev))} — check reading order")
#         prev = parts


# # ---------------------------------------------------------------------------
# # Table handling
# # ---------------------------------------------------------------------------
# def export_table_hierarchical(table_item: TableItem, doc) -> str:
#     """Fix #1: reconstructs multi-row headers into 'Parent > Child' column
#     labels using Docling's structured cell data, instead of TableFormer's
#     flattened single-row markdown (which is what caused Beams/Slabs/Ribs/
#     Columns to get confused with each other)."""
#     try:
#         data = table_item.data
#         cells = data.table_cells
#         n_rows, n_cols = data.num_rows, data.num_cols

#         grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
#         header_rows = set()
#         for cell in cells:
#             for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
#                 for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
#                     grid[r][c] = cell.text
#             if getattr(cell, "column_header", False):
#                 header_rows.update(range(cell.start_row_offset_idx, cell.end_row_offset_idx))

#         if not header_rows:
#             # TableFormer didn't mark any column_header cells — common on
#             # photographed/scanned tables where structure-recognition
#             # confidence is low, not necessarily a Docling version mismatch.
#             # Fall back to a positional heuristic: assume row 0 is a header,
#             # and include row 1 too if it also looks header-like (short,
#             # non-numeric cells — catches stacked headers like Table 16A's
#             # "Beams / Slabs / ..." then "Simply supported / Continuous").
#             def _looks_like_header_row(row_cells):
#                 non_empty = [c for c in row_cells if c.strip()]
#                 if not non_empty:
#                     return False
#                 numeric_like = sum(1 for c in non_empty if re.match(r"^[\d.\-±\s]+$", c))
#                 return (numeric_like / len(non_empty)) < 0.4

#             header_rows = {0}
#             if n_rows > 1 and _looks_like_header_row(grid[1]):
#                 header_rows.add(1)

#         header_idx = sorted(header_rows)
#         data_idx = [r for r in range(n_rows) if r not in header_rows]

#         col_labels = []
#         for c in range(n_cols):
#             parts = []
#             for r in header_idx:
#                 val = grid[r][c].strip()
#                 if val and (not parts or parts[-1] != val):
#                     parts.append(val)
#             col_labels.append(" > ".join(parts) if parts else f"col{c}")

#         lines = ["| " + " | ".join(col_labels) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
#         for r in data_idx:
#             lines.append("| " + " | ".join(grid[r]) + " |")
#         return "\n".join(lines)

#     except Exception as e:
#         print(f"  [warn] hierarchical table export failed ({e}) — falling back to plain markdown. "
#               f"If this fires a lot, share `dir(table_item.data)` from your Docling version.")
#         return table_item.export_to_markdown(doc)


# def summarize_table_heuristically(markdown_table: str) -> str:
#     """Header + first-column values from every row, not just the first 2 rows.
#     Long tables (like IS 456's Table 2, with 15 grade rows) were previously
#     unfindable by row-specific queries ("M25", "M60"...) because the old
#     header+2-rows summary never contained most rows' key terms at all —
#     the embedding literally didn't have the words being searched for."""
#     rows = [r for r in markdown_table.split("\n") if r.strip()]
#     if len(rows) < 3:
#         return markdown_table
#     header = rows[0]
#     data_rows = rows[2:]  # skip header + markdown separator row
#     row_keys = []
#     for r in data_rows:
#         cells = [c.strip() for c in r.strip("|").split("|")]
#         if cells and cells[0]:
#             row_keys.append(cells[0])
#     return f"{header}\nRow entries: {', '.join(row_keys[:60])}"


# def process_table(table_item: TableItem, doc, source: str, page: int, seen_table_hashes: set) -> dict | None:
#     markdown = export_table_hierarchical(table_item, doc)

#     # Fix #5: dedup identical/near-identical tables (e.g. your PDF's two
#     # scans of Table 2) so retrieval mass isn't split across copies.
#     normalized = re.sub(r"\s+", " ", markdown).strip().lower()
#     table_hash = hashlib.sha256(normalized.encode()).hexdigest()
#     if table_hash in seen_table_hashes:
#         print(f"  [dedup] skipping duplicate table on page {page} (matches an earlier table)")
#         return None
#     seen_table_hashes.add(table_hash)

#     summary = summarize_table_heuristically(markdown)
#     return {
#         "text": markdown,
#         "embed_text": summary,
#         "metadata": {
#             "source": source, "page": page, "content_type": "table",
#             "clause": "unknown", "low_confidence": flag_low_confidence(markdown),
#         },
#     }


# # ---------------------------------------------------------------------------
# # Figure captions (fix #4) — still no VLM description (needs GPU), but the
# # caption text alone makes "Fig. 5 ..." queries findable by name.
# # ---------------------------------------------------------------------------
# def process_picture(item: PictureItem, doc, source: str, page: int) -> dict | None:
#     caption = None
#     try:
#         caption = item.caption_text(doc)
#     except Exception:
#         try:
#             caption = " ".join(c.text for c in getattr(item, "captions", []) if hasattr(c, "text"))
#         except Exception:
#             pass

#     if not caption or not caption.strip():
#         return None

#     return {
#         "text": caption.strip(),
#         "embed_text": caption.strip(),
#         "metadata": {
#             "source": source, "page": page, "content_type": "figure_caption",
#             "clause": "unknown", "low_confidence": False,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Main per-document pipeline
# # ---------------------------------------------------------------------------
# def process_document(pdf_path: Path, seen_table_hashes: set) -> list[dict]:
#     print(f"Parsing {pdf_path.name} ...")
#     converter = DocumentConverter()
#     result = converter.convert(str(pdf_path))
#     doc = result.document

#     chunks = []
#     text_buffer = []
#     buffer_page = None
#     last_table_context = None  # Fix #2: what table was just processed, for note-linking

#     def flush_text_buffer():
#         if text_buffer:
#             joined = "\n".join(text_buffer)
#             new_chunks = chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0, last_table_context)
#             chunks.extend(new_chunks)
#             text_buffer.clear()

#     for item, _level in doc.iterate_items():
#         page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

#         if isinstance(item, TableItem):
#             flush_text_buffer()
#             try:
#                 table_chunk = process_table(item, doc, pdf_path.name, page_no, seen_table_hashes)
#                 if table_chunk:
#                     chunks.append(table_chunk)
#                     last_table_context = f"the table on {pdf_path.name} page {page_no}"
#             except Exception as e:
#                 print(f"  [warn] table on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, PictureItem):
#             flush_text_buffer()
#             try:
#                 pic_chunk = process_picture(item, doc, pdf_path.name, page_no)
#                 if pic_chunk:
#                     chunks.append(pic_chunk)
#             except Exception as e:
#                 print(f"  [warn] figure caption on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, TextItem):
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(item.text)
#             continue

#     flush_text_buffer()
#     validate_clause_sequence(chunks, pdf_path.name)
#     print(f"  -> {len(chunks)} chunks "
#           f"({sum(1 for c in chunks if c['metadata']['content_type'] == 'table')} tables, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'table_note')} table-notes, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'figure_caption')} captions, "
#           f"{sum(1 for c in chunks if c['metadata'].get('low_confidence'))} flagged low-confidence)")
#     return chunks


# # ---------------------------------------------------------------------------
# # Embedding + storage
# # ---------------------------------------------------------------------------
# def embed_and_store(chunks: list[dict], collection):
#     ids, embeddings, documents, metadatas = [], [], [], []
#     seen_ids_this_batch = {}  # base_hash -> count, so genuine duplicates get -1, -2... instead of colliding

#     for chunk in chunks:
#         try:
#             resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
#             vector = resp["embedding"]
#         except Exception as e:
#             print(f"  [warn] embedding failed, skipping chunk: {e}")
#             continue

#         # Hash the FULL text (not just a 100-char prefix) and include content_type,
#         # so a table and a caption on the same page can never collide, and two
#         # long chunks that only share their opening lines don't collide either.
#         base_hash = hashlib.sha256(
#             (chunk["metadata"]["source"] + chunk["metadata"]["content_type"]
#              + str(chunk["metadata"]["page"]) + chunk["text"]).encode()
#         ).hexdigest()[:16]

#         # Even with a full-text hash, two chunks can legitimately be byte-identical
#         # (e.g. a genuinely repeated caption) — disambiguate rather than crash.
#         n = seen_ids_this_batch.get(base_hash, 0)
#         seen_ids_this_batch[base_hash] = n + 1
#         chunk_id = f"{base_hash}-{n}" if n else base_hash

#         # Chroma metadata values must be str/int/float/bool — cast explicitly
#         metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
#                     for k, v in chunk["metadata"].items()}

#         ids.append(chunk_id)
#         embeddings.append(vector)
#         documents.append(chunk["text"])
#         metadatas.append(metadata)

#     if ids:
#         collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"  -> stored {len(ids)} embedded chunks")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     pdfs = sorted(DOCS_DIR.glob("*.pdf"))
#     if not pdfs:
#         print(f"No PDFs found in {DOCS_DIR.resolve()} — add your PDFs there first.")
#         return

#     seen_table_hashes = set()  # shared across all PDFs in this run — dedups cross-document too

#     for pdf_path in pdfs:
#         chunks = process_document(pdf_path, seen_table_hashes)
#         embed_and_store(chunks, collection)

#     print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Ingestion Pipeline (v2)
# =============================================
# v2 changes, addressing the 6 failure modes found in trial-PDF testing:
#   1. Hierarchical table headers  — reconstructs multi-row headers (e.g.
#      Beams/Slabs/Ribs/Columns each split into Simply-supported/Continuous)
#      into unambiguous "Parent > Child" column labels, instead of trusting
#      TableFormer's flattened single-row markdown export.
#   2. Table notes                — "NOTES 1/2/3" blocks following a table
#      are now their own retrievable chunk, tagged and linked to that table,
#      instead of being silently absorbed or lost.
#   3. Formula/OCR confidence flag — text chunks with gibberish-looking
#      tokens (common OCR corruption pattern) get flagged low_confidence
#      so the LLM is told explicitly not to trust/reconstruct them.
#   4. Figure captions             — captions are now their own chunk (still
#      no VLM description — that still needs GPU — but "Fig. 5 ..." becomes
#      findable by name, which fixed most of the figure-query failures).
#   5. Duplicate table dedup       — identical tables (e.g. a dirty rescan +
#      a clean rescan of the same page) are only embedded once, so retrieval
#      mass doesn't get diluted across near-duplicate copies.
#   6. Clause sequence validation  — logs (doesn't block) suspicious jumps
#      in detected clause numbers, so you can spot-check OCR/reading-order
#      mistakes without manually reading every page at scale.

# CAVEAT: fix #1 depends on Docling's structured TableCell fields
# (start_row_offset_idx, column_header, etc.). Attribute names vary across
# Docling versions. It's wrapped in try/except and falls back to plain
# markdown export if it doesn't match your installed version — if you see
# the fallback warning firing, paste me the actual TableItem/TableData
# attributes from your version (e.g. `print(dir(table_item.data))`) and
# I'll adjust it to match, rather than guessing further.

# Install:
#     pip install docling chromadb ollama
# """

# import re
# import hashlib
# from pathlib import Path

# import ollama
# import chromadb
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import TableItem, TextItem, PictureItem

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")
# NOTE_PATTERN = re.compile(r"^\s*NOTES?\b", re.IGNORECASE)

# SLIDING_WINDOW_WORDS = 500
# SLIDING_WINDOW_OVERLAP = 0.10

# # Fix #3: crude OCR-gibberish detector — catches mixed-case+digit blobs
# # like "ooOjJc7" that showed up when a formula-heavy region OCR'd badly.
# # This is a heuristic, not a real OCR-confidence score (Tesseract/Docling
# # don't expose one at this granularity) — tune it against your own corpus
# # if it over/under-fires.
# GIBBERISH_TOKEN = re.compile(r"\b[a-z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\d[a-zA-Z0-9]*\b")


# def flag_low_confidence(text: str) -> bool:
#     return bool(GIBBERISH_TOKEN.search(text))


# # ---------------------------------------------------------------------------
# # Text chunking — clause-aware, note-aware, sliding-window fallback
# # ---------------------------------------------------------------------------
# def chunk_text_by_clause(text: str, source: str, page: int, table_context: str | None) -> list[dict]:
#     lines = text.split("\n")

#     # Fix #2: if this whole block is a NOTES block following a table,
#     # tag it distinctly and prepend the table's identity so it's
#     # retrievable by "notes on Table 16" style queries, not just by
#     # clause number (notes rarely have their own clause number).
#     if NOTE_PATTERN.match(text.strip()) and table_context:
#         note_text = text.strip()
#         return [{
#             "text": note_text,
#             "embed_text": f"Notes related to {table_context}:\n{note_text}",
#             "metadata": {
#                 "source": source, "page": page, "content_type": "table_note",
#                 "clause": "unknown", "related_table": table_context,
#                 "low_confidence": flag_low_confidence(note_text),
#             },
#         }]

#     clause_chunks = []
#     current_clause = None
#     current_lines = []

#     def flush():
#         if current_lines:
#             clause_chunks.append({"text": "\n".join(current_lines).strip(), "clause": current_clause})

#     for line in lines:
#         match = CLAUSE_PATTERN.match(line)
#         if match:
#             flush()
#             current_clause = match.group(1)
#             current_lines = [line]
#         else:
#             current_lines.append(line)
#     flush()

#     found_any_clause = any(c["clause"] for c in clause_chunks)

#     if found_any_clause:
#         chunks = clause_chunks
#     else:
#         words = text.split()
#         step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
#         chunks = []
#         for i in range(0, len(words), step):
#             window = words[i:i + SLIDING_WINDOW_WORDS]
#             if not window:
#                 continue
#             chunks.append({"text": " ".join(window), "clause": None})
#             if i + SLIDING_WINDOW_WORDS >= len(words):
#                 break

#     out = []
#     for c in chunks:
#         if not c["text"].strip():
#             continue
#         out.append({
#             "text": c["text"],
#             "embed_text": c["text"],
#             "metadata": {
#                 "source": source, "page": page, "content_type": "text",
#                 "clause": c["clause"] or "unknown",
#                 "low_confidence": flag_low_confidence(c["text"]),
#             },
#         })
#     return out


# def validate_clause_sequence(chunks: list[dict], source: str):
#     """Fix #6: logs suspicious jumps in clause numbers for manual spot-checking.
#     Doesn't block ingestion — at hundreds-of-pages scale you can't manually
#     read every page, so this narrows down where to actually look."""
#     prev = None
#     for c in chunks:
#         clause = c["metadata"].get("clause")
#         if not clause or clause == "unknown":
#             continue
#         try:
#             parts = tuple(int(x) for x in clause.split("."))
#         except ValueError:
#             continue
#         if prev is not None and parts[0] < prev[0] - 1:
#             print(f"  [validate] {source} p{c['metadata']['page']}: clause {clause} "
#                   f"comes after {'.'.join(map(str, prev))} — check reading order")
#         prev = parts


# # ---------------------------------------------------------------------------
# # Table handling
# # ---------------------------------------------------------------------------
# def export_table_hierarchical(table_item: TableItem, doc) -> str:
#     """Fix #1: reconstructs multi-row headers into 'Parent > Child' column
#     labels using Docling's structured cell data, instead of TableFormer's
#     flattened single-row markdown (which is what caused Beams/Slabs/Ribs/
#     Columns to get confused with each other)."""
#     try:
#         data = table_item.data
#         cells = data.table_cells
#         n_rows, n_cols = data.num_rows, data.num_cols

#         grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
#         header_rows = set()
#         for cell in cells:
#             for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
#                 for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
#                     grid[r][c] = cell.text
#             if getattr(cell, "column_header", False):
#                 header_rows.update(range(cell.start_row_offset_idx, cell.end_row_offset_idx))

#         if not header_rows:
#             # TableFormer didn't mark any column_header cells — common on
#             # photographed/scanned tables where structure-recognition
#             # confidence is low, not necessarily a Docling version mismatch.
#             # Fall back to a positional heuristic: assume row 0 is a header,
#             # and include row 1 too if it also looks header-like (short,
#             # non-numeric cells — catches stacked headers like Table 16A's
#             # "Beams / Slabs / ..." then "Simply supported / Continuous").
#             def _looks_like_header_row(row_cells):
#                 non_empty = [c for c in row_cells if c.strip()]
#                 if not non_empty:
#                     return False
#                 numeric_like = sum(1 for c in non_empty if re.match(r"^[\d.\-±\s]+$", c))
#                 return (numeric_like / len(non_empty)) < 0.4

#             header_rows = {0}
#             if n_rows > 1 and _looks_like_header_row(grid[1]):
#                 header_rows.add(1)

#         header_idx = sorted(header_rows)
#         data_idx = [r for r in range(n_rows) if r not in header_rows]

#         col_labels = []
#         for c in range(n_cols):
#             parts = []
#             for r in header_idx:
#                 val = grid[r][c].strip()
#                 if val and (not parts or parts[-1] != val):
#                     parts.append(val)
#             col_labels.append(" > ".join(parts) if parts else f"col{c}")

#         lines = ["| " + " | ".join(col_labels) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
#         for r in data_idx:
#             lines.append("| " + " | ".join(grid[r]) + " |")
#         return "\n".join(lines)

#     except Exception as e:
#         print(f"  [warn] hierarchical table export failed ({e}) — falling back to plain markdown. "
#               f"If this fires a lot, share `dir(table_item.data)` from your Docling version.")
#         return table_item.export_to_markdown(doc)


# def summarize_table_heuristically(markdown_table: str) -> str:
#     """Header + first-column values from every row, not just the first 2 rows.
#     Long tables (like IS 456's Table 2, with 15 grade rows) were previously
#     unfindable by row-specific queries ("M25", "M60"...) because the old
#     header+2-rows summary never contained most rows' key terms at all —
#     the embedding literally didn't have the words being searched for."""
#     rows = [r for r in markdown_table.split("\n") if r.strip()]
#     if len(rows) < 3:
#         return markdown_table
#     header = rows[0]
#     data_rows = rows[2:]  # skip header + markdown separator row
#     row_keys = []
#     for r in data_rows:
#         cells = [c.strip() for c in r.strip("|").split("|")]
#         if cells and cells[0]:
#             row_keys.append(cells[0])
#     return f"{header}\nRow entries: {', '.join(row_keys[:60])}"


# def process_table(table_item: TableItem, doc, source: str, page: int, seen_table_hashes: set) -> dict | None:
#     markdown = export_table_hierarchical(table_item, doc)

#     # Fix #5: dedup identical/near-identical tables (e.g. your PDF's two
#     # scans of Table 2) so retrieval mass isn't split across copies.
#     normalized = re.sub(r"\s+", " ", markdown).strip().lower()
#     table_hash = hashlib.sha256(normalized.encode()).hexdigest()
#     if table_hash in seen_table_hashes:
#         print(f"  [dedup] skipping duplicate table on page {page} (matches an earlier table)")
#         return None
#     seen_table_hashes.add(table_hash)

#     summary = summarize_table_heuristically(markdown)
#     return {
#         "text": markdown,
#         "embed_text": summary,
#         "metadata": {
#             "source": source, "page": page, "content_type": "table",
#             "clause": "unknown", "low_confidence": flag_low_confidence(markdown),
#         },
#     }


# # ---------------------------------------------------------------------------
# # Figure captions (fix #4) — still no VLM description (needs GPU), but the
# # caption text alone makes "Fig. 5 ..." queries findable by name.
# # ---------------------------------------------------------------------------
# def process_picture(item: PictureItem, doc, source: str, page: int) -> dict | None:
#     caption = None
#     try:
#         caption = item.caption_text(doc)
#     except Exception:
#         try:
#             caption = " ".join(c.text for c in getattr(item, "captions", []) if hasattr(c, "text"))
#         except Exception:
#             pass

#     if not caption or not caption.strip():
#         return None

#     return {
#         "text": caption.strip(),
#         "embed_text": caption.strip(),
#         "metadata": {
#             "source": source, "page": page, "content_type": "figure_caption",
#             "clause": "unknown", "low_confidence": False,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Main per-document pipeline
# # ---------------------------------------------------------------------------
# def process_document(pdf_path: Path, seen_table_hashes: set) -> list[dict]:
#     print(f"Parsing {pdf_path.name} ...")
#     converter = DocumentConverter()
#     result = converter.convert(str(pdf_path))
#     doc = result.document

#     chunks = []
#     text_buffer = []
#     buffer_page = None
#     last_table_context = None  # Fix #2: what table was just processed, for note-linking
#     dropped_items = 0  # visibility for anything with no .text at all -- so a future
#                         # silent loss like this one shows up in the console immediately

#     def flush_text_buffer():
#         if text_buffer:
#             joined = "\n".join(text_buffer)
#             new_chunks = chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0, last_table_context)
#             chunks.extend(new_chunks)
#             text_buffer.clear()

#     for item, _level in doc.iterate_items():
#         page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

#         if isinstance(item, TableItem):
#             flush_text_buffer()
#             try:
#                 table_chunk = process_table(item, doc, pdf_path.name, page_no, seen_table_hashes)
#                 if table_chunk:
#                     chunks.append(table_chunk)
#                     last_table_context = f"the table on {pdf_path.name} page {page_no}"
#             except Exception as e:
#                 print(f"  [warn] table on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, PictureItem):
#             flush_text_buffer()
#             try:
#                 pic_chunk = process_picture(item, doc, pdf_path.name, page_no)
#                 if pic_chunk:
#                     chunks.append(pic_chunk)
#             except Exception as e:
#                 print(f"  [warn] figure caption on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, TextItem):
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(item.text)
#             continue

#         # Fix: catch-all for any other Docling item type (ListItem,
#         # SectionHeaderItem, etc.) that isn't TableItem/PictureItem/TextItem.
#         # This was the real bug behind the missing Table 16 notes: numbered
#         # "1 / 2 / 3" note lists are almost certainly classified as ListItem,
#         # not TextItem, and the old loop had no branch for that at all — the
#         # content was silently dropped before any chunking code ever ran,
#         # with no warning printed. Anything with a .text attribute now gets
#         # captured through the normal text buffer instead of vanishing.
#         text = getattr(item, "text", None)
#         if text:
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(text)
#             continue

#         dropped_items += 1

#     flush_text_buffer()
#     if dropped_items:
#         print(f"  [warn] {dropped_items} item(s) had no extractable .text and were skipped "
#               f"(check item types if this number is large)")
#     validate_clause_sequence(chunks, pdf_path.name)
#     print(f"  -> {len(chunks)} chunks "
#           f"({sum(1 for c in chunks if c['metadata']['content_type'] == 'table')} tables, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'table_note')} table-notes, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'figure_caption')} captions, "
#           f"{sum(1 for c in chunks if c['metadata'].get('low_confidence'))} flagged low-confidence)")
#     return chunks


# # ---------------------------------------------------------------------------
# # Embedding + storage
# # ---------------------------------------------------------------------------
# def embed_and_store(chunks: list[dict], collection):
#     ids, embeddings, documents, metadatas = [], [], [], []
#     seen_ids_this_batch = {}  # base_hash -> count, so genuine duplicates get -1, -2... instead of colliding

#     for chunk in chunks:
#         try:
#             resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
#             vector = resp["embedding"]
#         except Exception as e:
#             print(f"  [warn] embedding failed, skipping chunk: {e}")
#             continue

#         # Hash the FULL text (not just a 100-char prefix) and include content_type,
#         # so a table and a caption on the same page can never collide, and two
#         # long chunks that only share their opening lines don't collide either.
#         base_hash = hashlib.sha256(
#             (chunk["metadata"]["source"] + chunk["metadata"]["content_type"]
#              + str(chunk["metadata"]["page"]) + chunk["text"]).encode()
#         ).hexdigest()[:16]

#         # Even with a full-text hash, two chunks can legitimately be byte-identical
#         # (e.g. a genuinely repeated caption) — disambiguate rather than crash.
#         n = seen_ids_this_batch.get(base_hash, 0)
#         seen_ids_this_batch[base_hash] = n + 1
#         chunk_id = f"{base_hash}-{n}" if n else base_hash

#         # Chroma metadata values must be str/int/float/bool — cast explicitly
#         metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
#                     for k, v in chunk["metadata"].items()}

#         ids.append(chunk_id)
#         embeddings.append(vector)
#         documents.append(chunk["text"])
#         metadatas.append(metadata)

#     if ids:
#         collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"  -> stored {len(ids)} embedded chunks")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     pdfs = sorted(DOCS_DIR.glob("*.pdf"))
#     if not pdfs:
#         print(f"No PDFs found in {DOCS_DIR.resolve()} — add your PDFs there first.")
#         return

#     seen_table_hashes = set()  # shared across all PDFs in this run — dedups cross-document too

#     for pdf_path in pdfs:
#         chunks = process_document(pdf_path, seen_table_hashes)
#         embed_and_store(chunks, collection)

#     print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


# if __name__ == "__main__":
#     main()
# """
# IS-Code RAG — Ingestion Pipeline (v2)
# =============================================
# v2 changes, addressing the 6 failure modes found in trial-PDF testing:
#   1. Hierarchical table headers  — reconstructs multi-row headers (e.g.
#      Beams/Slabs/Ribs/Columns each split into Simply-supported/Continuous)
#      into unambiguous "Parent > Child" column labels, instead of trusting
#      TableFormer's flattened single-row markdown export.
#   2. Table notes                — "NOTES 1/2/3" blocks following a table
#      are now their own retrievable chunk, tagged and linked to that table,
#      instead of being silently absorbed or lost.
#   3. Formula/OCR confidence flag — text chunks with gibberish-looking
#      tokens (common OCR corruption pattern) get flagged low_confidence
#      so the LLM is told explicitly not to trust/reconstruct them.
#   4. Figure captions             — captions are now their own chunk (still
#      no VLM description — that still needs GPU — but "Fig. 5 ..." becomes
#      findable by name, which fixed most of the figure-query failures).
#   5. Duplicate table dedup       — identical tables (e.g. a dirty rescan +
#      a clean rescan of the same page) are only embedded once, so retrieval
#      mass doesn't get diluted across near-duplicate copies.
#   6. Clause sequence validation  — logs (doesn't block) suspicious jumps
#      in detected clause numbers, so you can spot-check OCR/reading-order
#      mistakes without manually reading every page at scale.

# CAVEAT: fix #1 depends on Docling's structured TableCell fields
# (start_row_offset_idx, column_header, etc.). Attribute names vary across
# Docling versions. It's wrapped in try/except and falls back to plain
# markdown export if it doesn't match your installed version — if you see
# the fallback warning firing, paste me the actual TableItem/TableData
# attributes from your version (e.g. `print(dir(table_item.data))`) and
# I'll adjust it to match, rather than guessing further.

# Install:
#     pip install docling chromadb ollama
# """

# import re
# import hashlib
# from pathlib import Path

# import ollama
# import chromadb
# from docling.document_converter import DocumentConverter
# from docling.datamodel.document import TableItem, TextItem, PictureItem

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
# CHROMA_DIR = "./chroma_db"
# COLLECTION_NAME = "is_codes"
# EMBED_MODEL = "nomic-embed-text"

# CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")
# NOTE_PATTERN = re.compile(r"^\s*NOTES?\b", re.IGNORECASE)

# SLIDING_WINDOW_WORDS = 500
# SLIDING_WINDOW_OVERLAP = 0.10

# # Fix #3: crude OCR-gibberish detector — catches mixed-case+digit blobs
# # like "ooOjJc7" that showed up when a formula-heavy region OCR'd badly.
# # This is a heuristic, not a real OCR-confidence score (Tesseract/Docling
# # don't expose one at this granularity) — tune it against your own corpus
# # if it over/under-fires.
# GIBBERISH_TOKEN = re.compile(r"\b[a-z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\d[a-zA-Z0-9]*\b")


# def flag_low_confidence(text: str) -> bool:
#     return bool(GIBBERISH_TOKEN.search(text))


# # ---------------------------------------------------------------------------
# # Text chunking — clause-aware, note-aware, sliding-window fallback
# # ---------------------------------------------------------------------------
# def chunk_text_by_clause(text: str, source: str, page: int, table_context: str | None) -> list[dict]:
#     lines = text.split("\n")

#     # Fix #2: if this whole block is a NOTES block following a table,
#     # tag it distinctly and prepend the table's identity so it's
#     # retrievable by "notes on Table 16" style queries, not just by
#     # clause number (notes rarely have their own clause number).
#     # Fix #2, revised: two problems in the old version, both confirmed by
#     # real ingestion output.
#     #   (a) It only fired when the block started with literal "NOTES" --
#     #       but a numbered note item can arrive as its own buffer flush
#     #       (separated from its "NOTES" header by Docling's ListItem
#     #       boundaries), so bare "1 In the designation..." never matched
#     #       and fell through to clause-regex chunking instead, where it
#     #       got misread as a bogus top-level clause "1" -- exactly what
#     #       the [validate] warnings ("clause 1 comes after 6.2.4") caught.
#     #   (b) When it DID fire, it swallowed the ENTIRE rest of the buffer
#     #       with no stopping point, so a note block could run on past its
#     #       actual end and absorb unrelated later content (confirmed via
#     #       inspect_store.py: a note chunk trailed into "8.1 General...").
#     # Fix: broaden the trigger to also catch a bare "N text" opening line;
#     # cut the note block off at the first DOTTED clause (e.g. "8.1"), since
#     # bare single digits are note items but a dotted number is a real
#     # clause; split multiple numbered items into separate chunks; and feed
#     # anything left over back through normal clause chunking instead of
#     # discarding it.
#     stripped = text.strip()
#     looks_like_notes = bool(NOTE_PATTERN.match(stripped) or re.match(r"^\s*[1-9]\s+\S", stripped))
#     if looks_like_notes and table_context:
#         lines = text.split("\n")
#         cutoff = len(lines)
#         for i, line in enumerate(lines):
#             if i == 0:
#                 continue  # don't cut on the block's own opening line
#             if re.match(r"^\s*\d{1,2}\.\d{1,3}", line):  # a real dotted clause -> note block ends here
#                 cutoff = i
#                 break
#         note_block = "\n".join(lines[:cutoff]).strip()
#         remainder = "\n".join(lines[cutoff:]).strip()

#         item_pattern = re.compile(r"(?=^\s*\d{1,2}\s+\S)", re.MULTILINE)
#         raw_items = [seg.strip() for seg in item_pattern.split(note_block) if seg.strip()]
#         if not raw_items and note_block:
#             raw_items = [note_block]

#         note_chunks = [{
#             "text": item_text,
#             "embed_text": f"Notes related to {table_context}:\n{item_text}",
#             "metadata": {
#                 "source": source, "page": page, "content_type": "table_note",
#                 "clause": "unknown", "related_table": table_context,
#                 "low_confidence": flag_low_confidence(item_text),
#             },
#         } for item_text in raw_items]

#         # Don't lose whatever comes after the note block -- chunk it
#         # normally (table_context=None so it can't re-trigger note mode).
#         remainder_chunks = chunk_text_by_clause(remainder, source, page, None) if remainder else []
#         return note_chunks + remainder_chunks

#     clause_chunks = []
#     current_clause = None
#     current_lines = []

#     def flush():
#         if current_lines:
#             clause_chunks.append({"text": "\n".join(current_lines).strip(), "clause": current_clause})

#     for line in lines:
#         match = CLAUSE_PATTERN.match(line)
#         if match:
#             flush()
#             current_clause = match.group(1)
#             current_lines = [line]
#         else:
#             current_lines.append(line)
#     flush()

#     found_any_clause = any(c["clause"] for c in clause_chunks)

#     if found_any_clause:
#         chunks = clause_chunks
#     else:
#         words = text.split()
#         step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
#         chunks = []
#         for i in range(0, len(words), step):
#             window = words[i:i + SLIDING_WINDOW_WORDS]
#             if not window:
#                 continue
#             chunks.append({"text": " ".join(window), "clause": None})
#             if i + SLIDING_WINDOW_WORDS >= len(words):
#                 break

#     out = []
#     for c in chunks:
#         if not c["text"].strip():
#             continue
#         out.append({
#             "text": c["text"],
#             "embed_text": c["text"],
#             "metadata": {
#                 "source": source, "page": page, "content_type": "text",
#                 "clause": c["clause"] or "unknown",
#                 "low_confidence": flag_low_confidence(c["text"]),
#             },
#         })
#     return out


# def validate_clause_sequence(chunks: list[dict], source: str):
#     """Fix #6: logs suspicious jumps in clause numbers for manual spot-checking.
#     Doesn't block ingestion — at hundreds-of-pages scale you can't manually
#     read every page, so this narrows down where to actually look."""
#     prev = None
#     for c in chunks:
#         clause = c["metadata"].get("clause")
#         if not clause or clause == "unknown":
#             continue
#         try:
#             parts = tuple(int(x) for x in clause.split("."))
#         except ValueError:
#             continue
#         if prev is not None and parts[0] < prev[0] - 1:
#             print(f"  [validate] {source} p{c['metadata']['page']}: clause {clause} "
#                   f"comes after {'.'.join(map(str, prev))} — check reading order")
#         prev = parts


# # ---------------------------------------------------------------------------
# # Table handling
# # ---------------------------------------------------------------------------
# def export_table_hierarchical(table_item: TableItem, doc) -> str:
#     """Fix #1: reconstructs multi-row headers into 'Parent > Child' column
#     labels using Docling's structured cell data, instead of TableFormer's
#     flattened single-row markdown (which is what caused Beams/Slabs/Ribs/
#     Columns to get confused with each other)."""
#     try:
#         data = table_item.data
#         cells = data.table_cells
#         n_rows, n_cols = data.num_rows, data.num_cols

#         grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
#         header_rows = set()
#         for cell in cells:
#             for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
#                 for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
#                     grid[r][c] = cell.text
#             if getattr(cell, "column_header", False):
#                 header_rows.update(range(cell.start_row_offset_idx, cell.end_row_offset_idx))

#         if not header_rows:
#             # TableFormer didn't mark any column_header cells — common on
#             # photographed/scanned tables where structure-recognition
#             # confidence is low, not necessarily a Docling version mismatch.
#             # Fall back to a positional heuristic: assume row 0 is a header,
#             # and include row 1 too if it also looks header-like (short,
#             # non-numeric cells — catches stacked headers like Table 16A's
#             # "Beams / Slabs / ..." then "Simply supported / Continuous").
#             def _looks_like_header_row(row_cells):
#                 non_empty = [c for c in row_cells if c.strip()]
#                 if not non_empty:
#                     return False
#                 numeric_like = sum(1 for c in non_empty if re.match(r"^[\d.\-±\s]+$", c))
#                 return (numeric_like / len(non_empty)) < 0.4

#             header_rows = {0}
#             if n_rows > 1 and _looks_like_header_row(grid[1]):
#                 header_rows.add(1)

#         header_idx = sorted(header_rows)
#         data_idx = [r for r in range(n_rows) if r not in header_rows]

#         col_labels = []
#         for c in range(n_cols):
#             parts = []
#             for r in header_idx:
#                 val = grid[r][c].strip()
#                 if val and (not parts or parts[-1] != val):
#                     parts.append(val)
#             col_labels.append(" > ".join(parts) if parts else f"col{c}")

#         lines = ["| " + " | ".join(col_labels) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
#         for r in data_idx:
#             lines.append("| " + " | ".join(grid[r]) + " |")
#         return "\n".join(lines)

#     except Exception as e:
#         print(f"  [warn] hierarchical table export failed ({e}) — falling back to plain markdown. "
#               f"If this fires a lot, share `dir(table_item.data)` from your Docling version.")
#         return table_item.export_to_markdown(doc)


# def summarize_table_heuristically(markdown_table: str) -> str:
#     """Header + first-column values from every row, not just the first 2 rows.
#     Long tables (like IS 456's Table 2, with 15 grade rows) were previously
#     unfindable by row-specific queries ("M25", "M60"...) because the old
#     header+2-rows summary never contained most rows' key terms at all —
#     the embedding literally didn't have the words being searched for."""
#     rows = [r for r in markdown_table.split("\n") if r.strip()]
#     if len(rows) < 3:
#         return markdown_table
#     header = rows[0]
#     data_rows = rows[2:]  # skip header + markdown separator row
#     row_keys = []
#     for r in data_rows:
#         cells = [c.strip() for c in r.strip("|").split("|")]
#         if cells and cells[0]:
#             row_keys.append(cells[0])
#     return f"{header}\nRow entries: {', '.join(row_keys[:60])}"


# def process_table(table_item: TableItem, doc, source: str, page: int, seen_table_hashes: set) -> dict | None:
#     markdown = export_table_hierarchical(table_item, doc)

#     # Fix #5: dedup identical/near-identical tables (e.g. your PDF's two
#     # scans of Table 2) so retrieval mass isn't split across copies.
#     normalized = re.sub(r"\s+", " ", markdown).strip().lower()
#     table_hash = hashlib.sha256(normalized.encode()).hexdigest()
#     if table_hash in seen_table_hashes:
#         print(f"  [dedup] skipping duplicate table on page {page} (matches an earlier table)")
#         return None
#     seen_table_hashes.add(table_hash)

#     summary = summarize_table_heuristically(markdown)
#     return {
#         "text": markdown,
#         "embed_text": summary,
#         "metadata": {
#             "source": source, "page": page, "content_type": "table",
#             "clause": "unknown", "low_confidence": flag_low_confidence(markdown),
#         },
#     }


# # ---------------------------------------------------------------------------
# # Figure captions (fix #4) — still no VLM description (needs GPU), but the
# # caption text alone makes "Fig. 5 ..." queries findable by name.
# # ---------------------------------------------------------------------------
# def process_picture(item: PictureItem, doc, source: str, page: int) -> dict | None:
#     caption = None
#     try:
#         caption = item.caption_text(doc)
#     except Exception:
#         try:
#             caption = " ".join(c.text for c in getattr(item, "captions", []) if hasattr(c, "text"))
#         except Exception:
#             pass

#     if not caption or not caption.strip():
#         return None

#     return {
#         "text": caption.strip(),
#         "embed_text": caption.strip(),
#         "metadata": {
#             "source": source, "page": page, "content_type": "figure_caption",
#             "clause": "unknown", "low_confidence": False,
#         },
#     }


# # ---------------------------------------------------------------------------
# # Main per-document pipeline
# # ---------------------------------------------------------------------------
# def process_document(pdf_path: Path, seen_table_hashes: set) -> list[dict]:
#     print(f"Parsing {pdf_path.name} ...")
#     converter = DocumentConverter()
#     result = converter.convert(str(pdf_path))
#     doc = result.document

#     chunks = []
#     text_buffer = []
#     buffer_page = None
#     last_table_context = None  # Fix #2: what table was just processed, for note-linking
#     dropped_items = 0  # visibility for anything with no .text at all -- so a future
#                         # silent loss like this one shows up in the console immediately

#     def flush_text_buffer():
#         if text_buffer:
#             joined = "\n".join(text_buffer)
#             new_chunks = chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0, last_table_context)
#             chunks.extend(new_chunks)
#             text_buffer.clear()

#     for item, _level in doc.iterate_items():
#         page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

#         if isinstance(item, TableItem):
#             flush_text_buffer()
#             try:
#                 table_chunk = process_table(item, doc, pdf_path.name, page_no, seen_table_hashes)
#                 if table_chunk:
#                     chunks.append(table_chunk)
#                     last_table_context = f"the table on {pdf_path.name} page {page_no}"
#             except Exception as e:
#                 print(f"  [warn] table on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, PictureItem):
#             flush_text_buffer()
#             try:
#                 pic_chunk = process_picture(item, doc, pdf_path.name, page_no)
#                 if pic_chunk:
#                     chunks.append(pic_chunk)
#             except Exception as e:
#                 print(f"  [warn] figure caption on page {page_no} failed: {e}")
#             continue

#         if isinstance(item, TextItem):
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(item.text)
#             continue

#         # Fix: catch-all for any other Docling item type (ListItem,
#         # SectionHeaderItem, etc.) that isn't TableItem/PictureItem/TextItem.
#         # This was the real bug behind the missing Table 16 notes: numbered
#         # "1 / 2 / 3" note lists are almost certainly classified as ListItem,
#         # not TextItem, and the old loop had no branch for that at all — the
#         # content was silently dropped before any chunking code ever ran,
#         # with no warning printed. Anything with a .text attribute now gets
#         # captured through the normal text buffer instead of vanishing.
#         text = getattr(item, "text", None)
#         if text:
#             if buffer_page is not None and page_no != buffer_page:
#                 flush_text_buffer()
#             buffer_page = page_no
#             text_buffer.append(text)
#             continue

#         dropped_items += 1

#     flush_text_buffer()
#     if dropped_items:
#         print(f"  [warn] {dropped_items} item(s) had no extractable .text and were skipped "
#               f"(check item types if this number is large)")
#     validate_clause_sequence(chunks, pdf_path.name)
#     print(f"  -> {len(chunks)} chunks "
#           f"({sum(1 for c in chunks if c['metadata']['content_type'] == 'table')} tables, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'table_note')} table-notes, "
#           f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'figure_caption')} captions, "
#           f"{sum(1 for c in chunks if c['metadata'].get('low_confidence'))} flagged low-confidence)")
#     return chunks


# # ---------------------------------------------------------------------------
# # Embedding + storage
# # ---------------------------------------------------------------------------
# def embed_and_store(chunks: list[dict], collection):
#     ids, embeddings, documents, metadatas = [], [], [], []
#     seen_ids_this_batch = {}  # base_hash -> count, so genuine duplicates get -1, -2... instead of colliding

#     for chunk in chunks:
#         try:
#             resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
#             vector = resp["embedding"]
#         except Exception as e:
#             print(f"  [warn] embedding failed, skipping chunk: {e}")
#             continue

#         # Hash the FULL text (not just a 100-char prefix) and include content_type,
#         # so a table and a caption on the same page can never collide, and two
#         # long chunks that only share their opening lines don't collide either.
#         base_hash = hashlib.sha256(
#             (chunk["metadata"]["source"] + chunk["metadata"]["content_type"]
#              + str(chunk["metadata"]["page"]) + chunk["text"]).encode()
#         ).hexdigest()[:16]

#         # Even with a full-text hash, two chunks can legitimately be byte-identical
#         # (e.g. a genuinely repeated caption) — disambiguate rather than crash.
#         n = seen_ids_this_batch.get(base_hash, 0)
#         seen_ids_this_batch[base_hash] = n + 1
#         chunk_id = f"{base_hash}-{n}" if n else base_hash

#         # Chroma metadata values must be str/int/float/bool — cast explicitly
#         metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
#                     for k, v in chunk["metadata"].items()}

#         ids.append(chunk_id)
#         embeddings.append(vector)
#         documents.append(chunk["text"])
#         metadatas.append(metadata)

#     if ids:
#         collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
#     print(f"  -> stored {len(ids)} embedded chunks")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def main():
#     client = chromadb.PersistentClient(path=CHROMA_DIR)
#     collection = client.get_or_create_collection(COLLECTION_NAME)

#     pdfs = sorted(DOCS_DIR.glob("*.pdf"))
#     if not pdfs:
#         print(f"No PDFs found in {DOCS_DIR.resolve()} — add your PDFs there first.")
#         return

#     seen_table_hashes = set()  # shared across all PDFs in this run — dedups cross-document too

#     for pdf_path in pdfs:
#         chunks = process_document(pdf_path, seen_table_hashes)
#         embed_and_store(chunks, collection)

#     print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


# if __name__ == "__main__":
#     main()
"""
IS-Code RAG — Ingestion Pipeline (v2)
=============================================
v2 changes, addressing the 6 failure modes found in trial-PDF testing:
  1. Hierarchical table headers  — reconstructs multi-row headers (e.g.
     Beams/Slabs/Ribs/Columns each split into Simply-supported/Continuous)
     into unambiguous "Parent > Child" column labels, instead of trusting
     TableFormer's flattened single-row markdown export.
  2. Table notes                — "NOTES 1/2/3" blocks following a table
     are now their own retrievable chunk, tagged and linked to that table,
     instead of being silently absorbed or lost.
  3. Formula/OCR confidence flag — text chunks with gibberish-looking
     tokens (common OCR corruption pattern) get flagged low_confidence
     so the LLM is told explicitly not to trust/reconstruct them.
  4. Figure captions             — captions are now their own chunk (still
     no VLM description — that still needs GPU — but "Fig. 5 ..." becomes
     findable by name, which fixed most of the figure-query failures).
  5. Duplicate table dedup       — identical tables (e.g. a dirty rescan +
     a clean rescan of the same page) are only embedded once, so retrieval
     mass doesn't get diluted across near-duplicate copies.
  6. Clause sequence validation  — logs (doesn't block) suspicious jumps
     in detected clause numbers, so you can spot-check OCR/reading-order
     mistakes without manually reading every page at scale.

CAVEAT: fix #1 depends on Docling's structured TableCell fields
(start_row_offset_idx, column_header, etc.). Attribute names vary across
Docling versions. It's wrapped in try/except and falls back to plain
markdown export if it doesn't match your installed version — if you see
the fallback warning firing, paste me the actual TableItem/TableData
attributes from your version (e.g. `print(dir(table_item.data))`) and
I'll adjust it to match, rather than guessing further.

Install:
    pip install docling chromadb ollama
"""

import re
import hashlib
from pathlib import Path

import ollama
import chromadb
from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem, TextItem, PictureItem

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCS_DIR = Path("/home/baadalvm/btp/trail/data")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "is_codes"
EMBED_MODEL = "nomic-embed-text"

CLAUSE_PATTERN = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+(.*)")
NOTE_PATTERN = re.compile(r"^\s*NOTES?\b", re.IGNORECASE)

SLIDING_WINDOW_WORDS = 500
SLIDING_WINDOW_OVERLAP = 0.10

# Fix #3: crude OCR-gibberish detector — catches mixed-case+digit blobs
# like "ooOjJc7" that showed up when a formula-heavy region OCR'd badly.
# This is a heuristic, not a real OCR-confidence score (Tesseract/Docling
# don't expose one at this granularity) — tune it against your own corpus
# if it over/under-fires.
GIBBERISH_TOKEN = re.compile(r"\b[a-z]*[A-Z][a-z]+[A-Z][a-zA-Z]*\d[a-zA-Z0-9]*\b")

# A table's own caption line (e.g. "Table 16 Nominal Cover...") -- captured
# separately so it can be fused into that table's chunk instead of getting
# lost as an unrelated preceding text chunk.
TABLE_CAPTION_PATTERN = re.compile(r"^\s*Table\s+\d+[A-Z]?\b", re.IGNORECASE)


def normalize_extraction_artifacts(text: str) -> str:
    """Confirmed via inspect_store.py: this PDF's embedded font consistently
    extracts superscript '2' (as in N/mm2) as a bare apostrophe -- seen
    identically across multiple, unrelated chunks ("415 N/mm'" three times).
    This is a narrow, evidence-based substitution, not a guess -- only the
    exact "mm'" pattern is touched, nothing else with an apostrophe."""
    return text.replace("N/mm'", "N/mm\u00b2").replace("mm'", "mm\u00b2")


def flag_low_confidence(text: str) -> bool:
    return bool(GIBBERISH_TOKEN.search(text))


# ---------------------------------------------------------------------------
# Text chunking — clause-aware, note-aware, sliding-window fallback
# ---------------------------------------------------------------------------
def chunk_text_by_clause(text: str, source: str, page: int, table_context: str | None) -> list[dict]:
    lines = text.split("\n")

    # Fix #2: if this whole block is a NOTES block following a table,
    # tag it distinctly and prepend the table's identity so it's
    # retrievable by "notes on Table 16" style queries, not just by
    # clause number (notes rarely have their own clause number).
    # Fix #2, revised: two problems in the old version, both confirmed by
    # real ingestion output.
    #   (a) It only fired when the block started with literal "NOTES" --
    #       but a numbered note item can arrive as its own buffer flush
    #       (separated from its "NOTES" header by Docling's ListItem
    #       boundaries), so bare "1 In the designation..." never matched
    #       and fell through to clause-regex chunking instead, where it
    #       got misread as a bogus top-level clause "1" -- exactly what
    #       the [validate] warnings ("clause 1 comes after 6.2.4") caught.
    #   (b) When it DID fire, it swallowed the ENTIRE rest of the buffer
    #       with no stopping point, so a note block could run on past its
    #       actual end and absorb unrelated later content (confirmed via
    #       inspect_store.py: a note chunk trailed into "8.1 General...").
    # Fix: broaden the trigger to also catch a bare "N text" opening line;
    # cut the note block off at the first DOTTED clause (e.g. "8.1"), since
    # bare single digits are note items but a dotted number is a real
    # clause; split multiple numbered items into separate chunks; and feed
    # anything left over back through normal clause chunking instead of
    # discarding it.
    stripped = text.strip()

    def _is_real_section_header(s: str) -> bool:
        # "7 WORKABILITY OF CONCRETE" is a real header wrongly caught by the
        # bare-digit pattern below (it also matches "N text"). Real headers
        # here are ALL CAPS; genuine note sentences are normal case -- use
        # that to tell them apart.
        first_line = s.split("\n", 1)[0]
        match = re.match(r"^\s*[1-9]\s+(.*)", first_line)
        if not match:
            return False
        rest = match.group(1)
        letters = [c for c in rest if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters)

    looks_like_notes = bool(
        (NOTE_PATTERN.match(stripped) or re.match(r"^\s*[1-9]\s+\S", stripped))
        and not _is_real_section_header(stripped)
    )
    if looks_like_notes and table_context:
        lines = text.split("\n")
        cutoff = len(lines)
        for i, line in enumerate(lines):
            if i == 0:
                continue  # don't cut on the block's own opening line
            if re.match(r"^\s*\d{1,2}\.\d{1,3}", line):  # a real dotted clause -> note block ends here
                cutoff = i
                break
        note_block = "\n".join(lines[:cutoff]).strip()
        remainder = "\n".join(lines[cutoff:]).strip()

        item_pattern = re.compile(r"(?=^\s*\d{1,2}\s+\S)", re.MULTILINE)
        raw_items = [seg.strip() for seg in item_pattern.split(note_block) if seg.strip()]
        if not raw_items and note_block:
            raw_items = [note_block]

        note_chunks = [{
            "text": item_text,
            "embed_text": f"Notes related to {table_context}:\n{item_text}",
            "metadata": {
                "source": source, "page": page, "content_type": "table_note",
                "clause": "unknown", "related_table": table_context,
                "low_confidence": flag_low_confidence(item_text),
            },
        } for item_text in raw_items]

        # Don't lose whatever comes after the note block -- chunk it
        # normally (table_context=None so it can't re-trigger note mode).
        remainder_chunks = chunk_text_by_clause(remainder, source, page, None) if remainder else []
        return note_chunks + remainder_chunks

    clause_chunks = []
    current_clause = None
    current_lines = []

    def flush():
        if current_lines:
            clause_chunks.append({"text": "\n".join(current_lines).strip(), "clause": current_clause})

    for line in lines:
        match = CLAUSE_PATTERN.match(line)
        if match:
            flush()
            current_clause = match.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    found_any_clause = any(c["clause"] for c in clause_chunks)

    if found_any_clause:
        chunks = clause_chunks
    else:
        words = text.split()
        step = int(SLIDING_WINDOW_WORDS * (1 - SLIDING_WINDOW_OVERLAP))
        chunks = []
        for i in range(0, len(words), step):
            window = words[i:i + SLIDING_WINDOW_WORDS]
            if not window:
                continue
            chunks.append({"text": " ".join(window), "clause": None})
            if i + SLIDING_WINDOW_WORDS >= len(words):
                break

    out = []
    for c in chunks:
        if not c["text"].strip():
            continue
        out.append({
            "text": c["text"],
            "embed_text": c["text"],
            "metadata": {
                "source": source, "page": page, "content_type": "text",
                "clause": c["clause"] or "unknown",
                "low_confidence": flag_low_confidence(c["text"]),
            },
        })
    return out


def validate_clause_sequence(chunks: list[dict], source: str):
    """Fix #6: logs suspicious jumps in clause numbers for manual spot-checking.
    Doesn't block ingestion — at hundreds-of-pages scale you can't manually
    read every page, so this narrows down where to actually look."""
    prev = None
    for c in chunks:
        clause = c["metadata"].get("clause")
        if not clause or clause == "unknown":
            continue
        try:
            parts = tuple(int(x) for x in clause.split("."))
        except ValueError:
            continue
        if prev is not None and parts[0] < prev[0] - 1:
            print(f"  [validate] {source} p{c['metadata']['page']}: clause {clause} "
                  f"comes after {'.'.join(map(str, prev))} — check reading order")
        prev = parts


# ---------------------------------------------------------------------------
# Table handling
# ---------------------------------------------------------------------------
def export_table_hierarchical(table_item: TableItem, doc) -> str:
    """Fix #1: reconstructs multi-row headers into 'Parent > Child' column
    labels using Docling's structured cell data, instead of TableFormer's
    flattened single-row markdown (which is what caused Beams/Slabs/Ribs/
    Columns to get confused with each other)."""
    try:
        data = table_item.data
        cells = data.table_cells
        n_rows, n_cols = data.num_rows, data.num_cols

        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        header_rows = set()
        for cell in cells:
            for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                    grid[r][c] = cell.text
            if getattr(cell, "column_header", False):
                header_rows.update(range(cell.start_row_offset_idx, cell.end_row_offset_idx))

        if not header_rows:
            # TableFormer didn't mark any column_header cells — common on
            # photographed/scanned tables where structure-recognition
            # confidence is low, not necessarily a Docling version mismatch.
            # Fall back to a positional heuristic: assume row 0 is a header,
            # and include row 1 too if it also looks header-like (short,
            # non-numeric cells — catches stacked headers like Table 16A's
            # "Beams / Slabs / ..." then "Simply supported / Continuous").
            def _looks_like_header_row(row_cells):
                non_empty = [c for c in row_cells if c.strip()]
                if not non_empty:
                    return False
                numeric_like = sum(1 for c in non_empty if re.match(r"^[\d.\-±\s]+$", c))
                return (numeric_like / len(non_empty)) < 0.4

            header_rows = {0}
            if n_rows > 1 and _looks_like_header_row(grid[1]):
                header_rows.add(1)

        header_idx = sorted(header_rows)
        data_idx = [r for r in range(n_rows) if r not in header_rows]

        col_labels = []
        for c in range(n_cols):
            parts = []
            for r in header_idx:
                val = grid[r][c].strip()
                if val and (not parts or parts[-1] != val):
                    parts.append(val)
            col_labels.append(" > ".join(parts) if parts else f"col{c}")

        lines = ["| " + " | ".join(col_labels) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
        for r in data_idx:
            lines.append("| " + " | ".join(grid[r]) + " |")
        return "\n".join(lines)

    except Exception as e:
        print(f"  [warn] hierarchical table export failed ({e}) — falling back to plain markdown. "
              f"If this fires a lot, share `dir(table_item.data)` from your Docling version.")
        return table_item.export_to_markdown(doc)


def summarize_table_heuristically(markdown_table: str) -> str:
    """Header + first-column values from every row, not just the first 2 rows.
    Long tables (like IS 456's Table 2, with 15 grade rows) were previously
    unfindable by row-specific queries ("M25", "M60"...) because the old
    header+2-rows summary never contained most rows' key terms at all —
    the embedding literally didn't have the words being searched for."""
    rows = [r for r in markdown_table.split("\n") if r.strip()]
    if len(rows) < 3:
        return markdown_table
    header = rows[0]
    data_rows = rows[2:]  # skip header + markdown separator row
    row_keys = []
    for r in data_rows:
        cells = [c.strip() for c in r.strip("|").split("|")]
        if cells and cells[0]:
            row_keys.append(cells[0])
    return f"{header}\nRow entries: {', '.join(row_keys[:60])}"


def process_table(table_item: TableItem, doc, source: str, page: int, seen_table_hashes: set,
                   caption: str | None = None) -> dict | None:
    markdown = export_table_hierarchical(table_item, doc)

    # Fix: fuse the table's caption (e.g. "Table 16 Nominal Cover...") into
    # its own chunk. Confirmed via inspect_store.py that the real, correctly
    # extracted Table 16 chunk had no identifying text at all -- Docling
    # captures the caption as a separate preceding TextItem, so a query
    # naming "Table 16" had nothing to match against, and at 790-chunk scale
    # it lost to other tables that happen to share keywords like "severe".
    if caption:
        markdown = f"{caption}\n\n{markdown}"

    # Fix #5: dedup identical/near-identical tables (e.g. your PDF's two
    # scans of Table 2) so retrieval mass isn't split across copies.
    normalized = re.sub(r"\s+", " ", markdown).strip().lower()
    table_hash = hashlib.sha256(normalized.encode()).hexdigest()
    if table_hash in seen_table_hashes:
        print(f"  [dedup] skipping duplicate table on page {page} (matches an earlier table)")
        return None
    seen_table_hashes.add(table_hash)

    summary = summarize_table_heuristically(markdown)
    return {
        "text": markdown,
        "embed_text": summary,
        "metadata": {
            "source": source, "page": page, "content_type": "table",
            "clause": "unknown", "low_confidence": flag_low_confidence(markdown),
        },
    }


# ---------------------------------------------------------------------------
# Figure captions (fix #4) — still no VLM description (needs GPU), but the
# caption text alone makes "Fig. 5 ..." queries findable by name.
# ---------------------------------------------------------------------------
def process_picture(item: PictureItem, doc, source: str, page: int) -> dict | None:
    caption = None
    try:
        caption = item.caption_text(doc)
    except Exception:
        try:
            caption = " ".join(c.text for c in getattr(item, "captions", []) if hasattr(c, "text"))
        except Exception:
            pass

    if not caption or not caption.strip():
        return None

    return {
        "text": caption.strip(),
        "embed_text": caption.strip(),
        "metadata": {
            "source": source, "page": page, "content_type": "figure_caption",
            "clause": "unknown", "low_confidence": False,
        },
    }


# ---------------------------------------------------------------------------
# Main per-document pipeline
# ---------------------------------------------------------------------------
def process_document(pdf_path: Path, seen_table_hashes: set) -> list[dict]:
    print(f"Parsing {pdf_path.name} ...")
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    chunks = []
    text_buffer = []
    buffer_page = None
    last_table_context = None  # Fix #2: what table was just processed, for note-linking
    dropped_items = 0  # visibility for anything with no .text at all -- so a future
                        # silent loss like this one shows up in the console immediately

    def flush_text_buffer():
        if text_buffer:
            joined = "\n".join(text_buffer)
            new_chunks = chunk_text_by_clause(joined, pdf_path.name, buffer_page or 0, last_table_context)
            chunks.extend(new_chunks)
            text_buffer.clear()

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else buffer_page

        if isinstance(item, TableItem):
            # Peek at the buffer for a caption line ("Table 16 ...") before
            # flushing -- if found, pull it out so it fuses into the table's
            # own chunk instead of vanishing into an unrelated text chunk.
            caption = None
            if text_buffer and TABLE_CAPTION_PATTERN.match(text_buffer[-1].strip()):
                caption = text_buffer.pop().strip()
            flush_text_buffer()
            try:
                table_chunk = process_table(item, doc, pdf_path.name, page_no, seen_table_hashes, caption)
                if table_chunk:
                    chunks.append(table_chunk)
                    last_table_context = f"the table on {pdf_path.name} page {page_no}" + (f" ({caption})" if caption else "")
            except Exception as e:
                print(f"  [warn] table on page {page_no} failed: {e}")
            continue

        if isinstance(item, PictureItem):
            flush_text_buffer()
            try:
                pic_chunk = process_picture(item, doc, pdf_path.name, page_no)
                if pic_chunk:
                    chunks.append(pic_chunk)
            except Exception as e:
                print(f"  [warn] figure caption on page {page_no} failed: {e}")
            continue

        if isinstance(item, TextItem):
            if buffer_page is not None and page_no != buffer_page:
                flush_text_buffer()
            buffer_page = page_no
            text_buffer.append(normalize_extraction_artifacts(item.text))
            continue

        # Fix: catch-all for any other Docling item type (ListItem,
        # SectionHeaderItem, etc.) that isn't TableItem/PictureItem/TextItem.
        # This was the real bug behind the missing Table 16 notes: numbered
        # "1 / 2 / 3" note lists are almost certainly classified as ListItem,
        # not TextItem, and the old loop had no branch for that at all — the
        # content was silently dropped before any chunking code ever ran,
        # with no warning printed. Anything with a .text attribute now gets
        # captured through the normal text buffer instead of vanishing.
        text = getattr(item, "text", None)
        if text:
            if buffer_page is not None and page_no != buffer_page:
                flush_text_buffer()
            buffer_page = page_no
            text_buffer.append(normalize_extraction_artifacts(text))
            continue

        dropped_items += 1

    flush_text_buffer()
    if dropped_items:
        print(f"  [warn] {dropped_items} item(s) had no extractable .text and were skipped "
              f"(check item types if this number is large)")
    validate_clause_sequence(chunks, pdf_path.name)
    print(f"  -> {len(chunks)} chunks "
          f"({sum(1 for c in chunks if c['metadata']['content_type'] == 'table')} tables, "
          f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'table_note')} table-notes, "
          f"{sum(1 for c in chunks if c['metadata']['content_type'] == 'figure_caption')} captions, "
          f"{sum(1 for c in chunks if c['metadata'].get('low_confidence'))} flagged low-confidence)")
    return chunks


# ---------------------------------------------------------------------------
# Embedding + storage
# ---------------------------------------------------------------------------
def embed_and_store(chunks: list[dict], collection):
    ids, embeddings, documents, metadatas = [], [], [], []
    seen_ids_this_batch = {}  # base_hash -> count, so genuine duplicates get -1, -2... instead of colliding

    for chunk in chunks:
        try:
            resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk["embed_text"])
            vector = resp["embedding"]
        except Exception as e:
            print(f"  [warn] embedding failed, skipping chunk: {e}")
            continue

        # Hash the FULL text (not just a 100-char prefix) and include content_type,
        # so a table and a caption on the same page can never collide, and two
        # long chunks that only share their opening lines don't collide either.
        base_hash = hashlib.sha256(
            (chunk["metadata"]["source"] + chunk["metadata"]["content_type"]
             + str(chunk["metadata"]["page"]) + chunk["text"]).encode()
        ).hexdigest()[:16]

        # Even with a full-text hash, two chunks can legitimately be byte-identical
        # (e.g. a genuinely repeated caption) — disambiguate rather than crash.
        n = seen_ids_this_batch.get(base_hash, 0)
        seen_ids_this_batch[base_hash] = n + 1
        chunk_id = f"{base_hash}-{n}" if n else base_hash

        # Chroma metadata values must be str/int/float/bool — cast explicitly
        metadata = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                    for k, v in chunk["metadata"].items()}

        ids.append(chunk_id)
        embeddings.append(vector)
        documents.append(chunk["text"])
        metadatas.append(metadata)

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"  -> stored {len(ids)} embedded chunks")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR.resolve()} — add your PDFs there first.")
        return

    seen_table_hashes = set()  # shared across all PDFs in this run — dedups cross-document too

    for pdf_path in pdfs:
        chunks = process_document(pdf_path, seen_table_hashes)
        embed_and_store(chunks, collection)

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()