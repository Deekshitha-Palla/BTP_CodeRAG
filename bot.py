import os
import re
import io
import pickle
import hashlib
import shutil

import ollama
import chromadb
import fitz  # PyMuPDF — used for cropping table/figure regions and OCR fallback
import pytesseract
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma4:e2b"
VLM_MODEL = "gemma4:e2b"   # only used for figures/diagrams, not tables or plain text

# One-time diagnostic dump of a real picture item's attributes -- leave False
# unless you need to re-verify the Docling caption API again.
DEBUG_DUMP_PICTURE_ATTRS = False

CLAUSE_HEADER_REGEX = r"^\s*(\d{1,2}(?:\.\d{1,3}){0,4})\s+[A-Z]"

VERIFIED_TABLES = []


def build_verified_chunks(verified_tables=VERIFIED_TABLES):
    chunks = []
    for i, vt in enumerate(verified_tables):
        chunks.append({
            "text": vt["text"],
            "embed_text": vt["text"],
            "chunk_type": "table",
            "page_num": vt["page_num"],
            "source_file": vt["source_file"],
            "extraction_method": "verified",
            "chunk_id": f"verified_{i:03d}",
            "_keywords": vt["keywords"],
        })
    return chunks


def force_include_verified(query, chunks, candidate_ids):
    query_lower = query.lower()
    forced = []
    for c in chunks:
        if c.get("extraction_method") == "verified":
            if any(kw.lower() in query_lower for kw in c.get("_keywords", [])):
                forced.append(c["chunk_id"])
    return forced + [cid for cid in candidate_ids if cid not in forced]


# --------------------- 1. DATA INGESTION + CHUNKING -------------------------

HEADER_RE = re.compile(CLAUSE_HEADER_REGEX, re.MULTILINE)

FIGURE_PROMPT = """You are looking at a figure extracted from a technical document. It \
could be an engineering diagram, a biological/scientific illustration, a chart, a \
flowchart, or anything else -- you do not know the subject matter in advance.

Provide:
1. FIGURE TYPE: name what kind of figure this actually is, based only on what you can \
   see in the image. Do not default to engineering terminology (e.g. "interaction \
   diagram", "moment-curvature curve") unless the figure genuinely shows that kind of \
   engineering data (load/stress/moment values on labeled engineering axes). A diagram \
   with boxes and arrows is not automatically an "engineering diagram" -- describe what \
   it actually depicts (e.g. "neural feedback loop", "biological process flowchart").
2. AXES / VARIABLES / LABELS: what each axis, curve, or key label represents, with units if visible.
3. KEY VALUES: any numerical values, curve labels, or annotated points visible on the figure.
4. INTERPRETATION: 2-4 sentences explaining what this figure is used for and how a reader \
   would apply it, in whatever domain it actually belongs to.

If any part is illegible, say so explicitly rather than guessing. Do not invent \
numbers you cannot actually read."""


def _get_page_no(item):
    prov = getattr(item, "prov", None)
    if prov:
        first = prov[0] if isinstance(prov, list) else prov
        return getattr(first, "page_no", 0)
    return 0


def _get_bbox(item):
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    return getattr(first, "bbox", None)


def _bbox_to_fitz_rect(bbox, page_width, page_height):
    """Robust to both coordinate-origin mismatch and genuinely bad/degenerate
    detections on noisy scanned pages -- tries multiple interpretations,
    intersects each against the real page rect, and returns the first valid
    non-degenerate result, or None if nothing works (caller skips the figure
    instead of crashing)."""
    l, t, r, b = bbox.l, bbox.t, bbox.r, bbox.b
    page_rect = fitz.Rect(0, 0, page_width, page_height)

    candidates = []
    if hasattr(bbox, "to_top_left_origin"):
        try:
            conv = bbox.to_top_left_origin(page_height)
            candidates.append((conv.l, conv.t, conv.r, conv.b))
        except Exception:
            pass
    candidates.append((l, t, r, b))                              # as-is
    candidates.append((l, page_height - t, r, page_height - b))  # manual bottom-left flip

    MIN_CROP_SIZE = 4
    for (x0, y0, x1, y1) in candidates:
        x0n, x1n = min(x0, x1), max(x0, x1)
        y0n, y1n = min(y0, y1), max(y0, y1)
        rect = fitz.Rect(x0n, y0n, x1n, y1n) & page_rect
        if rect.width >= MIN_CROP_SIZE and rect.height >= MIN_CROP_SIZE:
            return rect
    return None


def _crop_region(doc_fitz, pdf_path, page_no, bbox, crop_dir, zoom=3.0):
    page = doc_fitz[page_no - 1] if page_no > 0 else doc_fitz[0]
    rect = _bbox_to_fitz_rect(bbox, page.rect.width, page.rect.height)
    if rect is None:
        print(f"    Warning: no valid crop region for a figure on page {page_no}, skipping.")
        return None
    mat = fitz.Matrix(zoom, zoom)
    try:
        pix = page.get_pixmap(matrix=mat, clip=rect)
        if pix.width < 1 or pix.height < 1:
            print(f"    Warning: zero-size pixmap on page {page_no}, skipping this figure crop.")
            return None
    except Exception as e:
        print(f"    Warning: failed to rasterize figure on page {page_no}: {e}")
        return None
    out_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_no}_{int(rect.x0)}_{int(rect.y0)}.png"
    out_path = os.path.join(crop_dir, out_name)
    pix.save(out_path)
    return out_path


def _get_figure_caption(item, doc):
    """Ground-truth-confirmed: item.caption_text is a BOUND METHOD, not a
    string attribute -- it must be called with `doc` to resolve the
    caption's RefItem(s) into real text. (Confirmed against a real Docling
    debug dump; the earlier guess treated it as a plain attribute, which
    silently corrupted figure chunks with a raw object repr.)"""
    cap_method = getattr(item, "caption_text", None)
    if callable(cap_method):
        try:
            resolved = cap_method(doc)
            if resolved:
                return resolved
        except Exception:
            pass
    elif cap_method:
        return cap_method

    cap_refs = getattr(item, "captions", None)
    if cap_refs:
        try:
            texts = []
            for ref in cap_refs:
                resolved = ref.resolve(doc) if hasattr(ref, "resolve") else ref
                t = getattr(resolved, "text", None)
                if t:
                    texts.append(t)
            if texts:
                return " ".join(texts)
        except Exception:
            pass
    return ""


def _load_and_resize_for_vlm(image_path, max_dimension=1024):
    """Downscales the crop before sending it to the VLM -- Qwen2.5-VL
    resizes internally to its own patch grid anyway, so this cuts
    encode/transfer/vision-token overhead without losing signal."""
    img = Image.open(image_path)
    img.thumbnail((max_dimension, max_dimension))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def interpret_figure_with_vlm(image_path, caption="", max_dimension=1024, max_output_tokens=350):
    import base64
    prompt = FIGURE_PROMPT
    if caption:
        prompt += f"\n\nThe document's own caption for this figure is: \"{caption}\""

    image_bytes = _load_and_resize_for_vlm(image_path, max_dimension=max_dimension)

    response = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(image_bytes).decode("utf-8")],
        }],
        options={"temperature": 0.1, "num_predict": max_output_tokens},
        keep_alive="10m",
    )
    return response["message"]["content"].strip()


def _summarize_table_heuristically(markdown_table):
    lines = [l for l in markdown_table.strip().splitlines() if l.strip()]
    if not lines:
        return "Table (empty)"
    header = lines[0]
    sample_rows = lines[2:4] if len(lines) > 2 else []
    return f"Table with columns: {header}. Sample data: {' | '.join(sample_rows)}"


def _table_looks_incomplete(markdown_table):
    tail = markdown_table.strip().splitlines()[-3:]
    tail_text = " ".join(tail).lower()
    return not any(k in tail_text for k in ("total", "note", "*", "source:"))


def _table_looks_headerless(markdown_table):
    lines = markdown_table.strip().splitlines()
    if not lines:
        return False
    first_row = lines[0]
    cells = [c.strip() for c in first_row.strip("|").split("|")]
    return all(re.match(r"^-?\d", c) for c in cells if c)


def _chunk_text_block(text, page_no, source_file, chunk_counter, max_words=250):
    chunks = []
    headers = list(HEADER_RE.finditer(text))

    if headers:
        boundaries = [m.start() for m in headers] + [len(text)]
        for i in range(len(headers)):
            segment = text[boundaries[i]:boundaries[i + 1]].strip()
            if not segment:
                continue
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter[0]:05d}",
                "text": segment,
                "embed_text": segment,
                "chunk_type": "text",
                "page_num": page_no,
                "source_file": source_file,
                "section_hint": headers[i].group(1),
                "extraction_method": "docling-text",
            })
            chunk_counter[0] += 1
        return chunks

    words = text.split()
    for start in range(0, len(words), max_words):
        segment = " ".join(words[start:start + max_words])
        if not segment.strip():
            continue
        chunks.append({
            "chunk_id": f"chunk_{chunk_counter[0]:05d}",
            "text": segment,
            "embed_text": segment,
            "chunk_type": "text",
            "page_num": page_no,
            "source_file": source_file,
            "section_hint": "",
            "extraction_method": "docling-text",
        })
        chunk_counter[0] += 1
    return chunks


def ingest_and_chunk(pdf_path, ocr_dpi=300, crop_dir="./crops"):
    os.makedirs(crop_dir, exist_ok=True)
    from docling.document_converter import DocumentConverter

    source_name = os.path.basename(pdf_path)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    doc_fitz = fitz.open(pdf_path)

    chunks = []
    chunk_counter = [0]
    pending_table_tail = None
    pages_with_content = set()
    figures_done = 0
    _picture_attrs_dumped = [False]
    import time as _time
    ingest_start = _time.time()

    for item, _level in doc.iterate_items():
        page_no = None
        try:
            item_type = type(item).__name__.lower()
            page_no = _get_page_no(item)
            bbox = _get_bbox(item)
            if page_no:
                pages_with_content.add(page_no)

            if "table" in item_type:
                md = item.export_to_markdown(doc) if hasattr(item, "export_to_markdown") else str(item)

                if (pending_table_tail is not None
                        and _table_looks_headerless(md)
                        and page_no == pending_table_tail["page_num"] + 1):
                    pending_table_tail["text"] += "\n" + md
                    pending_table_tail["embed_text"] = _summarize_table_heuristically(pending_table_tail["text"])
                    if not _table_looks_incomplete(md):
                        pending_table_tail = None
                    continue

                table_chunk = {
                    "chunk_id": f"chunk_{chunk_counter[0]:05d}",
                    "text": md,
                    "embed_text": _summarize_table_heuristically(md),
                    "chunk_type": "table",
                    "page_num": page_no,
                    "source_file": source_name,
                    "section_hint": "",
                    "extraction_method": "docling-tableformer",
                }
                chunk_counter[0] += 1
                chunks.append(table_chunk)
                pending_table_tail = table_chunk if _table_looks_incomplete(md) else None

            elif "picture" in item_type or "figure" in item_type:
                if not bbox:
                    continue
                crop_path = _crop_region(doc_fitz, pdf_path, page_no, bbox, crop_dir)
                if crop_path is None:
                    continue

                if DEBUG_DUMP_PICTURE_ATTRS and not _picture_attrs_dumped[0]:
                    print("\n    --- DEBUG: one-time dump of a real picture item's attributes ---")
                    print(f"    type(item) = {type(item)}")
                    print(f"    dir(item)  = {[a for a in dir(item) if not a.startswith('_')]}")
                    for attr_name in ("caption_text", "captions", "caption", "annotations"):
                        val = getattr(item, attr_name, "<no such attribute>")
                        print(f"    item.{attr_name} = {val!r}")
                    print("    --- end debug dump ---\n")
                    _picture_attrs_dumped[0] = True

                caption = _get_figure_caption(item, doc)
                figures_done += 1
                fig_start = _time.time()
                print(f"    Interpreting figure {figures_done} (page {page_no})...", end=" ", flush=True)
                try:
                    interpretation = interpret_figure_with_vlm(crop_path, caption)
                except Exception as e:
                    print(f"FAILED after {_time.time() - fig_start:.1f}s: {e}")
                    continue
                print(f"done in {_time.time() - fig_start:.1f}s")

                full_text = f"Caption: {caption}\n\n{interpretation}" if caption else interpretation

                # Guard against an empty/whitespace-only result (e.g. VLM
                # returned nothing usable and there was no caption either).
                # An empty embed_text later causes Ollama to return an empty
                # embedding vector, which crashes ChromaDB's add() -- much
                # cheaper to catch it here than downstream.
                if not full_text or not full_text.strip():
                    full_text = f"[Figure on page {page_no} -- no readable caption or VLM interpretation available]"
                    print(f"    Warning: figure on page {page_no} produced no usable text "
                          f"(empty caption AND empty VLM interpretation) -- using a placeholder "
                          f"so it stays embeddable, but it will not be useful for retrieval.")

                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter[0]:05d}",
                    "text": full_text,
                    "embed_text": full_text,
                    "chunk_type": "figure",
                    "page_num": page_no,
                    "source_file": source_name,
                    "section_hint": "",
                    "extraction_method": "vlm-figure",
                })
                chunk_counter[0] += 1

            else:
                text = getattr(item, "text", "") or ""
                if text.strip():
                    chunks.extend(_chunk_text_block(text, page_no, source_name, chunk_counter))

        except Exception as e:
            print(f"    Warning: skipping one element on page {page_no or '?'} "
                  f"due to unexpected error: {e}")
            continue

    for page_no in range(1, doc_fitz.page_count + 1):
        if page_no in pages_with_content:
            continue
        page = doc_fitz[page_no - 1]
        pix = page.get_pixmap(dpi=ocr_dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img).strip()
        if ocr_text:
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter[0]:05d}",
                "text": ocr_text,
                "embed_text": ocr_text,
                "chunk_type": "text",
                "page_num": page_no,
                "source_file": source_name,
                "section_hint": "",
                "extraction_method": "ocr-fallback",
            })
            chunk_counter[0] += 1
        else:
            print(f"  Warning: page {page_no} produced no text (docling or OCR) "
                  f"— likely blank or unreadable.")

    doc_fitz.close()
    elapsed = _time.time() - ingest_start
    print(f"  -> Ingestion took {elapsed:.1f}s total "
          f"({figures_done} figure(s) sent to VLM; that step is almost always "
          f"the bulk of this number).")
    return chunks


# ------------------------- DENSE (EMBEDDING) SEARCH --------------------------

def generate_embeddings(chunks, query, collection_name="structural_analysis",
                         persist_dir="./chroma_store", max_retries=3):
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(collection_name)
        existing_ids = set(collection.get()["ids"])
    except Exception:
        collection = client.create_collection(collection_name)
        existing_ids = set()

    to_embed = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  {len(existing_ids)} chunks already embedded, {len(to_embed)} remaining.")

    skipped_chunks = []

    for i, c in enumerate(to_embed, start=1):
        text_to_embed = c["embed_text"]
        if not text_to_embed or not text_to_embed.strip():
            # Should be rare now that ingest_and_chunk guards against this,
            # but stay defensive here too -- this function gets called on
            # cached chunks from OLDER ingestion runs too, which may predate
            # that guard.
            text_to_embed = f"[{c.get('chunk_type', 'unknown')} chunk on page {c.get('page_num', '?')}, no content]"

        embedded_successfully = False
        for attempt in range(1, max_retries + 1):
            try:
                emb = ollama.embeddings(model=EMBED_MODEL, prompt=text_to_embed)["embedding"]
                if not emb:
                    # Ollama returned 200 OK but an empty vector -- this is
                    # exactly what crashed ChromaDB's add() with
                    # "IndexError: list index out of range" before. Treat it
                    # as a retryable failure rather than letting it reach
                    # collection.add() at all.
                    raise ValueError("Ollama returned an empty embedding vector for this text.")
                collection.add(ids=[c["chunk_id"]], embeddings=[emb], documents=[text_to_embed])
                embedded_successfully = True
                break
            except Exception as e:
                if "context length" in str(e).lower() and attempt < max_retries:
                    text_to_embed = text_to_embed[: len(text_to_embed) // 2]
                    print(f"    {c['chunk_id']} too long, truncating and retrying "
                          f"(new length {len(text_to_embed)})...")
                    continue
                print(f"    Retry {attempt}/{max_retries} for {c['chunk_id']} after error: {e}")

        if not embedded_successfully:
            print(f"    Warning: giving up on {c['chunk_id']} after {max_retries} attempts -- "
                  f"SKIPPING this chunk (it will not be searchable this session). "
                  f"Continuing with the rest of the corpus rather than aborting the whole run.")
            skipped_chunks.append(c["chunk_id"])
            continue

        if i % 50 == 0 or i == len(to_embed):
            print(f"  Embedded {i}/{len(to_embed)} new chunks...")

    if skipped_chunks:
        print(f"  -> {len(skipped_chunks)} chunk(s) could not be embedded and were skipped: "
              f"{skipped_chunks}")

    q_embed = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    n_results = min(50, max(1, len(chunks) - len(skipped_chunks)))
    result = collection.query(query_embeddings=[q_embed], n_results=n_results)
    return result["ids"][0]


# --------------------------- SPARSE (BM25) SEARCH -----------------------------

def bm25_search(chunks, query):
    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["embed_text"] for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked]


# ------------------------------- RRF FUSION -----------------------------------

def reciprocal_rank_fusion(ranked_lists, k=60):
    score_chunk = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            score_chunk[chunk_id] = score_chunk.get(chunk_id, 0.0) + (1.0 / (rank + k + 1))
    fused = sorted(score_chunk.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


# ---------------------------- METADATA FILTERING ------------------------------

def infer_metadata_filter(query):
    q = query.lower()
    if any(kw in q for kw in ("table", "value of", "look up", "coefficient for")):
        return "table"
    if any(kw in q for kw in ("diagram", "graph", "curve", "figure", "chart")):
        return "figure"
    return None


def metadata_filtering(fused_ids, chunks, source_file=None, page_range=None, chunk_type=None):
    lookup = {c["chunk_id"]: c for c in chunks}
    filtered = []
    for chunk_id in fused_ids:
        c = lookup.get(chunk_id)
        if c is None:
            continue  # chunk may have been skipped during embedding
        if source_file and c["source_file"] != source_file:
            continue
        if page_range and not (page_range[0] <= c["page_num"] <= page_range[1]):
            continue
        if chunk_type and c.get("chunk_type") != chunk_type:
            continue
        filtered.append(chunk_id)
    return filtered


# -------------------------------- RERANKING -----------------------------------

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def reranking(candidate_ids, chunks, query, top_k=5):
    lookup = {c["chunk_id"]: c for c in chunks}
    candidates = [lookup[cid] for cid in candidate_ids if cid in lookup]
    if not candidates:
        return []
    model = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_k]]


    print("\n===== RERANKER SCORES =====")
    for rank, (chunk, score) in enumerate(ranked, start=1):
        print(
            f"Rank {rank:2d} | "
            f"Score {score:8.4f} | "
            f"ID {chunk['chunk_id']} | "
            f"Page {chunk['page_num']} | "
            f"Type {chunk['chunk_type']}"
        )

    return [chunk for chunk, score in ranked[:top_k]]

# ------------------------------- Agentic Loop ----------------------------------

def agent(query, reranked_chunks):
    if not reranked_chunks:
        return "No relevant content was retrieved for this question."
    label_map = {"text": "TEXT", "table": "TABLE", "figure": "FIGURE (visual interpretation)"}
    context = "\n\n".join(
        f"[{label_map.get(c.get('chunk_type'), 'EXCERPT')} — Page {c['page_num']}, {c['source_file']}]\n{c['text']}"
        for c in reranked_chunks
    )
    prompt = (
        "Answer the question using ONLY the context below. Cite page numbers. "
        "Treat TABLE excerpts as exact structured data — quote values precisely. "
        "Treat FIGURE excerpts as visually estimated, not verified fact — flag when a "
        "number came from a figure reading. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    response = ollama.generate(model=GEN_MODEL, prompt=prompt)
    return response["response"]


EXTRACTION_LOGIC_VERSION = "docling-tableformer-vlm-v6-empty-embedding-guard"


def compute_source_fingerprint(pdf_path, logic_version=EXTRACTION_LOGIC_VERSION):
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        hasher.update(f.read())
    hasher.update(logic_version.encode())
    return hasher.hexdigest()


if __name__ == "__main__":
    print(f"=== chatbot_v3.py — build: {EXTRACTION_LOGIC_VERSION} ===\n")
    path = r"/home/baadalvm/btp/trail/data/sample.pdf"
    chunks_cache_path = "./chunks_cache.pkl"

    current_fingerprint = compute_source_fingerprint(path)
    cache_valid = False

    if os.path.exists(chunks_cache_path):
        with open(chunks_cache_path, "rb") as f:
            cached_data = pickle.load(f)
        if cached_data.get("fingerprint") == current_fingerprint:
            chunks = cached_data["chunks"]
            cache_valid = True
            print(f"  -> Loaded {len(chunks)} chunks from cache (fingerprint matches).")
        else:
            print("  -> Cache fingerprint mismatch (PDF or extraction logic changed) "
                  "— rebuilding automatically.")

    if not cache_valid:
        print("Extracting PDF via Docling (DocLayNet + TableFormer, VLM for figures)...")
        chunks = ingest_and_chunk(path)

        verified_chunks = build_verified_chunks()
        chunks = chunks + verified_chunks
        print(f"  -> {len(chunks) - len(verified_chunks)} chunks extracted "
              f"({sum(1 for c in chunks if c.get('chunk_type')=='table')} tables, "
              f"{sum(1 for c in chunks if c.get('chunk_type')=='figure')} figures) "
              f"+ {len(verified_chunks)} verified entries = {len(chunks)} total")

        with open(chunks_cache_path, "wb") as f:
            pickle.dump({"fingerprint": current_fingerprint, "chunks": chunks}, f)
        print(f"  -> Cached to {chunks_cache_path}.\n")
        if os.path.exists("./chroma_store"):
            shutil.rmtree("./chroma_store")
            print("  -> Cleared ./chroma_store (fingerprint changed, old embeddings invalid).\n")

    while True:
        query = input("\n[User] (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break

        print("  Searching...")
        dense_ranking = generate_embeddings(chunks, query)
        sparse_ranking = bm25_search(chunks, query)
        fused_ids = reciprocal_rank_fusion([dense_ranking, sparse_ranking])

        chunk_type_filter = infer_metadata_filter(query)
        filtered_ids = metadata_filtering(fused_ids, chunks, chunk_type=chunk_type_filter)
        filtered_ids = force_include_verified(query, chunks, filtered_ids)
        print("\n========== BEFORE RERANKING ==========")

        lookup = {c["chunk_id"]: c for c in chunks}

        for i, chunk_id in enumerate(filtered_ids[:25], start=1):
            c = lookup.get(chunk_id)
            if c is None:
                continue

            print(f"\n--- Candidate {i} ---")
            print(f"Chunk ID:          {c['chunk_id']}")
            print(f"Page:              {c['page_num']}")
            print(f"Type:              {c['chunk_type']}")
            print(f"Extraction method: {c['extraction_method']}")
            print("TEXT:")
            print(c["text"])


        print("  Reranking...")
        top_chunks = reranking(filtered_ids[:25], chunks, query, top_k=10)
        
        print("\n==========After RERANKING ==========")

        for i, c in enumerate(top_chunks, start=1):
            print(f"\n--- Reranked #{i} ---")
            print(f"Chunk ID:          {c['chunk_id']}")
            print(f"Page:              {c['page_num']}")
            print(f"Type:              {c['chunk_type']}")
            print(f"Extraction method: {c['extraction_method']}")
            print("TEXT:")
            print(c["text"])

        print("  Generating answer...")
        response = agent(query, top_chunks)

        print("\n[Assistant]:", response)
        print("\nSources used:")
        for c in top_chunks:
            method = c.get("extraction_method", "unknown")
            ctype = c.get("chunk_type", "unknown")
            print(f"  Page {c['page_num']} ({c['source_file']}) [{ctype}/{method}]")