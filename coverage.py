import re
import chromadb
import pypdfium2 as pdfium

PDF = "data/IS456.pdf"
c = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("is_codes")
db = c.get(include=["documents", "metadatas"])
db_text = "\n".join(db["documents"])
db_pages = {}
for m in db["metadatas"]:
    db_pages[m.get("page")] = db_pages.get(m.get("page"), 0) + 1

pdf = pdfium.PdfDocument(PDF)
heading = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){1,3})\s+[A-Z]", re.M)
missing, thin, no_text = [], [], []
for i in range(len(pdf)):
    pno = i + 1
    text = pdf[i].get_textpage().get_text_range()
    if len(text.strip()) < 50:
        no_text.append(pno)
        continue
    if len(text) > 1500 and db_pages.get(pno, 0) < 2:
        thin.append((pno, len(text), db_pages.get(pno, 0)))
    for num in set(heading.findall(text)):
        if not re.search(re.escape(num) + r"\b", db_text):
            missing.append((num, pno))

print("pages with no PDF text layer:", no_text)
print("pages with lots of PDF text but almost no chunks (page, pdf_chars, chunks):", thin)
print(f"clause headings in PDF but not found in database: {len(missing)}")
for num, pno in sorted(missing, key=lambda x: x[1])[:80]:
    print(f"  {num}  (page {pno})")
p3 = c.get(where={"page": 3}, include=["documents"])
print("page 3 chunks:", len(p3["ids"]))
for d in p3["documents"][:3]:
    print("  ", d[:150].replace("\n", " "))
