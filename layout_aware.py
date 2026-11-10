#generally most of the older pdfs do not contain the real text which makes our pymupdf(can only extract the real text) to extract the text.
# so use py wrapper on google's tesseract ocr engine(ocr-optical charachter recognisation fallback) to capture the 
#install Pillow and import PIL means:Python Imaging Library, Now maintained as Pillow.This library lets Python work with images.



#get_text("blocks") → works for real text (Session 2 gave you clean output)
#When that returns nothing → OCR fallback kicks in, and now you've seen why it's a fallback, not a first choice — quality drops noticeably
#In your real chunk_pages() pipeline, you'd want to flag which chunks came from OCR (e.g., "extraction_method": "ocr" in metadata) so you know to trust those less, or manually review them for a project as precision-sensitive as structural engineering data



import fitz
from PIL import Image
import io
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# print(pytesseract.get_tesseract_version())

def extract_page_text(page, dpi=300):
    """Try native text extraction first; fall back to OCR if the page has no text layer."""
    blocks = page.get_text("blocks")
    real_text_blocks = [b for b in blocks if b[4].strip()]             # {1,1,1,1,"steel design is crucial"} here b[4]="steel design is crucial."

    if real_text_blocks:
        return real_text_blocks, "native"

    # No text found — render page as image and OCR it
    pix = page.get_pixmap(dpi=dpi)                              
    img = Image.open(io.BytesIO(pix.tobytes("png")))            # from pixmap to image
    ocr_text = pytesseract.image_to_string(img)                    # from image to string
    return ocr_text, "ocr"


doc = fitz.open(r"C:\Users\DELL\Downloads\table.jpg")
page = doc[0]
result, method = extract_page_text(page)
print(f"Extraction method used: {method}")
print(result)