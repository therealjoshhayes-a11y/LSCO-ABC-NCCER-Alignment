# ocr_test.py
# Quick OCR test on a single NCCER OBJ PDF page
# Run this to inspect raw Tesseract output before writing extraction rules

import pytesseract
from pdf2image import convert_from_path
from code.config import TESSERACT_CMD, NCCER_FILES, POPPLER_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def test_ocr(filepath, pages=3):
    """Convert first n pages of a PDF to images and run OCR."""
    print(f"\nTesting OCR on: {filepath.name}")
    images = convert_from_path(filepath, dpi=300, first_page=1, last_page=pages, poppler_path=POPPLER_PATH)
    for i, image in enumerate(images):
        print(f"\n--- PAGE {i+1} ---")
        text = pytesseract.image_to_string(image)
        print(text[:3000])

if __name__ == "__main__":
    test_ocr(NCCER_FILES["t0_core"][0])