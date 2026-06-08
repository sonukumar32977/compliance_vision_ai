from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({"page": i, "text": text})
    doc.close()
    return pages

def list_pdfs(input_dir: Path):
    pdfs = sorted([p for p in input_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {input_dir}")
    return pdfs