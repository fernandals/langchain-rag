import fitz
from rag.models import DocumentType

def detect_pdf_type(page: fitz.Page) -> DocumentType:
  width = page.mediabox_size[0]
  height = page.mediabox_size[1]

  return (
    DocumentType.SLIDES
    if width / height > 1.1
    else DocumentType.PDF
  )

def extract_slide_structure(text: str) -> tuple[str, str]:
  lines = [l.strip() for l in text.splitlines() if l.strip()]

  if not lines:
    return "", ""
 
  title = lines[0]
  body = "\n".join(lines[1:])

  return title, body