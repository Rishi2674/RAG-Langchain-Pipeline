# src/pdf_loader.py
from pypdf import PdfReader

def load_pdf(file_path):
    """Load and extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
