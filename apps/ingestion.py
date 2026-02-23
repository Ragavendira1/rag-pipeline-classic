# Objective Take any PDF and Extract Text out of it and create Chunks of it
import os
import re
from typing import List, Dict
from pypdf import PdfReader
from .config import CHUNK_SIZE, CHUNK_OVERLAP


# Step 1: Extract Text from PDF
def extract_pages_from_pdf(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # Only add pages that have text
            pages.append({"page_number": i + 1, "text": text})
    return pages

def extract_pages_from_txt(txt_path: str) -> List[Dict]:
    with open(txt_path, 'r', encoding='utf-8') as file:
        return [{"page_number": 1, "text": file.read()}]

def extract_pages(file_path: str) -> List[Dict]:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == '.pdf':
        return extract_pages_from_pdf(file_path)
    elif extension in ('.txt', '.md'):
        return extract_pages_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


# Step 2: Create Chunks of Text
def clean_text(text: str) -> str:
    # Remove multiple spaces and newlines
    return re.sub(r'\s+', ' ', text).strip()

def create_chunks(pages: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    chunks = []
    full_text = ""
    page_map = []  # tracks which page each character belongs to

    for page in pages:
        text = clean_text(page["text"])
        if text:
            if full_text:
                full_text += " "  # Add space between pages
                page_map.append(page["page_number"])
            full_text += text
            page_map.extend([page["page_number"]] * len(text))

    # Create chunks with overlap
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk_text = full_text[i:i + chunk_size]
        chunk_page_numbers = set(page_map[i:i + chunk_size])
        chunks.append({
            "chunk_text": chunk_text,
            "page_numbers": list(chunk_page_numbers)
        })

    return chunks


def ingest_document(file_path: str) -> List[Dict]:
    file_name = os.path.basename(file_path)
    pages = extract_pages(file_path)
    chunks = create_chunks(pages)
    records = []
    for ix, chunk in enumerate(chunks):
        page_string = ",".join(str(num) for num in chunk["page_numbers"])
        records.append({
            "id": f"{file_name}::chunk_{ix+1}",
            "chunk_text": chunk["chunk_text"],
            "source": file_name,
            "pages": page_string,
        })
    print(f"Processed '{file_name}'-- {len(chunks)} chunks created.")
    return records
