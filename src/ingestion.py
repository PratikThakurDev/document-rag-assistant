import pymupdf


def extract_pages(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    doc.close()

    return pages

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = words[start:end]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if end >= len(words):
            break

    return chunks

def process_pdf(pdf_path, chunk_size=150, overlap=30):
    pages = extract_pages(pdf_path)

    all_chunks = []
    chunk_id = 0

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for text in page_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": text
            })

            chunk_id += 1

    return all_chunks