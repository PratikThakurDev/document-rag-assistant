import re
import pymupdf


def split_into_sections(text):
    lines = text.splitlines()

    sections = []
    current_section = []

    # Matches headings like:
    # 2.4
    # 2.4.3
    # 10.2.1
    heading_pattern = re.compile(r"^\d+(?:\.\d+)+\.?$")

    for line in lines:
        line = line.strip()

        if heading_pattern.match(line):
            # Save everything before the new section
            if current_section:
                sections.append(
                    clean_text("\n".join(current_section))
                )

            current_section = [line]

        else:
            current_section.append(line)

    # Save final section
    if current_section:
        sections.append(
            clean_text("\n".join(current_section))
        )

    return sections

def clean_page_text(text):
    text = text.replace("\x07", "=")

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)

def clean_text(text):
    # Common PDF extraction artifact seen in this document
    text = text.replace("\x07", "=")

    # Normalize whitespace
    text = " ".join(text.split())

    return text

def extract_pages(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = clean_page_text(page.get_text())

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

        # Split page at section boundaries first
        sections = split_into_sections(page["text"])

        for section in sections:

            # Then make word chunks only inside that section
            section_chunks = chunk_text(
                section,
                chunk_size=chunk_size,
                overlap=overlap
            )

            for text in section_chunks:
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": text
                })

                chunk_id += 1

    return all_chunks

