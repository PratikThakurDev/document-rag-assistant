# 📄 Document RAG Assistant

A local Retrieval-Augmented Generation (RAG) application that allows
users to upload PDF documents and ask questions grounded in their content.

## Features

- PDF text extraction using PyMuPDF
- Section-aware document chunking
- Semantic retrieval using MiniLM embeddings + FAISS
- Keyword retrieval using BM25
- Reciprocal Rank Fusion (RRF)
- CrossEncoder reranking
- Query-intent-aware answer generation
- Local SmolLM2-1.7B-Instruct generation
- 4-bit quantization using bitsandbytes
- Source-page inspection
- Streamlit interface

## Architecture

PDF
↓
Text Extraction
↓
Section-Aware Chunking
↓
┌─────────────────┬─────────────────┐
│ Semantic Search │  Lexical Search │
│ MiniLM + FAISS  │      BM25       │
└────────┬────────┴────────┬────────┘
         ↓
Reciprocal Rank Fusion
         ↓
CrossEncoder Reranking
         ↓
Top Context Chunks
         ↓
SmolLM2-1.7B-Instruct
         ↓
Answer + Source Passages

## Why Hybrid Retrieval?

Semantic search is useful when the wording of a question differs from
the document, while BM25 is effective for exact terminology.

The application combines both rankings using Reciprocal Rank Fusion
before reranking the strongest candidates with a CrossEncoder.

## Section-Aware Chunking

Instead of blindly splitting every fixed number of words, the ingestion
pipeline detects numbered section boundaries where possible and performs
overlapping word chunking within each section.

This reduces contamination between adjacent topics while retaining
context through overlap.

## Local Generation

Answers are generated using:

HuggingFaceTB/SmolLM2-1.7B-Instruct

The model runs locally with 4-bit NF4 quantization. This reduces memory
requirements and makes local inference practical on limited GPU hardware.

## Installation

Clone the repository:

git clone https://github.com/PratikThakurDev/document-rag-assistant.git

cd document-rag-assistant

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Usage

1. Upload a PDF.
2. Wait for the document to be extracted, chunked, embedded, and indexed.
3. Ask a question about the document.
4. Read the generated answer.
5. Expand the source passages to inspect the retrieved evidence.

## Testing

The system was tested on both technical and general-purpose documents.

For a simple solar-energy document, it successfully handled:

- direct definition questions
- property/feature extraction
- conceptual explanation questions

A mathematics-heavy quantum-mechanics textbook was also used as a
stress test. Retrieval improved substantially after introducing
section-aware chunking, but generation remained less reliable for dense
mathematical material.

## Limitations

- Mathematical notation may be corrupted during PDF text extraction.
- Scanned/image-only PDFs require OCR, which is not currently included.
- Tables and complex layouts may not be reconstructed accurately.
- SmolLM2-1.7B is lightweight enough for local use but has limited
  reasoning ability compared with larger language models.
- Generated answers should be checked against the displayed source
  passages when high precision is required.

## Future Improvements

- Stronger optional API-based LLM backend
- Improved mathematical/document-layout parsing
- Better support for tables and scanned PDFs
- Multi-document retrieval
- Persistent vector indexes
- More systematic RAG evaluation

## Tech Stack

Python · Streamlit · PyMuPDF · Sentence Transformers · FAISS · BM25 ·
CrossEncoder · Hugging Face Transformers · PyTorch · bitsandbytes

## License

This project is intended for educational and portfolio use.