import streamlit as st
import tempfile
from src.retriever import Retriever
from src.generator import Generator
from src.ingestion import process_pdf

@st.cache_resource
def load_generator():
    return Generator()

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄"
)

st.title("📄 Document RAG Assistant")

st.write(
    "Upload a PDF and ask questions based on its contents."
)

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

if (
    "retriever" not in st.session_state
    or st.session_state.get("file_id") != file_id
):

    if (
        "retriever" not in st.session_state
        or st.session_state.get("file_id") != file_id
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        with st.spinner("Processing and indexing document..."):
            chunks = process_pdf(temp_path)

            retriever = Retriever()
            retriever.build_index(chunks)

            st.session_state.retriever = retriever
            st.session_state.chunk_count = len(chunks)
            st.session_state.file_id = file_id

        st.success("Document processed successfully!")

    st.write(
        "Total chunks:",
        st.session_state.chunk_count
    )

generator = load_generator()

query = st.text_input(
    "Ask a question about the document"
)

if query:
    retriever = st.session_state.retriever

    with st.spinner("Searching document..."):

        # FAISS + BM25 + RRF
        candidates = retriever.hybrid_search(
            query,
            top_k=20
        )

        # CrossEncoder + intent bonus
        results = retriever.rerank(
            query,
            candidates,
            top_k=5
        )

        # Generate answer
        response = generator.generate(
            query,
            results
        )

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Sources")

    for result in results[:3]:
        with st.expander(f"Page {result['page']}"):
            st.write(result["text"])
