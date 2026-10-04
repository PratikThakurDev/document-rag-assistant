import os
import tempfile
import streamlit as st

from src.retriever import Retriever
from src.generator import Generator
from src.ingestion import process_pdf


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_generator():
    return Generator()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document RAG Assistant")

st.caption(
    "Ask questions about your PDF using hybrid retrieval "
    "and a local language model."
)

st.divider()


# --------------------------------------------------
# Load generator
# --------------------------------------------------

generator = load_generator()


# --------------------------------------------------
# Document upload
# --------------------------------------------------

st.subheader("1. Upload document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    # Process only when a new document is uploaded
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

        try:

            with st.spinner(
                "Reading, chunking, and indexing document..."
            ):

                chunks = process_pdf(temp_path)

                retriever = Retriever()
                retriever.build_index(chunks)

                st.session_state.retriever = retriever
                st.session_state.chunk_count = len(chunks)
                st.session_state.file_id = file_id

            st.success("Document ready!")

        finally:
            os.unlink(temp_path)

    # Document information
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Document",
            uploaded_file.name
        )

    with col2:
        st.metric(
            "Chunks",
            st.session_state.chunk_count
        )

    st.divider()


    # --------------------------------------------------
    # Question answering
    # --------------------------------------------------

    st.subheader("2. Ask a question")

    query = st.text_input(
        "Question",
        placeholder="e.g. What are the main features discussed in this document?",
        key="document_question"
    )

    if query:

        retriever = st.session_state.retriever

        with st.spinner("Finding relevant information..."):

            # Semantic + lexical retrieval followed by RRF
            candidates = retriever.hybrid_search(
                query,
                top_k=20
            )

            # CrossEncoder reranking
            results = retriever.rerank(
                query,
                candidates,
                top_k=5
            )

            # Determine query type
            intent = retriever.classify_intent(query)

            # Generate grounded answer
            response = generator.generate(
                query,
                results,
                intent=intent
            )

        st.divider()

        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(
            response["answer"]
        )


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        st.subheader("Sources")

        st.caption(
            "The following passages were used as context "
            "for generating the answer."
        )

        for result in results[:3]:

            with st.expander(
                f"📄 Page {result['page']}"
            ):
                st.write(
                    result["text"]
                )


    # --------------------------------------------------
    # Limitations
    # --------------------------------------------------

    st.divider()

    with st.expander("ℹ️ About this assistant"):

        st.write(
            """
            This assistant uses hybrid retrieval combining semantic
            search and keyword search, followed by reranking and a
            locally running language model.

            Complex equations, tables, scanned documents, and unusual
            PDF formatting may not always be extracted accurately.

            Answers should be verified against the displayed source
            passages when precision is important.
            """
        )


else:

    st.info(
        "Upload a PDF above to start asking questions."
    )