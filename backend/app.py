# app.py

import streamlit as st
import os

from backend.pdf_loader import load_pdf
from backend.chunking import chunk_text
from backend.embeddings_faiss import create_faiss_index, save_faiss_index
from backend.local_llm_summarizer import summarize_text
from backend.agent import DocumentAgent


UPLOAD_DIR = "data/uploads"
INDEX_DIR = "vector_db/faiss_index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.set_page_config(page_title="GenAI Compliance Summarizer", layout="wide")

st.title("📄 GenAI Compliance Document Summarizer + Q&A Agent")

# =========================
# File Upload
# =========================

uploaded_file = st.file_uploader("Upload Compliance PDF", type=["pdf"])

if uploaded_file:

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded")

    # =========================
    # Text Extraction
    # =========================

    text = load_pdf(file_path)
    st.info(f"Extracted characters: {len(text)}")

    # =========================
    # Chunking
    # =========================

    chunks = chunk_text(text)
    st.info(f"Total chunks created: {len(chunks)}")

    # =========================
    # Build FAISS
    # =========================

    if st.button("🔎 Build Vector Index"):

        index = create_faiss_index(chunks)
        save_faiss_index(index, INDEX_DIR)

        st.success("FAISS index built and saved")

    # =========================
    # Summarization
    # =========================

    if st.button("📝 Generate Summary"):

        summary = summarize_text(text[:4000])  # safe size
        st.subheader("Document Summary")
        st.write(summary)

    # =========================
    # Agent Chat
    # =========================

    st.divider()
    st.subheader("💬 Ask Questions From Document")

    if "agent" not in st.session_state:
        st.session_state.agent = DocumentAgent(INDEX_DIR)

    question = st.text_input("Ask a question")

    if question:
        answer = st.session_state.agent.answer_question(question)
        st.write("### Answer")
        st.write(answer)
