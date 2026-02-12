# import streamlit as st
# import tempfile
# import os

# from backend.rag_pipeline_local import run_rag_pipeline

# st.set_page_config(page_title="GenAI Compliance Summarizer", layout="wide")

# st.title("📄 GenAI Compliance Document Summarizer + Agent")

# st.write("""
# Upload a compliance PDF document.
# Generate summary and ask questions from it using a Local GenAI Agent.
# """)

# # -----------------------
# # Session State
# # -----------------------

# if "rag_obj" not in st.session_state:
#     st.session_state.rag_obj = None

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # -----------------------
# # Upload
# # -----------------------

# uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded_file:

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         pdf_path = tmp.name

#     st.success("PDF uploaded")

#     if st.button("Build RAG Index"):

#         with st.spinner("Chunking → Embeddings → FAISS → Ready"):
#             rag = run_rag_pipeline(pdf_path)
#             st.session_state.rag_obj = rag

#         st.success("Document indexed!")

# # -----------------------
# # Summary
# # -----------------------

# if st.session_state.rag_obj:

#     if st.button("Generate Summary"):

#         with st.spinner("Generating summary..."):
#             summary = st.session_state.rag_obj.summarize()

#         st.subheader("Summary")
#         st.write(summary)

# # -----------------------
# # Chat
# # -----------------------

# if st.session_state.rag_obj:

#     query = st.text_input("Ask question from document")

#     if st.button("Ask") and query:

#         answer = st.session_state.rag_obj.ask(query)

#         st.session_state.chat_history.append(("You", query))
#         st.session_state.chat_history.append(("Agent", answer))

# # -----------------------
# # Display Chat
# # -----------------------

# for role, msg in st.session_state.chat_history:
#     st.markdown(f"**{role}:** {msg}")




# Fancy loader 


import streamlit as st
import tempfile
import os

from backend.rag_pipeline_local import run_rag_pipeline

# -----------------------------------
# Page Config
# -----------------------------------

st.set_page_config(
    page_title="GenAI Compliance Summarizer",
    layout="wide"
)

# st.title("📄 GenAI Compliance Document Summarizer + Agent")
# st.warning("""
# This AI assistant is designed to answer questions ONLY from the uploaded document.

# It does not think like a human or general chatbot.
# It does not use internet knowledge.
# It only reads your document, finds relevant parts, and gives a short answer from that content.

# Best results come when you ask document-related questions.

# Not suitable for:
# • Creative questions
# • Opinion questions
# • Title or slogan generation
# • Questions not related to the uploaded file
# """)


# st.write("""
# Upload a compliance PDF document.  
# Build the RAG index → Generate Summary → Ask questions from the document using the Local GenAI Agent.
# """)

# -----------------------------------
# Session State Init
# -----------------------------------

st.markdown(
    "<h1 style='text-align: center;'>📄 GenAI Compliance Document Summarizer + Agent</h1>",
    unsafe_allow_html=True
)
st.markdown("""
<div style="text-align:center; font-size:16px;">
This tool reads your uploaded document and answers only from it.<br>
It is not a general chatbot.<br>
Ask only document-related questions for best results.
</div>
""", unsafe_allow_html=True)





if "rag_obj" not in st.session_state:
    st.session_state.rag_obj = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# -----------------------------------
# Upload Section
# -----------------------------------

st.header("Upload Document")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.pdf_path = tmp.name

    st.success("✅ PDF uploaded successfully")

# -----------------------------------
# Build RAG Index
# -----------------------------------

st.header("Build RAG Index")

if st.session_state.pdf_path:

    if st.button("📚 Build RAG Index"):

        with st.spinner("🔄 Extracting → Chunking → Embedding → FAISS Indexing..."):
            rag_engine = run_rag_pipeline(st.session_state.pdf_path)

        st.session_state.rag_obj = rag_engine
        st.success("✅ Document indexed and RAG ready")

# -----------------------------------
# Summary Section
# -----------------------------------

st.header("Generate Summary")

if st.session_state.rag_obj:

    if st.button("🧾 Generate Summary"):

        with st.spinner("🧠 Reading document and generating summary..."):
            summary = st.session_state.rag_obj.summarize()

        st.subheader("📌 Document Summary")
        st.write(summary)

# -----------------------------------
# Chat Section
# -----------------------------------

st.header("Ask Questions From Document")

if st.session_state.rag_obj:

    query = st.text_input("Ask question from document")

    if st.button("💬 Ask Agent"):

        if query.strip() == "":
            st.warning("Please enter a question first.")
        else:
            with st.spinner("🔍 Retrieving context + Generating answer..."):
                answer = st.session_state.rag_obj.ask(query)

            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Agent", answer))

# -----------------------------------
# Chat Display
# -----------------------------------

if st.session_state.chat_history:

    st.subheader("Conversation")

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🥱 Agent:** {msg}")

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")
# st.caption("GenAI Compliance Summarizer • by Mayank Nagar • © 2026")

st.markdown(
    "<p style='text-align: center; font-size:14px;'>by Mayank Nagar | copyright © 2026</p>",
    unsafe_allow_html=True
)

