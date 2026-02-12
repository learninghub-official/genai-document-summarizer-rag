# 📄 GenAI Compliance Document Summarizer + RAG Assistant

## 🔷 Project Overview

This project demonstrates a practical, end-to-end **Generative AI + RAG (Retrieval Augmented Generation)** system that can read compliance documents (PDF), understand their contents, and provide structured summaries and document-grounded answers through an interactive UI.

The system is designed to simulate how organizations can safely analyze regulatory and compliance documents using **local open-source models**, ensuring privacy, control, and zero external data leakage.

Unlike general chatbots, this assistant answers strictly from the uploaded document using embeddings + vector search + local LLM summarization.

---

## 🎯 Problem This Project Solves

Compliance and regulatory documents are:

* Long
* Dense
* Hard to review quickly
* Risky if misinterpreted

Manual reading is time-consuming and error-prone.

This system enables:

* Fast document summarization
* Context-based question answering
* Risk and policy extraction
* Controlled, document-grounded responses

---

## 🧠 Core GenAI Architecture Used

This project implements a **Local RAG Pipeline**:

PDF → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Top-K Retrieval → Local LLM Summarization → Answer

### Components:

**PDF Parsing**

* PyMuPDF extracts full document text

**Chunking**

* Text split into overlapping segments
* Preserves semantic continuity

**Embeddings**

* SentenceTransformer (MiniLM)
* Converts text into semantic vectors

**Vector Database**

* FAISS index
* Enables fast similarity search

**Retrieval**

* Top-K most relevant chunks selected per query

**Local LLM**

* facebook/bart-large-cnn
* Used for structured summarization of retrieved context

---

## 🤖 Why Local LLM Instead of Cloud API

This design choice was intentional:

* No external data transfer
* Better for compliance scenarios
* No API key dependency
* Fully offline capable
* Demonstrates open-source GenAI stack

Ideal for regulated environments.

---

## 💬 Assistant Behavior (Important)

This is **NOT a general chatbot**.

It is a:

> Document-Grounded Q&A Assistant

It:

✅ Answers only from uploaded document
✅ Uses semantic retrieval
✅ Summarizes relevant sections
❌ Does not use internet knowledge
❌ Does not generate creative opinions
❌ Not a reasoning chat LLM

---

## 🖥️ Streamlit UI Features

* PDF upload interface
* RAG index builder button
* One-click document summary
* Document Q&A assistant
* Conversation history
* Interview/demo friendly layout
* Dark-mode safe UI text
* Processing spinners for transparency

---

## 📦 Tech Stack

* Python
* Streamlit
* SentenceTransformers
* FAISS
* PyMuPDF
* HuggingFace Transformers
* BART Large CNN
* NumPy

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/learninghub-official/genai-document-summarizer-rag.git
cd GenAI_Compliance_Summarizer

python -m venv venv
source venv/bin/activate  # mac/linux
venv\Scripts\activate     # windows

pip install -r requirements.txt

streamlit run ui_streamlit_app.py
```

---

## 🌐 Live Demo

(Insert your Streamlit Cloud URL here after deploy)

---

## 🧪 Example Questions You Can Ask

* What are the reporting requirements?
* What are compliance risks mentioned?
* What penalties are described?
* What controls are required?
* Summarize risk management section
* What actions are prohibited?

---

## 📚 Concepts Demonstrated

* Generative AI workflow
* RAG architecture
* Embedding search
* Vector databases
* Document chunking strategy
* Local LLM inference
* Context-grounded generation
* Compliance AI use cases

---

## 👤 Author

Mayank Nagar
AI + ML + Cloud Systems
© 2026
