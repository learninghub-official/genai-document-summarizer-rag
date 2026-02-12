# 📄 GenAI Compliance Document Summarizer — RAG with Local LLM

## 📌 Project Overview

This project implements a Generative AI powered document summarization system using a Retrieval Augmented Generation (RAG) architecture with a local Large Language Model.

The system processes long compliance or policy documents and produces structured summaries by retrieving the most relevant content chunks and passing them to a local LLM for summarization.

The design prioritizes privacy, explainability, and offline capability — making it suitable for compliance and regulated environments.

---

## 🎯 Objective

To automatically summarize long compliance documents into structured, readable summaries while ensuring:

- Data privacy
- No cloud dependency
- Controlled inference pipeline

---

## 🧠 Why Generative AI?

Generative AI models understand context and generate human-like text. They are well suited for:

- Summarization
- Policy interpretation
- Knowledge extraction

---

## 🏗️ System Architecture

Pipeline implemented:

Document → Text Extraction → Chunking → Embeddings → FAISS Vector Store → Top-K Retrieval → Local LLM → Structured Summary

---

## ⚙️ Detailed Workflow

### 1️⃣ Document Ingestion
PDF compliance document loaded and text extracted.

### 2️⃣ Text Chunking
Large text split into smaller chunks to improve retrieval precision.

### 3️⃣ Embedding Generation
Each chunk converted into a semantic vector representation.

### 4️⃣ FAISS Vector Store
All embeddings stored in FAISS index for fast similarity search.

### 5️⃣ Top-K Retrieval
For a given query, most relevant chunks retrieved using vector similarity.

### 6️⃣ Local LLM Summarization
Retrieved chunks passed to local LLM (BART model) to generate structured summary.

---

## 🤖 Models Used

### facebook/bart-large-cnn
Transformer-based summarization model optimized for long text summarization tasks.

Chosen because:
- High summarization quality
- Stable local execution
- No external API dependency

---

## 🔐 Why Local LLM Instead of Cloud API

- Sensitive document privacy
- No data leakage risk
- No API cost
- Full pipeline control
- Offline capability

---

## 📊 Output

System generates structured summaries highlighting:

- Compliance requirements
- Deadlines
- Penalties
- Regulatory obligations

---

## 💼 Real-World Use Cases

- Compliance review automation
- Legal document summarization
- Policy analysis
- Audit preparation

---

## 🛠️ Tech Stack

- Python
- Transformers
- FAISS
- Sentence Embeddings
- Local LLM
- PDF parsers

---

## 📷 Screenshots to Include

- FAISS index size print
- Chunk retrieval output
- Final structured summary output
- Architecture diagram

---

## 👤 Author

Mayank Nagar — GenAI / RAG Practitioner
