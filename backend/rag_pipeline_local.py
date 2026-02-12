import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# --------------------------------
# PDF Extraction
# --------------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# --------------------------------
# Chunking
# --------------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


# --------------------------------
# RAG Engine Class
# --------------------------------
class RAGEngine:

    def __init__(self, pdf_path):

        # Step 1 — Extract text
        text = extract_text_from_pdf(pdf_path)

        # Step 2 — Chunk
        self.chunks = chunk_text(text)

        # Step 3 — Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(self.chunks)

        # Step 4 — FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

        # Step 5 — Local LLM summarizer
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )


    # --------------------------------
    # Retrieve top chunks
    # --------------------------------
    def retrieve(self, query, top_k=3):

        q_emb = self.embed_model.encode([query])
        distances, idx = self.index.search(np.array(q_emb), top_k)

        return [self.chunks[i] for i in idx[0]]


    # --------------------------------
    # Ask Question (RAG QA Mode)
    # --------------------------------
# --------------------------------
# Ask Question (RAG QA Mode — fixed)
# --------------------------------
    def ask(self, question):

        top_chunks = self.retrieve(question, top_k=3)
        context = " ".join(top_chunks)

        # Instead of prompt — summarize only context
        result = self.summarizer(
            context,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        answer = result[0]["summary_text"]

        # Optional: add short prefix for chat feel
        return f"Based on the document: {answer}"



    # --------------------------------
    # Full Document Summary
    # --------------------------------
    def summarize(self):

        combined = " ".join(self.chunks[:5])  # summarize first few chunks

        result = self.summarizer(
            combined,
            max_length=200,
            min_length=60,
            do_sample=False
        )

        return result[0]["summary_text"]


# # --------------------------------
# # UI Helper Function
# # --------------------------------
# def run_rag_pipeline(pdf_path):
#     return RAGEngine(pdf_path)


# lighter version of summarizer model for cloud deployment
# import fitz
# import numpy as np
# import faiss

# from sentence_transformers import SentenceTransformer

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# # --------------------------------
# # PDF Extraction
# # --------------------------------
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text


# # --------------------------------
# # Chunking
# # --------------------------------
# def chunk_text(text, chunk_size=500, overlap=100):
#     chunks = []
#     start = 0

#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap

#     return chunks


# # --------------------------------
# # RAG Engine
# # --------------------------------
# class RAGEngine:

#     def __init__(self, pdf_path):

#         text = extract_text_from_pdf(pdf_path)
#         self.chunks = chunk_text(text)

#         # Embeddings
#         self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
#         embeddings = self.embed_model.encode(self.chunks)

#         dim = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(np.array(embeddings))

#         # Summarization model (lazy load)
#         self.tokenizer = None
#         self.model = None


#     # --------------------------------
#     # Load summarizer safely
#     # --------------------------------
#     def load_summarizer(self):

#         if self.model is None:
#             model_name = "sshleifer/distilbart-cnn-12-6"

#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


#     # --------------------------------
#     # Generate summary helper
#     # --------------------------------
#     def generate_summary(self, text, max_len=200, min_len=60):

#         self.load_summarizer()

#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=1024
#         )

#         output = self.model.generate(
#             **inputs,
#             max_length=max_len,
#             min_length=min_len,
#             do_sample=False
#         )

#         return self.tokenizer.decode(output[0], skip_special_tokens=True)


#     # --------------------------------
#     # Retrieve
#     # --------------------------------
#     def retrieve(self, query, top_k=3):

#         q_emb = self.embed_model.encode([query])
#         distances, idx = self.index.search(np.array(q_emb), top_k)

#         return [self.chunks[i] for i in idx[0]]


#     # --------------------------------
#     # Ask Question
#     # --------------------------------
#     def ask(self, question):

#         context_chunks = self.retrieve(question, top_k=3)
#         context = " ".join(context_chunks)

#         answer = self.generate_summary(context, max_len=120, min_len=30)

#         return f"Based on the document: {answer}"


#     # --------------------------------
#     # Full Document Summary
#     # --------------------------------
#     def summarize(self):

#         combined = " ".join(self.chunks[:5])

#         return self.generate_summary(combined, max_len=220, min_len=80)


# # --------------------------------
# # UI Helper
# # --------------------------------
# def run_rag_pipeline(pdf_path):
#     return RAGEngine(pdf_path)
