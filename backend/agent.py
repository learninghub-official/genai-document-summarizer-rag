# backend/agent.py

from backend.embeddings_faiss import load_faiss_index, search_index
from backend.local_llm_summarizer import summarize_text


class DocumentAgent:
    def __init__(self, index_path="vector_db/faiss_index"):
        self.index = load_faiss_index(index_path)

    def answer_question(self, question, top_k=3):
        """
        Retrieve top-k chunks and ask LLM to answer based on them
        """

        results = search_index(self.index, question, top_k=top_k)

        context = "\n\n".join(results)

        prompt = f"""
You are a compliance document assistant.
Answer ONLY from the provided context.
If not found, say: Not present in document.

Context:
{context}

Question:
{question}
"""

        answer = summarize_text(prompt)
        return answer
