from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Regulatory obligations must be followed",
    "Failure leads to penalties",
    "Audits are mandatory"
]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("FAISS index size:", index.ntotal)
