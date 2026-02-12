def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

if __name__ == "__main__":
    sample_text = "This is a sample compliance document. " * 50
    chunks = chunk_text(sample_text)
    print(f"Total chunks: {len(chunks)}")
