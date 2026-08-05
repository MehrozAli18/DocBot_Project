"""
vector_store.py
----------------
Storing all the chunk vectors somewhere searchable.
Retrieving the most relevant ones for a given question.

"""

import faiss
import numpy as np


class DocBotVectorStore:
    def __init__(self):
        self.index = None         
        self.chunks = []          
        self.dimension = None     

    def build(self, chunks, embedding_model):
        """
        Takes the chunk dicts (from text_splitter.py) and an embedding
        model (from embeddings.py), then builds the searchable index.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        # Turn every chunk's text into a vector (a list of numbers).
        vectors = embedding_model.embed_documents(texts)
        vectors_np = np.array(vectors, dtype="float32")
        self.dimension = vectors_np.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors_np)

    def search(self, query_vector, top_k=None):

        top_k = top_k or __import__("config").TOP_K_CHUNKS
        query_np = np.array([query_vector], dtype="float32")

        distances, indexes = self.index.search(query_np, top_k)

        results = []
        for rank, idx in enumerate(indexes[0]):
            if idx == -1: 
                continue
            chunk = self.chunks[idx].copy()
            chunk["distance"] = float(distances[0][rank])
            results.append(chunk)
        return results


if __name__ == "__main__":
    import os
    import config
    from pdf_loader import load_multiple_pdfs
    from text_splitter import split_pages_into_chunks
    from embeddings import get_embedding_model

    sample_files = [
        os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)
        if f.endswith(".pdf")
    ]
    pages = load_multiple_pdfs(sample_files)
    chunks = split_pages_into_chunks(pages)

    model = get_embedding_model("tfidf") 
    store = DocBotVectorStore()
    store.build(chunks, model)

    question = "How many spaces should I use for indentation?"
    query_vector = model.embed_query(question)
    top_matches = store.search(query_vector, top_k=3)

    print(f"Question: {question}\n")
    for m in top_matches:
        print(f"[{m['source']} - page {m['page']}] (distance={m['distance']:.3f})")
        print(m["text"][:200])
        print("---")
