"""
embeddings.py
-------------
Turning text chunks into numbers vectors, so a computer can measure 
how "similar" two pieces of text are.

"""

import config


class HuggingFaceEmbeddingModel:

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        #Turn a LIST of chunk texts into a list of vectors.
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        #Turn a SINGLE question into one vector.
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class LocalTfidfEmbeddingModel:

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self._fitted = False

    def embed_documents(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return vectors.toarray().tolist()

    def embed_query(self, text):
        if not self._fitted:
            raise RuntimeError(
                "embed_documents() must be called once (to build the "
                "vocabulary) before embed_query()."
            )
        vector = self.vectorizer.transform([text])
        return vector.toarray()[0].tolist()


def get_embedding_model(mode: str = "auto"):
    if mode == "tfidf":
        return LocalTfidfEmbeddingModel()
    if mode == "huggingface":
        return HuggingFaceEmbeddingModel()

    try:
        return HuggingFaceEmbeddingModel()
    except Exception as e:
        print(f"[DocBot] Could not load Hugging Face model ({e}).")
        print("[DocBot] Falling back to local TF-IDF embeddings for this run.")
        return LocalTfidfEmbeddingModel()


if __name__ == "__main__":
    model = get_embedding_model("tfidf")  # use "auto" once you have internet
    sample_texts = ["Use 4 spaces per indentation level.", "Docstrings should end with a period."]
    vectors = model.embed_documents(sample_texts)
    print(f"Created {len(vectors)} vectors, each of length {len(vectors[0])}.")
