from sentence_transformers import SentenceTransformer

name = "Qwen/Qwen3-Embedding-0.6B"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(name)

    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True)

    def embed_query(self, query: str):
        return self.model.encode(query, normalize_embeddings=True)