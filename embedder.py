from sentence_transformers import SentenceTransformer
import torch
from langfuse import observe


class Embedder:
    def __init__(self, embedder_name='Qwen/Qwen3-Embedding-0.6B'):
        self.model = SentenceTransformer(embedder_name, model_kwargs={"dtype": torch.float16})

    @observe(name="embed_chunks", as_type="embedding", capture_input=False, capture_output=False)
    def embed_chunks(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    @observe(name="embed_query", as_type="embedding", capture_input=False, capture_output=False)
    def embed_query(self, query: str):
        return self.model.encode(query, prompt_name="query", normalize_embeddings=True)