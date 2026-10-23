from parser import Parser
from chunker import Chunker
from embedder import Embedder
from vector_store import VectorStore
from semantic_cache import SemanticCache
from generator import Generator
from reranker import Reranker
from history import History
from router import router
from pydantic import ValidationError
from helper import build_prompt, make_answer_model
from prompts import RAG_SYSTEM_PROMPT, JSON_OUTPUT_PROMPT
from errors import HistoryError, RankError
from enum import IntEnum
import logging
import json
import os

logger = logging.getLogger(__name__)

COLLECTION = "docs"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")

class Route(IntEnum):
    NO_RETRIEVAL = 0
    STANDALONE = 1
    FOLLOW_UP = 2
    MULTIPLE = 3 


class RAG:
    def __init__(self):
        self.parser = Parser()
        self.chunker = Chunker()
        self.embedder = Embedder()
        dim = self.embedder.model.get_sentence_embedding_dimension()
        self.store = VectorStore(COLLECTION, QDRANT_URL, dim)
        self.cache = SemanticCache(self.store.client, dim)  
        self.reranker = Reranker()
        self.gen = Generator(base_url=LLM_BASE_URL)
        self.history = History(REDIS_URL)


    def ingest(self, src):
        parsed = self.parser.parse_doc(src)
        chunks = self.chunker.chunk_doc(parsed)
        vectors = self.embedder.embed_chunks([c["text"] for c in chunks])
        self.store.add(chunks, vectors)
        self.cache.clear()
        logger.info("Ingested %s (%d chunks)", src, len(chunks))
        return len(chunks)

    def remove_doc(self, doc_name):
        n = self.store.delete(doc_name)
        if n:
            self.cache.clear()
            logger.info("Deleted document '%s' (%d chunks)", doc_name, n)
        return n

    def ask(self, session_id, query):
        cache_key = None
        retrieved = []
        used_chunk_ids = []
        
        try:
            history = self.history.get(session_id)

        except HistoryError:
            logger.warning("Could not retrieve history", exc_info=True)
            history = []

        route, fell_back = router(self.gen, query, history=history)

        if route.type == Route.NO_RETRIEVAL: 
            prompt = build_prompt(RAG_SYSTEM_PROMPT, query, history=history)
            answer= self.gen(prompt).strip()

        else:
            if route.type in (Route.STANDALONE, Route.FOLLOW_UP) and not fell_back:
                cache_key = route.queries[0]
                vec = self.embedder.embed_query(cache_key)
                hit = self.cache.lookup(vec)
                if hit is not None:
                    self._save_history(session_id, query, hit)
                    return hit

            retrieved = self._retrieve(route, original_query=query, top_k=(10 if route.type == Route.MULTIPLE else 5))
            if not retrieved:
                answer = "I couldn't find relevant information in the documents."
                self._save_history(session_id, query, answer)
                return answer

            chunks = [r["text"] for r in retrieved]
            prompt = build_prompt(RAG_SYSTEM_PROMPT + JSON_OUTPUT_PROMPT, query, chunks, history=history)

            Model = make_answer_model(len(chunks))
            raw = self.gen(prompt, schema=Model.model_json_schema())
            try:
                out = Model.model_validate_json(raw)
                answer, used_chunk_ids = out.answer, sorted(set(out.used_chunks))
                logger.info("query %r used chunks %s", query, used_chunk_ids)
            except ValidationError:
                logger.warning("Invalid answer output, falling back to raw text", exc_info=True)
                cache_key = None
                answer = json.loads(raw)["answer"]
        
        if cache_key and retrieved and used_chunk_ids:
            self.cache.store(cache_key, vec, answer)

        if answer:
            self._save_history(session_id, query, answer)

        return answer

    
    def _retrieve(self, route, original_query, k=20, fetch_k=50, top_k=5):
        seen = set()
        results = []

        for q in route.queries:
            hits = self.store.search(self.embedder.embed_query(q), q, k=k, fetch_k=fetch_k)
            for h in hits:
                key = (h["metadata"]["doc_id"], h["metadata"]["chunk_id"])
                if key not in seen:
                    seen.add(key)
                    results.append(h)

        rerank_query = original_query if route.type == Route.MULTIPLE else route.queries[0]
        try:
            return self.reranker.rerank(rerank_query, results, top_k=top_k)

        except RankError:
            logger.warning("Reranking failed; returning retrieved results", exc_info=True)
            return results[:top_k]


    def _save_history(self, session_id, query, answer):
        try:
            self.history.add(session_id, query, answer)

        except HistoryError:
            logger.warning("Could not save history", exc_info=True)