import uuid
from datetime import datetime, timedelta, timezone
from qdrant_client import models
import logging

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, client, dim, name="semantic_cache", threshold=0.95, ttl=timedelta(days=7)):
        self.name = name
        self.client = client
        self.dim = dim
        self.threshold = threshold
        self.ttl = ttl
        self._create()


    def _create(self):
        if not self.client.collection_exists(self.name):
            self.client.create_collection(
                self.name,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            )
  

    def lookup(self, vec):
        pts = self.client.query_points(
            self.name,
            query=vec.tolist(),
            query_filter=models.Filter(must=[models.FieldCondition(
                key="expires_at",
                range=models.DatetimeRange(gt=datetime.now(timezone.utc)),
            )]),
            limit=1,
            score_threshold=self.threshold,
            with_payload=True,
        ).points

        if pts:
            logger.info("Cache hit score=%.3f cached_question=%r",pts[0].score, pts[0].payload.get("question"))
            return pts[0].payload["answer"]
        
        return None


    def store(self, question, vec, answer):
        try:
            self.client.upsert(self.name, points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, question.strip().lower())),
                vector=vec.tolist(),
                payload={
                    "question": question,
                    "answer": answer,
                    "expires_at": (datetime.now(timezone.utc) + self.ttl).isoformat(),
                },
            )])
        except Exception:
            logger.warning("Cache write failed", exc_info=True)


    def clear(self):
        try:
            if self.client.count(self.name, exact=True).count == 0:
                return
            self.client.delete_collection(self.name)
            self._create()
        except Exception:
            logger.warning("Cache clear failed", exc_info=True)