from qdrant_client import QdrantClient, models
import uuid
from errors import StoreError


class VectorStore:
    def __init__(self, name, url, dim, timeout=60, sparse_model="Qdrant/bm25"):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.name = name
        self.sparse_model = sparse_model

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )


    def add(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise StoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
            
        self.delete(doc_name=chunks[0]['doc_name'], doc_id=chunks[0]['doc_id'])

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, c['chunk_id'])),
                vector={
                    "dense": v,
                    "sparse": models.Document(text=c['text'], model=self.sparse_model),
                },
                payload={
                    'text': c['text'],
                    'doc_id': c['doc_id'],
                    'doc_name': c['doc_name'],
                    'chunk_id': c['chunk_id'],
                    'headings': c['headings'],
                    'pages': c['pages'],
                },
            )
            for c, v in zip(chunks, vectors.tolist())
        ]
        
        try:
            self.client.upsert(collection_name=self.name, points=points)

        except Exception as e:
            raise StoreError(f"Failed to store '{chunks[0]['doc_name']}'") from e


    def search(self, query_vector, query_text, k=10, fetch_k=50):
        res = self.client.query_points(
            collection_name=self.name,
            prefetch=[
                models.Prefetch(
                    query=query_vector.tolist(), 
                    using="dense", 
                    limit=fetch_k
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model=self.sparse_model),
                    using="sparse",
                    limit=fetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )

        out = []
        for p in res.points:
            payload = p.payload
            out.append({
                "text": payload["text"],
                "metadata": {
                    "doc_id": payload["doc_id"],
                    "doc_name": payload["doc_name"],
                    "chunk_id": payload["chunk_id"],
                    "headings": payload["headings"],
                    "pages": payload["pages"],
                },
                "score": p.score,
            })

        return out


    def delete(self, doc_name=None, doc_id=None):
        conds = []
        if doc_name is not None:
            conds.append(models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)))
        if doc_id is not None:
            conds.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
        if not conds:
            raise ValueError("doc_name or doc_id is required")

        flt = models.Filter(should=conds)
        try:
            n = self.client.count(self.name, count_filter=flt, exact=True).count
            if n:
                self.client.delete(collection_name=self.name, points_selector=flt)
        except Exception as e:
            raise StoreError(f"Failed to delete document name={doc_name!r} id={doc_id!r}") from e
        return n