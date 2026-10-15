from qdrant_client import QdrantClient, models
import uuid

sparse_model = "Qdrant/bm25"


class VectorStore:
    def __init__(self, name, path, dim):
        self.client = QdrantClient(path=path)
        self.name = name

        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )


    def add(self, chunks, vectors):
        if not chunks:
            return

        self.delete(chunks[0]['doc_id'])

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{c['doc_id']}_{c['chunk_id']}")),
                vector={
                    "dense": v,
                    "sparse": models.Document(text=c['text'], model=sparse_model),
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

        self.client.upsert(collection_name=self.name, points=points)


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
                    query=models.Document(text=query_text, model=sparse_model),
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


    def delete(self, doc_id):
        self.client.delete(
            collection_name=self.name,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        )