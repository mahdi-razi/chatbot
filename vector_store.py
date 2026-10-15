import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid


class VectorStore:
    def __init__(self, name, path, dim):
        self.client = qdrant_client.QdrantClient(path=path)
        self.name = name
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )


    def add(self, chunks, vectors):
        if not chunks:
            return

        self.delete(chunks[0]['doc_id'])

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{c['doc_id']}_{c['chunk_id']}")),
                vector=v,
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


    def search(self, query_vector, k=10):
        res = self.client.query_points(collection_name=self.name, query=query_vector.tolist(), limit=k, with_payload=True)

        out = []
        for p in res.points:
            payload = p.payload
            text = payload['text']
            metadata = {
                'doc_id': payload['doc_id'],
                'doc_name': payload['doc_name'],
                'chunk_id': payload['chunk_id'],
                'headings': payload['headings'],
                'pages': payload['pages'],
            }
            out.append({'text': text, 'metadata': metadata, 'score': p.score})
        return out


    def delete(self, doc_id):
        self.client.delete(
            collection_name=self.name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )