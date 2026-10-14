import chromadb


class VectorStore:
    def __init__(self, name, path):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=name,
            configuration={"hnsw": {"space": "cosine"}},
        )


    def add(self, chunks, vectors):
        if not chunks:
            return

        self.collection.delete(where={'doc_id': chunks[0]['doc_id']})

        ids = [f'{c['doc_id']}_{c['chunk_id']}' for c in chunks]
        texts = [c['text'] for c in chunks]

        metadatas = []
        for c in chunks:
            metadatas.append({
                'doc_id': c['doc_id'],
                'chunk_id': c['chunk_id'],
                'headings': " > ".join(c['headings']),
                'pages': ','.join(map(str, c['pages'])),
            })

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
        )


    def search(self, query_vector, k=10):
        res = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=k)

        return [
            {'text': t, 'metadata': m, 'distance': d}
            for t, m, d in zip(res['documents'][0], res['metadatas'][0], res['distances'][0])
        ]


    def delete(self, doc_id):
        self.collection.delete(where={"doc_id": doc_id})