from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


class Chunker:
    def __init__(self, tokenizer_name='Qwen/Qwen3-Embedding-0.6B', max_tokens=512):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens))
    
    
    def chunk_doc(self, parsed):
        doc = parsed.document
        doc_id = parsed.input.document_hash
        doc_name = parsed.input.file.name 
        chunks = []

        for i, chunk in enumerate(self.chunker.chunk(doc)):
            chunks.append({
                'doc_id': doc_id,
                'doc_name': doc_name,
                'chunk_id': i,
                'text': self.chunker.contextualize(chunk=chunk),
                'headings': chunk.meta.headings or [],
                'pages': sorted({
                    p.page_no
                    for item in chunk.meta.doc_items
                    for p in item.prov
                })
            })

        return chunks