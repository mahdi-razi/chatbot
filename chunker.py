from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

name = 'Qwen/Qwen3-Embedding-0.6B'
max_tokens = 512
tokenizer = AutoTokenizer.from_pretrained(name)

chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens))

def chunk_doc(doc, src):
    doc_id = str(src)
    chunks = []

    for i, chunk in enumerate(chunker.chunk(doc)):
        chunks.append({
            'doc_id': doc_id,
            'chunk_id': i,
            'text': chunker.contextualize(chunk=chunk),
            'headings': chunk.meta.headings or [],
            'pages': sorted({
                p.page_no
                for item in chunk.meta.doc_items
                for p in item.prov
            }),
        })

    return chunks