from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

name = 'Qwen/Qwen3-Embedding-0.6B'
max_tokens = 512
tokenizer = AutoTokenizer.from_pretrained(name)

chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens))

def chunk_doc(parsed):
    doc = parsed.document
    doc_id = parsed.input.document_hash
    doc_name = parsed.input.file.name 
    chunks = []

    for i, chunk in enumerate(chunker.chunk(doc)):
        chunks.append({
            'doc_id': doc_id,
            'doc_name': doc_name,
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

# improve doc_id(check)
# result.input.file.name     # "report.pdf"(check)
# add allowed groups

# add updated at
# result.input.file          # PurePath object
# result.input.file.stem     # "report"
# result.input.file.suffix   # ".pdf"
# str(result.input.file)     # String representation of the path