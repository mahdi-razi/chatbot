from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

name = 'Qwen/Qwen3-Embedding-0.6B'
max_tokens = 512
tokenizer = AutoTokenizer.from_pretrained(name)

chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens))

def chunk_doc(doc):
    chunks = list(chunker.chunk(doc))
    texts = [chunker.contextualize(chunk=c) for c in chunks]
    return chunks, texts