from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

name = 'Qwen/Qwen3-Embedding-0.6B'
tokenizer = AutoTokenizer.from_pretrained(name)

chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=512))

def chunk_doc(doc):
    return list(chunker.chunk(doc))