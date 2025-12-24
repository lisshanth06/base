from openai import OpenAI
from .config import EMBEDDING_MODEL

client = OpenAI()

def embed_text(text: str) -> list[float]:
    """Generate embeddings for text"""
    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return res.data[0].embedding


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Split text into chunks"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks