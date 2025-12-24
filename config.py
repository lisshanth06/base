import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "sources"
VECTOR_DIM = 1536