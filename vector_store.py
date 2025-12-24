from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "sources"
VECTOR_DIM = 1536

client = QdrantClient(path=QDRANT_PATH)


def ensure_collection():
    """Create collection if it doesn't exist"""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )


def upsert_text_chunks(source_id: str, chunks: list[str], vectors: list[list[float]]):
    """Store text chunks with their embeddings"""
    points = []

    for text, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source_id": source_id,
                    "text": text,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def search_similar(query_vector, limit=5):
    """
    Search for similar vectors - FIXED for Qdrant v1.16.2
    """
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )
    
    return results.points