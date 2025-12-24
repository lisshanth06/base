# test_rag.py
from ai_engine.embeddings import chunk_text, embed_text
from ai_engine.vector_store import ensure_collection, upsert_text_chunks
from ai_engine.rag import answer_question

ensure_collection()

text = "Python was created by Guido van Rossum and released in 1991."
chunks = chunk_text(text)
vectors = [embed_text(c) for c in chunks]

upsert_text_chunks("python_intro", chunks, vectors)

print(answer_question("When was Python released?"))