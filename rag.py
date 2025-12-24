from openai import OpenAI
from .embeddings import embed_text
from .vector_store import search_similar

client = OpenAI()


def answer_question(question: str, source_ids: list[str]) -> str:
    """Answer questions using RAG"""
    # Convert question to vector
    query_vector = embed_text(question)

    # Search similar chunks
    hits = search_similar(query_vector)

    # Filter by project sources
    filtered = [
        hit for hit in hits
        if hit.payload.get("source_id") in source_ids
    ]

    if not filtered:
        return "No relevant sources found for this project."

    # Build context from matching chunks
    context = "\n".join(hit.payload["text"] for hit in filtered)

    prompt = f"""
You are an AI assistant.
Answer strictly using the context below.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only from context."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()