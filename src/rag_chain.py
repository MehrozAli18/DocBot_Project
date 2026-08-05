"""
rag_chain.py
------------
Taking the retrieved chunks
and asking an LLM to write a clear answer, but only using those chunks, 
never its own outside knowledge.

Four techniques are used together for hallucination prevention.

Strict Prompt Instructions:
Restrict the LLM to the provided context only. If the answer isn't found, it responds with "I don't know."

Low Temperature (0.0):
Reduces randomness, making responses more consistent, accurate, and less likely to include made-up information.

Similarity Threshold:
If no relevant document is found, return a fallback response instead of querying the LLM, preventing hallucinations.

Source References:
Include the document name and page number with each answer so users can easily verify the information.
"""

import config


def build_prompt(question, retrieved_chunks):
    context_text = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""You are DocBot, an assistant that answers questions using ONLY the
context provided below, taken from the user's uploaded documents.

Rules you must follow:
- Only use information found in the context. Do not use outside knowledge.
- If the answer is not in the context, reply exactly:
  "{config.NOT_FOUND_MESSAGE}"
- Keep your answer short and clear.
- Mention which source/page the answer came from.

Context:
{context_text}

Question: {question}

Answer (using ONLY the context above):"""
    return prompt


def is_confident_enough(retrieved_chunks, query_vector=None, distance_threshold=1.8):

    if not retrieved_chunks:
        return False

    if query_vector is not None:
        vector_norm = sum(v * v for v in query_vector) ** 0.5
        if vector_norm < 1e-6:  # essentially all zeros
            return False

    best_distance = min(c["distance"] for c in retrieved_chunks)
    return best_distance <= distance_threshold


def call_llm(prompt):
    """
    NOTE: requires an OPENAI_API_KEY environment variable to be set.
    """
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from your environment

    response = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,  # LOW temperature -> factual, not creative
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_llm_offline_demo(prompt, retrieved_chunks):
    best_chunk = min(retrieved_chunks, key=lambda c: c["distance"])
    return (
        f"[OFFLINE DEMO MODE - extractive answer, not a real LLM generation]\n"
        f"{best_chunk['text'][:400]}"
    )


def answer_question(question, vector_store, embedding_model, use_real_llm=False):

    query_vector = embedding_model.embed_query(question)
    retrieved_chunks = vector_store.search(query_vector, top_k=config.TOP_K_CHUNKS)

    if not is_confident_enough(retrieved_chunks, query_vector=query_vector):
        return {
            "answer": config.NOT_FOUND_MESSAGE,
            "sources": [],
        }

    prompt = build_prompt(question, retrieved_chunks)

    if use_real_llm:
        answer = call_llm(prompt)
    else:
        answer = call_llm_offline_demo(prompt, retrieved_chunks)

    sources = [
        {"source": c["source"], "page": c["page"], "distance": round(c["distance"], 3)}
        for c in retrieved_chunks
    ]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import os
    from pdf_loader import load_multiple_pdfs
    from text_splitter import split_pages_into_chunks
    from embeddings import get_embedding_model
    from vector_store import DocBotVectorStore

    sample_files = [
        os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)
        if f.endswith(".pdf")
    ]
    pages = load_multiple_pdfs(sample_files)
    chunks = split_pages_into_chunks(pages)

    model = get_embedding_model("tfidf")
    store = DocBotVectorStore()
    store.build(chunks, model)

    for q in [
        "How many spaces should I use for indentation?",
        "What is the capital of France?",  # NOT in our documents -> should say "not found"
    ]:
        result = answer_question(q, store, model, use_real_llm=False)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("=" * 60)
