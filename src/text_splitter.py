"""
text_splitter.py
-----------------

Embedding models and LLMs work best with short pieces of text. If we
gave the whole PDF page or the whole document to the embedding model,
it would either fail or produce a blurry vector that doesn't clearly
represent any one idea.

If we cut chunks with NO overlap, we risk slicing a sentence or an idea
right down the middle, which makes the chunk less useful. So we cut overlapping chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def split_pages_into_chunks(pages):
    """
    RecursiveCharacterTextSplitter is a smart splitter from LangChain that tries to cut text at
    natural boundaries first (paragraph breaks, then sentences, then
    words) instead of chopping mid-word. This keeps chunks readable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # try these, in order
    )

    all_chunks = []
    chunk_counter = 0

    for page in pages:
        pieces = splitter.split_text(page["text"])
        for piece in pieces:
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": chunk_counter,
                "source": page["source"],
                "page": page["page"],
                "text": piece,
            })

    return all_chunks


if __name__ == "__main__":
    import os
    from pdf_loader import load_multiple_pdfs

    sample_files = [
        os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)
        if f.endswith(".pdf")
    ]
    pages = load_multiple_pdfs(sample_files)
    chunks = split_pages_into_chunks(pages)

    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    print("--- Sample chunk ---")
    print(chunks[0])
