"""
config.py
---------

The Purpose of this file is to keep all the settings in one place,
so that it can be easily modified later without having to dig through the codebase.
"""

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "sample_pdfs")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")

# --- Chunking settings --------------------------------------------------
# LLMs and embedding models can't process a whole PDF at once.
# So we split documents into small overlapping chunks of text.
CHUNK_SIZE = 500        # roughly how many characters per chunk
CHUNK_OVERLAP = 50      # overlap so we don't cut a sentence/idea in half

# --- Embedding model ------------------------------------------------------
# We turn text into numbers (vectors) so the computer can measure how similar" two pieces of text are.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM settings -----------------------------------------------------
# This is the model that actually WRITES the final answer, using the chunks we retrieved.
LLM_PROVIDER = "openai"        # can be swapped to another provider later
LLM_MODEL_NAME = "gpt-4o-mini"  # a small, cheap, fast OpenAI model
LLM_TEMPERATURE = 0.0            # 0 = as factual/deterministic as possible

# --- Retrieval settings -------------------------------------------------
# We only fetch the TOP_K most relevant chunks for a question.
TOP_K_CHUNKS = 4

# --- Anti-hallucination fallback message --------------------------------
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."
