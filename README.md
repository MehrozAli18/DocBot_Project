# 📄 DocBot – AI-Powered Document Q&A Chatbot (RAG)

DocBot is a simple chatbot that answers questions **only using the content of
PDF documents you upload** — it does not make up information that isn't in
your documents. It was built as the Final Project for the AI Bootcamp
(NLP/LLM track), using the **Retrieval-Augmented Generation (RAG)** approach.

## What DocBot Does

1. You upload one or more PDF documents.
2. DocBot reads them, splits them into small chunks, and turns each chunk
   into a vector (embedding).
3. When you ask a question, DocBot finds the most relevant chunks and asks
   an LLM to answer **using only those chunks**.
4. If the answer isn't in your documents, DocBot says so honestly instead of
   guessing.

## Project Structure

```
docbot/
├── app.py                  # Streamlit web app (run this file)
├── requirements.txt        # Python dependencies
├── test_questions.py       # Evaluation script (15-20 test questions)
├── data/
│   └── sample_pdfs/        # Real sample documents (PEP 8 & PEP 257, official Python docs)
└── src/
    ├── config.py           # All settings in one place (chunk size, model names, etc.)
    ├── pdf_loader.py        # Step 1: extract text from PDFs
    ├── text_splitter.py     # Step 2: split text into chunks
    ├── embeddings.py        # Step 3: turn text into vectors
    ├── vector_store.py      # Step 4/5: store and search vectors (FAISS)
    └── rag_chain.py          # Step 6: retrieve chunks + generate a grounded answer
```

Each file above has detailed comments inside explaining **what** it does and
**why** it's needed — read them in order (pdf_loader → text_splitter →
embeddings → vector_store → rag_chain) to understand the full pipeline.

## How to Run It

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional but recommended) Set your OpenAI API key, so DocBot can generate
   real answers instead of running in offline demo mode:
   ```
   export OPENAI_API_KEY="your-key-here"
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. In the sidebar: upload a PDF (or use the sample ones in
   `data/sample_pdfs/`), click **Process documents**, then ask a question.

## Two Embedding Modes

- **Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)** — the real,
  recommended mode. Free, open-source, downloads automatically the first
  time you run it (needs internet once).
- **TF-IDF (offline fallback)** — a simpler, classic technique
  (scikit-learn) that needs no downloads or internet. Useful for quick
  testing, but less accurate at understanding meaning (see Limitations in
  the project report).

## How Hallucinations Are Prevented

1. The LLM is instructed to answer **only** from the retrieved chunks.
2. If the best-matching chunk still isn't close enough, DocBot replies
   *"I could not find this information in the uploaded documents."*
   instead of calling the LLM at all.
3. Temperature is set to `0` (as factual/deterministic as possible).
4. Every answer shows its **source document and page number**, so you can
   verify it yourself.

## Evaluation

Run `python test_questions.py` to test DocBot against 20 realistic
questions (15 that should be answerable from the sample documents, 5 that
deliberately are not, to test the "I don't know" fallback). See the project
report for the full results table and discussion.

## Known Limitations

- The offline TF-IDF mode does not always correctly reject unrelated
  questions (it matches on shared words, not true meaning) — the
  recommended Hugging Face embedding model handles this much better.
- The current version only supports PDF files (not Word docs or plain
  text).
- Answers are only as good as the documents provided — DocBot cannot answer
  questions about information that was never in the uploaded PDFs.

## Credits

Sample documents used: **PEP 8 – Style Guide for Python Code** and
**PEP 257 – Docstring Conventions**, both official, publicly available
Python documentation from the [python/peps](https://github.com/python/peps)
repository.
