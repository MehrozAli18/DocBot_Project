
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import config
from pdf_loader import load_multiple_pdfs
from text_splitter import split_pages_into_chunks
from embeddings import get_embedding_model
from vector_store import DocBotVectorStore
from rag_chain import answer_question


# ---------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------
st.set_page_config(page_title="DocBot", page_icon="📄")
st.title("📄 DocBot")
st.caption("An AI assistant that answers questions ONLY from the documents you upload.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question":..., "answer":..., "sources":...}

# ---------------------------------------------------------------------
# SIDEBAR: file upload + settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload your documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    use_real_llm = st.checkbox(
        "Use real LLM (requires OPENAI_API_KEY)",
        value=False,
        help="If unchecked, DocBot runs in offline demo mode: it shows the "
             "most relevant chunk directly instead of generating a new "
             "sentence with an LLM. Useful for testing without an API key.",
    )

    embedding_mode = st.selectbox(
        "Embedding model",
        options=["auto", "huggingface", "tfidf"],
        index=0,
        help="'auto' tries the real Hugging Face model first, and falls "
             "back to TF-IDF only if that model can't be downloaded "
             "(e.g. no internet). 'tfidf' is a lightweight offline mode "
             "useful for quick testing.",
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading PDFs, splitting into chunks, and building the search index..."):
            # Streamlit gives us in-memory uploaded files; we save them
            # temporarily so pypdf can open them like normal files.
            temp_paths = []
            os.makedirs("temp_uploads", exist_ok=True)
            for f in uploaded_files:
                path = os.path.join("temp_uploads", f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                temp_paths.append(path)

            pages = load_multiple_pdfs(temp_paths)
            chunks = split_pages_into_chunks(pages)

            embedding_model = get_embedding_model(embedding_mode)
            vector_store = DocBotVectorStore()
            vector_store.build(chunks, embedding_model)

            st.session_state.vector_store = vector_store
            st.session_state.embedding_model = embedding_model
            st.session_state.chat_history = []  # reset chat for the new document set

        st.success(f"Processed {len(uploaded_files)} document(s) into {len(chunks)} chunks. Ready!")

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []

# ---------------------------------------------------------------------
# MAIN AREA: question input + chat history
# ---------------------------------------------------------------------
st.header("2. Ask a question")

if st.session_state.vector_store is None:
    st.info("👈 Upload and process at least one PDF in the sidebar to get started.")
else:
    question = st.text_input("Your question about the uploaded document(s):")

    if st.button("Ask") and question.strip():
        with st.spinner("DocBot is retrieving relevant chunks and generating an answer..."):
            result = answer_question(
                question,
                st.session_state.vector_store,
                st.session_state.embedding_model,
                use_real_llm=use_real_llm,
            )
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        })

    # Show chat history, most recent first.
    st.subheader("Chat history")
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander("📚 Sources DocBot used for this answer"):
                    for s in turn["sources"]:
                        st.write(f"- **{s['source']}**, page {s['page']} (distance: {s['distance']})")
            else:
                st.caption("No confident source found for this question.")
