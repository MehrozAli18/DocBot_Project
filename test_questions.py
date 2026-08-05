import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import config
from pdf_loader import load_multiple_pdfs
from text_splitter import split_pages_into_chunks
from embeddings import get_embedding_model
from vector_store import DocBotVectorStore
from rag_chain import answer_question


# Each entry: (question, should_be_answerable)
# "should_be_answerable" = True means the answer SHOULD exist in our
# PEP 8 / PEP 257 documents. False means it's an out-of-scope question,
# used to test the anti-hallucination fallback.
TEST_QUESTIONS = [
    ("How many spaces should be used per indentation level?", True),
    ("Should I use tabs or spaces in Python code?", True),
    ("What is the maximum recommended line length in PEP 8?", True),
    ("How should import statements be organized?", True),
    ("What are the naming conventions for functions in Python?", True),
    ("Should class names use CamelCase or snake_case?", True),
    ("How many blank lines should separate top-level functions?", True),
    ("What does PEP 257 say about one-line docstrings?", True),
    ("Should a docstring's closing quotes be on their own line?", True),
    ("What is the purpose of a docstring according to PEP 257?", True),
    ("How should comments be written according to PEP 8?", True),
    ("What naming style is recommended for constants?", True),
    ("Is it okay to use a backslash to continue a line in Python?", True),
    ("What does PEP 8 say about whitespace around operators?", True),
    ("How should multi-line docstrings be formatted?", True),
    # Out-of-scope questions (should trigger "not found" or be clearly uncertain):
    ("What is the capital of France?", False),
    ("What's the best pizza topping?", False),
    ("Who won the last FIFA World Cup?", False),
    ("What is the current price of Bitcoin?", False),
    ("How do I bake a chocolate cake?", False),
]


def run_evaluation():
    sample_files = [
        os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)
        if f.endswith(".pdf")
    ]
    pages = load_multiple_pdfs(sample_files)
    chunks = split_pages_into_chunks(pages)

    embedding_model = get_embedding_model("tfidf")  # offline mode for this test run
    vector_store = DocBotVectorStore()
    vector_store.build(chunks, embedding_model)

    results = []
    for question, should_be_answerable in TEST_QUESTIONS:
        result = answer_question(question, vector_store, embedding_model, use_real_llm=False)
        found_source = len(result["sources"]) > 0
        correct_behaviour = found_source == should_be_answerable

        results.append({
            "question": question,
            "expected_answerable": should_be_answerable,
            "found_source": found_source,
            "correct_behaviour": correct_behaviour,
            "answer_preview": result["answer"][:120].replace("\n", " "),
        })

    return results


def print_report(results):
    correct = sum(1 for r in results if r["correct_behaviour"])
    total = len(results)

    print(f"\n{'='*70}\nDocBot Evaluation Report\n{'='*70}")
    for r in results:
        status = "✅" if r["correct_behaviour"] else "❌"
        expected = "should answer" if r["expected_answerable"] else "should say 'not found'"
        print(f"{status} [{expected}] {r['question']}")
        print(f"    -> {r['answer_preview']}...")
    print(f"\n{'='*70}")
    print(f"Score: {correct}/{total} questions handled correctly "
          f"({correct/total*100:.0f}%)")
    print(f"{'='*70}\n")
    print("NOTE: This score checks whether DocBot correctly found (or correctly")
    print("did NOT find) a source -- it does not automatically grade whether")
    print("the generated wording is perfectly phrased. That judgment is added")
    print("manually to the project report's evaluation table.")


if __name__ == "__main__":
    results = run_evaluation()
    print_report(results)
