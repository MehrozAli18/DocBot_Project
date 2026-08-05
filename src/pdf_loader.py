"""
pdf_loader.py
-------------

Getting text out of a PDF file.
For that we need a library (pypdf) to open it and pull the readable text out, page by page.

"""

from pypdf import PdfReader


def load_pdf(file_path_or_buffer):
    """
    WHY we keep the page number attached to each chunk of text:
    Later, when DocBot answers a question, we want to be able to say
    "this came from page 3". It can also help users to verify the answer by checking the original document.
    """
    reader = PdfReader(file_path_or_buffer)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if text:  # skip pages that turned out to have no readable text
            pages_text.append({"page": page_number, "text": text})

    return pages_text


def load_multiple_pdfs(file_paths):
    """
    Since DocBot supports uploading more than one PDF at a time, this function
    takes a LIST of PDF paths and returns one combined list of pages, remembering which
    document each page came from.
    """
    all_pages = []
    for path in file_paths:
        file_name = path.split("/")[-1]
        pages = load_pdf(path)
        for p in pages:
            p["source"] = file_name
        all_pages.extend(pages)
    return all_pages


if __name__ == "__main__":
    import config
    import os

    sample_files = [
        os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)
        if f.endswith(".pdf")
    ]
    pages = load_multiple_pdfs(sample_files)
    print(f"Loaded {len(pages)} pages from {len(sample_files)} PDF(s).")
    print("--- Sample of first page ---")
    print(pages[0]["source"], "| page", pages[0]["page"])
    print(pages[0]["text"][:300])
