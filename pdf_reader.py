# modules/pdf_reader.py
import fitz  # PyMuPDF


def extract_text_from_pdf(file_handle):
    text = ""
    try:
        doc = fitz.open(stream=file_handle.read(), filetype="pdf")

        # Add a check to ensure the document has pages
        if doc.page_count == 0:
            print("PDF document has no pages.")
            return ""

        for page in doc:
            text += page.get_text()

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

    return text
