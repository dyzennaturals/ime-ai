import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
import textwrap
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\library\bin"

# ---------- Setup ----------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file is in this folder.")

client = OpenAI(api_key=api_key)


# ---------- Helpers ----------

def load_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            all_text.append(page_text)
    return "\n\n".join(all_text)

def ocr_pdf(pdf_path: str) -> str:
    """Use OCR to extract text from a scanned/image-based PDF."""
    print("Running OCR on PDF pages...")
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    texts = []

    total_pages = len(pages)
    for i, page in enumerate(pages, start=1):
        print(f"OCR page {i}/{total_pages}...")
        text = pytesseract.image_to_string(page)
        texts.append(text)

    return "\n\n".join(texts)

def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """Split a long string into chunks of up to max_chars."""
    chunks = []
    current = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current_len + len(paragraph) > max_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def call_model(messages, temperature: float = 0.1) -> str:
    """Wrapper to call the chat model."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # you can change to "gpt-4.1" later if you want
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------- Core logic ----------

def summarize_chunk(chunk_text: str) -> str:
    """Summarize one chunk of medical records with a structured format."""
    system_prompt = (
        "You are assisting an orthopedic surgeon performing an independent medical examination (IME). "
        "You will be given a chunk of medical records. Extract ONLY clear, factual information relevant "
        "to orthopedic evaluation and litigation. Avoid speculation."
    )

    user_prompt = f"""
    From the following medical record text, extract and summarize the information using this structure:

    1. Chief Complaints / Symptoms
    2. Mechanism of Injury / Onset
    3. Past Medical & Surgical History (orthopedic-relevant)
    4. Imaging Findings (X-ray, MRI, CT, etc.)
    5. Prior Treatments (PT, injections, surgery, meds)
    6. Work Status / Restrictions (if mentioned)
    7. Inconsistencies or Red Flags (if clearly documented only)

    TEXT:
    {chunk_text}
    """

    return call_model(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": textwrap.dedent(user_prompt)},
        ]
    )


def combine_summaries(chunk_summaries: List[str]) -> str:
    """Take multiple chunk-level summaries and merge them into a final IME synopsis."""
    system_prompt = (
        "You are preparing a concise, structured pre-exam summary for an orthopedic IME. "
        "You will be given multiple partial summaries from different segments of the records. "
        "Merge them into a single, organized synopsis without repetition."
    )

    joined = "\n\n--- CHUNK SUMMARY ---\n\n".join(chunk_summaries)

    user_prompt = f"""
    Merge the following chunk-level summaries into one coherent, non-redundant report.

    Final structure:

    1. Patient Demographics (if available)
    2. Chief Complaints / Reported Symptoms
    3. Mechanism of Injury / Onset History
    4. Relevant Past Medical & Surgical History
    5. Imaging & Objective Findings
    6. Treatments to Date
    7. Functional Status & Work Capacity
    8. Notable Inconsistencies / Red Flags
    9. Key Points for IME Examiner

    CHUNK SUMMARIES:
    {joined}
    """

    return call_model(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": textwrap.dedent(user_prompt)},
        ]
    )


def summarize_pdf(pdf_path: str) -> str:
    """Main function: PDF -> final IME-style summary."""
    print(f"Loading PDF: {pdf_path}")
    full_text = load_pdf_text(pdf_path)
    print("PDF loaded, length (normal extract):", len(full_text), "characters")

    # If we got almost no text, try OCR instead
    if len(full_text.strip()) < 1000:
        print("Very little text extracted with normal method. Trying OCR...")
        full_text = ocr_pdf(pdf_path)
        print("OCR text length:", len(full_text), "characters")

    # DEBUG: save whatever text we actually extracted
    with open("raw_pdf_text.txt", "w", encoding="utf-8") as debug_file:
        debug_file.write(full_text)

    if not full_text.strip():
        return "No text could be extracted from this PDF, even with OCR."

    chunks = chunk_text(full_text, max_chars=6000)
    print(f"Split into {len(chunks)} chunk(s)")

    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"\nSummarizing chunk {i}/{len(chunks)}...")
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)

    print("\nCombining chunk summaries into final report...")
    final_summary = combine_summaries(chunk_summaries)
    return final_summary



# ---------- Run from command line ----------

if __name__ == "__main__":
    pdf_path = input("Enter path to the medical records PDF: ").strip()

    if not os.path.exists(pdf_path):
        print("File not found:", pdf_path)
        raise SystemExit(1)

    final = summarize_pdf(pdf_path)

    output_path = "ime_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final)

    print("\nDone. Summary saved to:", output_path)
