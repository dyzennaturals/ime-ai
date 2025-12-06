import os
import io
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from fpdf import FPDF

# 👉 Point pytesseract to the actual tesseract.exe location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ------------------- CONFIG / CLIENT -------------------
load_dotenv()  # loads OPENAI_API_KEY if present in .env (locally)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Update this if your Poppler path is different
POPPLER_PATH = r"C:\poppler\Library\bin"


# ------------------- PDF + OCR HELPERS -------------------
def extract_text(file_path: str) -> str:
    """
    Extracts text page-by-page with page tagging.
    If text is too light to read, automatically falls to OCR.
    """
    try:
        reader = PdfReader(file_path)
        output = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            output.append(f"<<PAGE {i}>>\n{text.strip()}\n")

        combined = "\n".join(output)

        # If PDF is basically empty → run OCR instead
        if len(combined.replace("\n","").strip()) < 400:
            return extract_text_ocr(file_path)  # new function below

        return combined

    except:
        return extract_text_ocr(file_path)  # always fallback



def extract_text_ocr(file_path: str) -> str:
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    output = []

    for i, img in enumerate(pages, start=1):
        text = pytesseract.image_to_string(img)
        output.append(f"<<PAGE {i}>>\n{text.strip()}\n")

    return "\n".join(output)



# ------------------- AI SUMMARY -------------------
def summarize(text: str, specialty: str, legal: str, focus: str) -> str:
    # Tailor instructions based on focus selection
    if focus == "Full Case Summary":
        focus_instruction = (
            "Include and fully populate ALL sections listed below if the information exists. "
            "If a section has no relevant information in the records, write 'INSUFFICIENT DATA IN RECORDS' under that heading."
        )
    elif focus == "Imaging Results Only":
        focus_instruction = (
            "Focus primarily on IMAGING SUMMARY. "
            "Only include very brief one-line context in other sections if absolutely necessary to interpret imaging."
        )
    elif focus == "Treatment History Only":
        focus_instruction = (
            "Focus primarily on TREATMENT SUMMARY. "
            "Keep other sections minimal, only enough for context."
        )
    elif focus == "Physical Exam Findings Only":
        focus_instruction = (
            "Focus primarily on PHYSICAL EXAM / FUNCTIONAL STATUS. "
            "Summarize other sections only if needed to interpret the exam."
        )
    else:
        focus_instruction = (
            "Use your best judgment but prefer a balanced full-case summary."
        )

    # Tailor instructions based on legal / case type
    if legal == "Standard Medical Summary":
        legal_instruction = (
            "Maintain a neutral, clinical tone. Do NOT speculate beyond the documented record."
        )
    elif legal == "Worker's Compensation":
        legal_instruction = (
            "Emphasize mechanism of injury, work-related causation IF documented, "
            "functional limitations, and work status/restrictions. "
            "If causation is not explicitly stated, do NOT invent it; instead mark as 'NOT CLEARLY STATED IN RECORDS'."
        )
    elif legal == "Auto Accident":
        legal_instruction = (
            "Emphasize mechanism of injury (MVA details if present), temporal relationship of symptoms to the accident, "
            "documented aggravation vs. pre-existing conditions, and functional impact. "
            "Do NOT fabricate causation; only report what is clearly documented."
        )
    elif legal == "Disability / Insurance Review":
        legal_instruction = (
            "Emphasize functional limitations, objective findings, imaging corroboration, "
            "treatment response, and any comments regarding ability to work or perform ADLs. "
            "Flag inconsistencies between subjective complaints and objective findings if present in the record."
        )
    else:
        legal_instruction = "Maintain a neutral medical-legal tone."

    system_prompt = f"""
You are a medical-legal summarization engine generating structured IME-style reports.

Specialty focus: {specialty}
Case / legal context: {legal}
Requested summary mode: {focus}

{focus_instruction}

{legal_instruction}

All text you receive includes explicit page tags formatted as:

<<PAGE 1>>
<<PAGE 2>>
<<PAGE 3>> ...

When reporting imaging, procedures, exam findings, or key diagnoses,
cite page numbers automatically like (p. 3) whenever possible.


Use the following EXACT section structure and headings in this order.
Use clear, concise bullet points or short paragraphs under each heading.
If the information is not available in the records, write 'INSUFFICIENT DATA IN RECORDS' under that heading.

1. PATIENT OVERVIEW
   - Age, sex (if available)
   - Chief complaint(s)
   - Mechanism of injury or onset (include dates if available)
   - Referral reason or purpose of evaluation if stated.

2. HISTORY OF PRESENT ILLNESS
   - Onset and progression of symptoms
   - Location, character, severity, and radiation of pain or primary symptoms
   - Aggravating/alleviating factors
   - Impact on function and activities of daily living (ADLs).

3. PRIOR MEDICAL / SURGICAL HISTORY
   - Relevant prior injuries or conditions, especially related to {specialty}
   - Prior surgeries or major interventions
   - Other comorbidities that affect current condition.

4. MEDICATIONS & ALLERGIES
   - Current medications and dosages if documented
   - Prior medication trials and responses
   - Documented allergies and reactions.

5. TREATMENT SUMMARY
   - Physical therapy, chiropractic, injections, surgeries, and other treatments
   - Dates or approximate time frames
   - Documented response or failure of each treatment.

6. IMAGING SUMMARY
   - Key findings from MRI, CT, X-ray, ultrasound, EMG/NCV, etc.
   - Levels/structures involved (e.g., L4-5 disc protrusion, medial meniscus tear)
   - Severity and any radiologist impressions relevant to {specialty}.

7. PHYSICAL EXAM / FUNCTIONAL STATUS
   - Objective exam findings (ROM, strength, neurologic deficits, gait, special tests)
   - Functional status (ability to work, stand, walk, lift, etc.) if documented
   - Any noted inconsistencies between complaints and exam (only if explicitly mentioned).

8. ASSESSMENT / IMPRESSION
   - Most consistent diagnosis/diagnoses based on documented records
   - Relationship of imaging and exam findings to reported complaints
   - For {specialty}, highlight the primary anatomic and functional problem(s).

9. PLAN / RECOMMENDATIONS
   - Documented recommendations from treating providers (e.g., further imaging, surgeries, PT, pain management)
   - Any follow-up plans or long-term management strategies mentioned in the record.

10. LEGAL / ADMINISTRATIVE NOTES (IF PRESENT)
   - Work status and restrictions (light duty, sedentary only, off work, etc.)
   - Comments regarding maximum medical improvement (MMI), permanent impairment, disability ratings if documented.
   - Any explicit statements about causation, apportionment, or secondary gain — ONLY if clearly documented in the text.

IMPORTANT RULES:
- Do NOT invent details. Only summarize what is actually present in the provided text.
- If you are unsure or the record is unclear, state that it is unclear or not documented.
- When you mention a key imaging finding, exam finding, or important treatment, include the page number in parentheses if you can infer it from the '===PAGE X===' markers, for example: (p. 3).
- When you mention a specific visit, surgery, or imaging study and a date is present in the text, include that date (e.g., 'MRI lumbar spine on 06/11/2024 (p. 4)').
- Keep the tone professional, neutral, and suitable for an IME physician or legal reviewer.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content


# ------------------- PDF EXPORT -------------------
# ===================== DYZEN-MED PDF BUILDER =====================
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import io
import textwrap

def generate_pdf(summary_text):
    buffer = io.BytesIO()

    pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))

    PAGE_WIDTH, PAGE_HEIGHT = letter
    LEFT_MARGIN = 0.75 * inch
    RIGHT_MARGIN = PAGE_WIDTH - 0.75 * inch
    TOP_MARGIN = PAGE_HEIGHT - 0.75 * inch
    LINE_HEIGHT = 14

    c = canvas.Canvas(buffer, pagesize=letter)
    PAGE_WIDTH, PAGE_HEIGHT = letter
    LEFT_MARGIN = 0.75 * inch
    TOP_MARGIN = PAGE_HEIGHT - 0.75 * inch
    y = TOP_MARGIN
    
    # ===== HEADER =====
    c.setFont("DejaVu", 18)
    c.drawString(LEFT_MARGIN, y, "DyZen-Med — AI Assisted Medical Review")
    y -= 0.35 * inch

    c.setFont("DejaVu", 10)
    c.drawString(LEFT_MARGIN, y, "Secure · Fast · Physician-Focused Documentation")
    y -= 0.25 * inch

    c.line(LEFT_MARGIN, y, PAGE_WIDTH - LEFT_MARGIN, y)   # horizontal rule
    y -= 0.35 * inch

    # ===== WATERMARK =====
    c.saveState()
    c.setFont("DejaVu", 55)             # Large watermark font
    c.setFillColorRGB(0, 0, 0, 0.08)    # Light opacity wash
    c.translate(PAGE_WIDTH/2, PAGE_HEIGHT/2)
    c.rotate(35)
    c.drawCentredString(0, 0, "DyZen-Med")
    c.restoreState()


    for line in summary_text.split("\n"):

        # Detect headers (1. PATIENT..., 2. HISTORY...)
        if line[:2].isdigit() and line[2] == ".":
            c.setFont("DejaVu", 11)
            wrapped = textwrap.wrap(line, width=95)
        else:
            c.setFont("DejaVu", 10)
            wrapped = textwrap.wrap(line, width=100)

        for seg in wrapped:
            if y < 1 * inch:  # new page threshold
                c.showPage()
                y = TOP_MARGIN
                c.setFont("DejaVu", 10)

            c.drawString(LEFT_MARGIN, y, seg)
            y -= LINE_HEIGHT

        y -= 5  # spacing between paragraph blocks

    c.save()
    buffer.seek(0)
    return buffer


# STREAMLIT PDF + TXT OUTPUT SECTION
# ==================================

# Only show export buttons if a summary was actually generated
if 'processed_report' in locals() and processed_report:

    # PDF EXPORT BUTTON
    if st.button("📄 Generate PDF"):
        try:
            pdf_data = generate_pdf(processed_report)
            st.download_button(
                label="⬇ Download DyZen-Med PDF Report",
                data=pdf_data,
                file_name="DyZen-Med_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"PDF Export Failed: {e}")

    # TXT EXPORT
    st.download_button(
        label="📝 Download TXT Summary",
        data=processed_report,
        file_name="DyZen-Med_Summary.txt"
    )
else:
    st.info("Upload a PDF and generate a summary to enable export options.")






# ------------------- STREAMLIT UI -------------------
st.title("DyZen-Med ⚕")
st.subheader("AI-Assisted IME Medical Record Summarizer")

st.markdown(
    "Upload a medical record PDF, choose the specialty and case type, and DyZen-Med will "
    "generate a structured, IME-style summary.\n\n"
    "**Note:** Only test or de-identified data should be used until HIPAA compliance is implemented."
)
st.markdown("---")

uploaded = st.file_uploader("📁 Upload Medical Record PDF", type=["pdf"])

specialty = st.selectbox("🩺 Select Specialty", [
    "Orthopedics", "Cardiology", "Neurology",
    "Gastroenterology", "Pediatrics", "General Medicine"
])

legal = st.selectbox("⚖ Case Type / Legal Context", [
    "Standard Medical Summary", "Worker's Compensation",
    "Auto Accident", "Disability / Insurance Review"
])

focus = st.selectbox("🧠 Summary Content Mode", [
    "Full Case Summary", "Imaging Results Only",
    "Treatment History Only", "Physical Exam Findings Only"
])

st.markdown("---")

if uploaded and st.button("Generate Summary"):
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())

    try:
        with st.spinner("📄 Extracting text (OCR if necessary)..."):
            text = extract_text(temp_path)

        if not text or len(text.strip()) < 50:
            st.error("Not enough readable text extracted. PDF may be image-only or low quality.")
        else:
            with st.spinner("🧠 Generating IME-style summary..."):
                result = summarize(text, specialty, legal, focus)

            st.success("Summary generated ✔")
            st.markdown(result)

            # ░▒ TXT Download ▒░
            st.download_button("⬇ Download Summary (TXT)",
                               result, "dyzen_med_summary.txt")

            # ░▒ PDF Download ▒░
            pdf_bytes = generate_pdf(result)
            if pdf_bytes:
                st.download_button(
                    "⬇ Download Branded PDF",
                    data=pdf_bytes,
                    file_name="DyZen-Med_Report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Processing error: {e}")

