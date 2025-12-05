import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

POPPLER_PATH = r"C:\poppler\Library\bin"  # ← your install path


# ------------------- PDF + OCR -------------------
def extract_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t

        if len(text) > 500:  # PDF probably text-based
            return text
        else:
            return ocr_pdf(file_path)
    except:
        return ocr_pdf(file_path)

def ocr_pdf(file_path):
    pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
    text = ""
    for p in pages:
        text += pytesseract.image_to_string(p)
    return text


# ------------------- AI SUMMARY -------------------
def summarize(text, specialty, legal, focus):
    return client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role":"system","content":f"""
            You are a medical-legal summarization engine generating structured IME reports.
            
            Specialty focus: {specialty}
            Legal format: {legal}
            Requested content: {focus}

            Extract relevant details only.
            No unnecessary narrative — structured clinical data preferred.
            """},
            {"role":"user","content":text}
        ]
    ).choices[0].message.content


# ------------------- STREAMLIT UI -------------------
st.title("⚕ IME Medical Record Summarizer")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

specialty = st.selectbox("Select Specialty", [
    "Orthopedics", "Cardiology", "Pediatrics", "Gastroenterology", 
    "Neurology", "General Medicine"
])

legal = st.selectbox("Case Type Format", [
    "Standard Medical Summary",
    "Worker's Compensation",
    "Auto Accident",
    "Disability / Insurance Review"
])

focus = st.selectbox("Summary Content Mode", [
    "Full Case Summary",
    "Imaging Results Only",
    "Treatment History Only",
    "Physical Exam Findings Only"
])

if uploaded and st.button("Generate Summary"):
    with open("temp.pdf","wb") as f:
        f.write(uploaded.read())

    st.write("📄 Extracting PDF… This may take a moment…")
    text = extract_text("temp.pdf")

    st.write("🧠 Generating Summary…")
    result = summarize(text, specialty, legal, focus)

    st.success("Summary complete:")
    st.write(result)

    st.download_button("⬇ Download Summary", result, "summary.txt")
