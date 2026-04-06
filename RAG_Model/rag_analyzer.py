#AI Report Analyzer

import os
import time
import base64
import mimetypes
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from safety_layer import detect_emergency, severity_warning, apply_post_safety
from dotenv import load_dotenv

load_dotenv()

#CONFIGURATION
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

MODEL_ID = "gemini-2.5-flash-preview-09-2025"
DB_PATH = os.getenv("VECTOR_DB_PATH", "shield_medical_db")

#HELPER FUNCTIONS
def encode_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_medical_file(file_path):
    mime, _ = mimetypes.guess_type(file_path)
    base64_data = encode_file(file_path)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    guidelines = "\n\n".join(
        d.page_content for d in db.similarity_search(
            "normal ranges blood pressure cholesterol heart health", k=4
        )
    )

    system_instruction = f"""
You are the S.H.I.E.L.D. Medical Report Explanation Assistant.

ROLE:
- Explain report findings in an educational manner
- Highlight values outside typical reference ranges
- Do NOT diagnose or predict outcomes
- Suggest questions to discuss with a doctor

CLINICAL GUIDELINES:
{guidelines}
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": "Please explain this medical report."},
                {"inlineData": {"mimeType": mime, "data": base64_data}}
            ]
        }],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        return "The report could not be analyzed."

    answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    severity = detect_emergency(answer) #Safety Post-Check
    if severity:
        return severity_warning(severity) + apply_post_safety(answer)

    return apply_post_safety(answer)


if __name__ == "__main__":
    file_path = input("Enter report file path:\n")
    print(analyze_medical_file(file_path))
