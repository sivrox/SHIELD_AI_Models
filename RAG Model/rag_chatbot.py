import os
import time
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from safety_layer import detect_emergency, severity_warning, apply_post_safety
from dotenv import load_dotenv

load_dotenv()

# ======================================================
# CONFIGURATION
# ======================================================

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

MODEL_ID = "gemini-2.5-flash-preview-09-2025"
DB_PATH = os.getenv("VECTOR_DB_PATH", "shield_medical_db")

# ======================================================
# CHATBOT
# ======================================================

def ask_shield_chatbot(user_query):
    # ---------- SAFETY PRE-CHECK ----------
    severity = detect_emergency(user_query)
    if severity:
        return severity_warning(severity) + apply_post_safety("")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    docs = db.similarity_search(user_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    system_instruction = f"""
You are the S.H.I.E.L.D. Medical Information Assistant.

RULES:
- Provide informational explanations only
- Do NOT diagnose or prescribe
- Use ONLY the provided clinical guidelines
- If information is missing, clearly say so

CLINICAL GUIDELINES:
{context}
"""

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"

    response = requests.post(url, json=payload)
    if response.status_code != 200:
        return "The system is temporarily unavailable."

    answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return apply_post_safety(answer)


if __name__ == "__main__":
    q = input("Ask our AI...\n")
    print(ask_shield_chatbot(q))