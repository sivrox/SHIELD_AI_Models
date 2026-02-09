import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from safety_layer import detect_emergency, severity_warning, apply_post_safety

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
DB_PATH = os.path.join(os.path.dirname(__file__), "shield_medical_db")

# Initialize RAG
print("Loading local Embedding Model & Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

def get_gemini_response(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    return "Error connecting to Gemini API."

@app.route('/api/v1/ask-shield', methods=['POST'])
def ask_ai():
    data = request.json
    user_query = data.get('question', '')
    vitals_context = data.get('vitals_snapshot', 'No recent vitals.')
    user_profile = data.get('profile', 'General Patient.')

    # SAFETY PRE-CHECK (ON USER INPUT) 
    emergency_level = detect_emergency(user_query)
    if emergency_level:
        # If user mentions chest pain, etc., return warning immediately
        return jsonify({
            "answer": severity_warning(emergency_level),
            "disclaimer": "This is an automated safety alert."
        })

    # RAG SEARCH
    # Search local Chroma DB for clinical guidelines
    docs = db.similarity_search(user_query, k=2)
    guidelines = "\n\n".join([d.page_content for d in docs])

    # GENERATE RESPONSE
    prompt = f"""
Role: S.H.I.E.L.D. Medical Assistant.
Patient Profile: {user_profile}
Patient Vitals: {vitals_context}
Guidelines: {guidelines}

Question: {user_query}

Instructions:
1. Use the Guidelines to explain the Vitals contextually.
2. Highlight values that look abnormal according to Guidelines.
3. NEVER diagnose or prescribe. 
4. Be concise (max 3-4 sentences).
"""

    raw_answer = get_gemini_response(prompt)

    # SAFETY POST-CHECK (ON AI OUTPUT)
    # This runs the 'apply_post_safety' which adds the disclaimer 
    # and checks for prohibited language like "I diagnose you with..."
    final_answer = apply_post_safety(raw_answer)
    
    return jsonify({
        "answer": final_answer
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)