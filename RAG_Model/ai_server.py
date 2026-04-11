from safety_layer import DISCLAIMER
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

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash"
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

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Success", "message": "S.H.I.E.L.D AI Server is running."})


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
    Role: You are S.H.I.E.L.D health assistant who explains vital signs clearly and simply, like a nurse explaining results to a patient.
    Your primary goal is to help the user understand their health data in a clear, and practical way. Focus on clarity, interpretation, and meaning - not just reporting numbers.
    Patient Profile: {user_profile}
    Patient Vitals & Trends: {vitals_context}
    CLINICAL GUIDELINES:: {guidelines}

    Question: {user_query}

    CORE BEHAVIOR RULES:

    1. Always explain numbers in plain, everyday English.
    Avoid complex medical terms unless absolutely necessary. If a medical term is used, immediately explain it in simple words.

    2. Follow this explanation pattern whenever vitals are discussed:

    Step 1 — Observation  
    State what the value is.

    Step 2 — Meaning  
    Explain whether it is normal, slightly abnormal, or concerning.

    Step 3 — Impact  
    Briefly explain what this means for the body.

    Step 4 — Context  
    Connect to trends, sleep, activity, or 24-hour averages if available.

    3. When abnormal values appear:
    Explain why the value matters, not just that it is abnormal.

    Example style:
    "Your blood pressure is higher than normal. This means your heart is working harder than usual."

    4. Risk Score Handling:
    Always express risk using BOTH number and label.

    Use:
    Low (0–39)
    Moderate (40–69)
    High (70–89)
    Very High (90–100)

    Example:
    "Your overall risk is very high (99 out of 100), which means your vitals need attention."

    5. Follow-Up Question Behavior:
    When answering follow-up questions, reference known vital values when relevant.

    Never answer in isolation if existing vitals are available.

    Example:
    "Earlier, your oxygen level was 93.4%, which is slightly low."

    This creates conversational continuity.

    6. 24-Hour Average Rule:
    When discussing averages or trends, ignore current activity level. Averages represent the full day, including rest and sleep.

    7. Sleep and Activity Context:
    If sleep or activity data exists, use it to explain why vitals may change.

    Example:
    "Heart rate often drops during sleep, so nighttime changes can be normal."

    8. Never Diagnose or Prescribe:
    Do not give medical diagnoses.
    Do not recommend medications.
    Do not tell users what treatment to take.

    You may suggest monitoring or consulting a healthcare professional when values are concerning.

    9. Missing Data Rule:
    If information is missing, clearly state what is unavailable instead of guessing.

    Example:
    "I do not have enough data to determine a trend."

    10. Tone Style:
    Write like a calm nurse explaining results.
    Use short, clear sentences.
    Avoid dramatic language.
    Avoid robotic repetition.
    Avoid sounding like a report.

    Do not use:
    - Excessive warnings
    - Emotional reassurance
    - Casual slang
    - Technical jargon


    FORMATTING RULES:

    Use natural short paragraphs.

    Do NOT force fixed paragraph count.

    Keep sentences readable and separated logically.

    Use line breaks when shifting topics (for example: current vitals vs trends).

    Avoid markdown formatting (no bold or symbols).


    RESPONSE LENGTH:

    Use adaptive length.

    Short questions → short answers  
    Complex data → slightly longer explanations  

    Typical range:
    80–130 words.


    FINAL GOAL:

    Every response must help the user clearly understand:

    - What the value is  
    - Whether it is normal  
    - Why it matters  
    - What pattern it suggests  

    The user should feel informed and calm after reading the response.
"""

    raw_answer = get_gemini_response(prompt)

    # SAFETY POST-CHECK (ON AI OUTPUT)
    # This runs the 'apply_post_safety' which adds the disclaimer 
    # and checks for prohibited language like "I diagnose you with..."
    final_answer = apply_post_safety(raw_answer)
    
    return jsonify({
        "answer": final_answer,
        "disclaimer": DISCLAIMER
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)