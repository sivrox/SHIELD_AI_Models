# ======================================================
# SAFETY LAYER (USED BY CHATBOT + ANALYZER)
# ======================================================

PROHIBITED_PHRASES = [
    "you have",
    "you are diagnosed",
    "diagnosis is",
    "diagnosed with",
    "prescribe",
    "prescription",
    "take this medication",
    "start taking",
    "stop taking",
    "treatment plan",
    "prognosis",
    "this will cure"
]

EMERGENCY_KEYWORDS = {
    "CRITICAL": [
        "chest pain",
        "difficulty breathing",
        "shortness of breath",
        "cardiac arrest",
        "stroke",
        "loss of consciousness",
        "seizure",
        "face drooping",
        "slurred speech",
        "one-sided weakness"
    ],
    "HIGH": [
        "very high blood pressure",
        "severe headache",
        "vision loss",
        "irregular heartbeat",
        "fainting",
        "confusion"
    ]
}

DISCLAIMER = (
    "\n\nMedical Disclaimer:\nThis information is provided for educational and informational purposes only and is not intended to serve as medical advice, diagnosis, or treatment.\n\n"
    "Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment, especially if you experience concerning symptoms or significant changes in your health."
)

def detect_emergency(text):
    text = text.lower()
    for level, keywords in EMERGENCY_KEYWORDS.items():
        for word in keywords:
            if word in text:
                return level
    return None


def severity_warning(level):
    if level == "CRITICAL":
        return (
            "MEDICAL EMERGENCY WARNING\n"
            "The information provided suggests a potentially life-threatening situation. "
            "Please seek immediate medical attention or contact emergency services."
        )
    elif level == "HIGH":
        return (
            "URGENT MEDICAL WARNING\n"
            "The findings may indicate a serious condition. "
            "Prompt evaluation by a healthcare professional is strongly advised."
        )
    return ""


def contains_prohibited_language(text):
    text = text.lower()
    for phrase in PROHIBITED_PHRASES:
        if phrase in text:
            return True
    return False


def apply_post_safety(answer):
    if contains_prohibited_language(answer):
        return (
            "The response was restricted due to medical safety policies."
            + DISCLAIMER
        )
    return answer + DISCLAIMER