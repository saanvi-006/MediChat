"""
MediChat - FINAL main.py (with memory)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models/model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models/vectorizer.pkl")

# ── Load ML ───────────────────────────────────────────
model = pickle.load(open(MODEL_PATH, "rb"))
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))

from src.symptom_normalizer import normalize_input
from src.llm import call_gemini_multi

# ── FastAPI ───────────────────────────────────────────
app = FastAPI(title="MediChat API")

# ── Request Models ────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []   # ✅ conversation memory

# ── Severity ──────────────────────────────────────────
def get_severity(user_input: str, confidence: float) -> str:
    text = user_input.lower()

    severe_keywords = [
        "severe", "extreme", "unbearable",
        "can't breathe", "chest pain", "blood",
        "fainting", "collapse"
    ]

    moderate_keywords = [
        "pain", "fever", "vomiting", "persistent",
        "high", "bad", "worse"
    ]

    if any(word in text for word in severe_keywords):
        return "Severe"

    if any(word in text for word in moderate_keywords):
        return "Moderate"

    if confidence > 85:
        return "Moderate"
    elif confidence > 60:
        return "Mild"
    else:
        return "Mild"

# ── Guidance ──────────────────────────────────────────
def get_guidance(category: str):
    guidance_map = {
        "Respiratory": {
            "advice": "Rest, stay hydrated, and monitor breathing.",
            "doctor": "Consult a doctor if breathing difficulty persists."
        },
        "Digestive": {
            "advice": "Eat light food and stay hydrated.",
            "doctor": "See a doctor if vomiting or pain worsens."
        },
        "Mental": {
            "advice": "Rest and manage stress.",
            "doctor": "Seek help if symptoms persist."
        },
        "Musculoskeletal": {
            "advice": "Rest the affected area.",
            "doctor": "Consult a doctor if pain continues."
        },
        "Skin": {
            "advice": "Keep area clean and avoid scratching.",
            "doctor": "See a doctor if spreading."
        },
        "General": {
            "advice": "Rest and stay hydrated.",
            "doctor": "Consult a doctor if symptoms worsen."
        }
    }

    return guidance_map.get(category, guidance_map["General"])

# ── Fallback ──────────────────────────────────────────
def generate_fallback(prediction: dict) -> str:
    return (
        f"{prediction['category']} issue with {prediction['severity']} severity. "
        f"{prediction['advice']}"
    )

# ── ROOT ──────────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "MediChat API running 🚀"}

# ── CHAT ──────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    user_input = request.message

    # ── ML prediction ─────────────────────────────
    norm = normalize_input(user_input)
    vec = vectorizer.transform([norm])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max() * 100

    severity = get_severity(user_input, prob)
    guidance = get_guidance(pred)

    prediction = {
        "category": pred,
        "severity": severity,
        "confidence": round(prob, 2),
        "advice": guidance["advice"]
    }

    # ── conversation history ─────────────────────
    history_text = "\n".join(request.history)

    # ── LLM call ────────────────────────────────
    reply = call_gemini_multi(user_input, prediction, history_text)

    # ── fallback ────────────────────────────────
    if not reply:
        reply = generate_fallback(prediction)

    return {
        "reply": reply,
        "category": pred,
        "severity": severity,
        "confidence": round(prob, 2)
    }