"""
MediChat - main.py
------------------
FastAPI backend with:
- ML prediction
- Red flag detection
- LLM integration
- Conversation memory
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

from src.symptom_normalizer import normalize_input
from src.llm import call_gemini_multi
from src.utils import check_red_flags, get_severity

app = FastAPI()

# ── Load model + vectorizer ───────────────────────────────

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# ── Request schema ────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[str] = []


# ── Summary endpoint ──────────────────────────────────────

@app.post("/summary")
def get_summary(req: ChatRequest):
    from llm import call_gemini_multi

    summary_prompt = {
        "category": "General",
        "severity": "Mild"
    }

    reply = call_gemini_multi(
        "Summarize this conversation briefly:\n" + "\n".join(req.history),
        summary_prompt,
        req.history
    )

    return {"summary": reply}


# ── Main chat endpoint ────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest):

    user_input = req.message

    # 🚨 Red flag override
    if check_red_flags(user_input):
        return {
            "reply": "⚠️ This may be a serious condition. Please seek immediate medical attention.",
            "category": "Emergency",
            "severity": "Severe",
            "confidence": 100.0,
        }

    # ── Normalize input ────────────────────────────────
    norm = normalize_input(user_input)

    # ── ML prediction ─────────────────────────────────
    vec = vectorizer.transform([norm])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0].max() * 100

    severity = get_severity(proba)

    prediction = {
        "category": pred,
        "severity": severity
    }

    # ── LLM call ──────────────────────────────────────
    reply = call_gemini_multi(user_input, prediction, req.history)

    if not reply:
        reply = "Please rest, stay hydrated, and monitor your symptoms."

    return {
        "reply": reply,
        "category": pred,
        "severity": severity,
        "confidence": round(float(proba), 2),
    }