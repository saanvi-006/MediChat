"""
MediChat - llm.py (Multi Gemini + Memory)
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# ── Model list ───────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]

# ── Main function ────────────────────────────────────
def call_gemini_multi(user_input: str, prediction: dict, history: str) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
You are MediChat, a calm and helpful AI health assistant.

Conversation so far:
{history}

User symptoms:
{user_input}

Model prediction:
- Category: {prediction['category']}
- Severity: {prediction['severity']}

Your role:
- Give clear, simple, and safe health guidance
- DO NOT diagnose diseases

Instructions:
- Keep response short and user-friendly
- Be calm and reassuring
- Suggest doctor if needed

IMPORTANT:
Respond ONLY in this format:

Advice:
<text>

Precautions:
<points>

Doctor:
<when to see doctor>

Response:
"""

    # ── Try models one by one ─────────────────────
    for model_name in GEMINI_MODELS:
        try:
            print(f"Trying: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            text = getattr(response, "text", None)

            if text and len(text.strip()) > 10:
                print(f"[LLM SUCCESS] {model_name}")
                return text

        except Exception as e:
            print(f"[LLM FAIL] {model_name} → {e}")
            continue

    return None