"""
MediChat - llm.py (FINAL STABLE VERSION)
---------------------------------------
Multi-Gemini handler with strong prompt + safe fallback
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
]


def call_gemini_multi(user_input: str, prediction: dict, history: list[str]) -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    history_text = "\n".join(history[-6:])

    prompt = f"""
You are MediChat, a calm and helpful AI health assistant.

Conversation so far:
{history_text}

User's latest message:
{user_input}

Model context:
- Category: {prediction['category']}
- Severity: {prediction['severity']}

Your role:
- Talk like a real human assistant (NOT bullet points)
- Be conversational, natural, and supportive
- Give helpful, practical advice

Instructions:
- Keep response short (3–4 sentences)
- Do NOT use headings, labels, or bullet points
- Do NOT sound robotic or structured
- Gently guide the user on what to do next
- If severity is Moderate or Severe → suggest seeing a doctor naturally
- Use category only as loose context (do NOT rely on it blindly)

Conversation style:
- Start naturally (e.g., "It sounds like...", "Since you're feeling...")
- Give advice within the sentence (not as a list)
- End with ONE simple follow-up question if helpful

Response:
"""

    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.7},
            )

            text = getattr(response, "text", None)

            if text and len(text.strip()) > 10:
                print(f"[LLM SUCCESS] {model}")
                return text.strip()

        except Exception as e:
            print(f"[LLM FAIL] {model} → {e}")

    # Safe fallback
    return "I'm having trouble responding right now. Please try again."