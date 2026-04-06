"""
Test file for llm.py
--------------------
Run: python MediChat/src/test_llm.py
"""

from llm import call_gemini_multi

# ── Fake prediction (same format as main.py)
prediction = {
    "category": "Digestive",
    "severity": "Moderate",
    "confidence": 85.0,
    "advice": "Eat light food, stay hydrated, and avoid spicy meals."
}

# ── Test input
user_input = "I have stomach pain and feel like vomiting"

# ── Call LLM
response = call_gemini_multi(user_input, prediction)

# ── Output
print("\n=== LLM TEST RESULT ===\n")

if response:
    print("✅ LLM Response:\n")
    print(response)
else:
    print("❌ All models failed → fallback will be used")