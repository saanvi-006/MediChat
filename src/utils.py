"""
Utility functions: red flag detection + severity logic
"""

RED_FLAGS = [
    "chest pain",
    "difficulty breathing",
    "can't breathe",
    "severe bleeding",
    "unconscious",
    "fainting",
]


def check_red_flags(text: str) -> bool:
    text = text.lower()
    return any(flag in text for flag in RED_FLAGS)


def get_severity(confidence: float) -> str:
    if confidence >= 80:
        return "Severe"
    elif confidence >= 60:
        return "Moderate"
    else:
        return "Mild"