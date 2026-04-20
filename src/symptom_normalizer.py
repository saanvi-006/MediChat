"""
MediChat - symptom_normalizer.py  (v4 — minimal & unbiased)
----------------------------------------------------------------
Goal:
- Preserve meaning
- Avoid bias injection
- Keep ML model in control

Principles:
- Phrase map → minimal expansion
- Word fallback → 1–2 closest tokens ONLY
- No category forcing
"""

import re

# ─────────────────────────────────────────────────────────────
# PASS 1 — PHRASE MAP (MINIMAL, NO OVER-EXPANSION)
# ─────────────────────────────────────────────────────────────

PHRASE_MAP = [

    # Digestive
    ("loose motions", "diarrhea"),
    ("loose stool", "diarrhea"),
    ("watery stool", "diarrhea"),
    ("blood in stool", "blood in stool"),
    ("stomach pain", "abdominal pain"),
    ("stomach ache", "abdominal pain"),
    ("belly pain", "abdominal pain"),
    ("abdominal pain", "abdominal pain"),
    ("acid reflux", "heartburn"),
    ("food poisoning", "nausea vomiting diarrhea"),
    ("throwing up", "vomiting"),

    # Respiratory
    ("runny nose", "nasal congestion"),
    ("stuffy nose", "nasal congestion"),
    ("sore throat", "sore throat"),
    ("difficulty breathing", "shortness of breath"),
    ("short of breath", "shortness of breath"),
    ("chest tightness", "chest tightness"),

    # Musculoskeletal
    ("body ache", "body ache"),
    ("body pain", "body ache"),
    ("back pain", "back pain"),
    ("joint pain", "joint pain"),

    # Mental
    ("panic attack", "anxiety"),
    ("can't sleep", "insomnia"),

    # General
    ("high fever", "fever"),
    ("feeling sick", "feeling ill"),
    ("after eating", "after eating"),
    ("after meals", "after eating"),
    ("with fever", "fever"),
    ("along with fever", "fever"),

]

PHRASE_MAP = sorted(PHRASE_MAP, key=lambda x: len(x[0]), reverse=True)


# ─────────────────────────────────────────────────────────────
# PASS 2 — WORD FALLBACK (STRICTLY MINIMAL)
# ─────────────────────────────────────────────────────────────

WORD_FALLBACK = {

    # Respiratory
    "feverish": "fever",
    "congested": "congestion",
    "stuffy": "congestion",
    "scratchy": "throat",
    "hoarse": "voice",
    "breathless": "breath",

    # Digestive
    "nauseous": "nausea",
    "nauseated": "nausea",
    "queasy": "nausea",
    "vomitting": "vomiting",
    "puking": "vomiting",
    "bloated": "bloating",
    "gassy": "gas",
    "constipated": "constipation",

    # General / systemic
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhausted": "fatigue",
    "weak": "weakness",
    "unwell": "ill",
    "fever": "fever",

    # Pain / musculo
    "aching": "pain",
    "achy": "pain",
    "stiff": "stiffness",
    "numb": "numbness",
    "tingly": "tingling",

    # Skin
    "itchy": "itching",
    "rashy": "rash",
    "tired": "fatigue",
    "weak": "weakness",
    "hurts": "pain",

}


# ─────────────────────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "i","have","am","a","an","the","my","me","is","are","and","or",
    "with","some","feel","feeling","been","get","got",
    "very","really","so","too","bit","little","lot",
    "since","days","weeks","ago","also","still","always","can",
    "cant","dont","do","not","of","in","on","at","to","for",
    "bad","severe","mild","worse","worst",
    "it","its","you","we","they","he","she",
}


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def normalize_input(user_text: str) -> str:
    """
    3-pass normalization:
    1. Phrase match (minimal expansion)
    2. Word fallback (minimal mapping)
    3. Raw token keep (if meaningful)
    """

    text = user_text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    expanded_tokens = []
    covered_positions = set()

    # ── PASS 1: Phrase mapping ──
    for phrase, tokens in PHRASE_MAP:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx == -1:
                break

            end = idx + len(phrase)
            before_ok = (idx == 0 or text[idx - 1] == " ")
            after_ok = (end == len(text) or text[end] == " ")

            if before_ok and after_ok:
                positions = set(range(idx, end))
                if not positions & covered_positions:
                    expanded_tokens.append(tokens)
                    covered_positions |= positions

            start = end

    # ── PASS 2 & 3: Word handling ──
    pos = 0
    for word in text.split():
        word_start = text.find(word, pos)
        word_end = word_start + len(word)
        pos = word_end

        word_positions = set(range(word_start, word_end))

        # Skip if already handled
        if word_positions & covered_positions:
            continue

        # Skip stopwords
        if word in STOPWORDS:
            continue

        # Fallback mapping
        if word in WORD_FALLBACK:
            expanded_tokens.append(WORD_FALLBACK[word])
        else:
            # Keep raw word (important!)
            expanded_tokens.append(word)

    return " ".join(expanded_tokens)