"""
MediChat - symptom_normalizer.py
---------------------------------
Bridges the gap between what users type and what the model was trained on.

THE CORE PROBLEM:
  Training data uses structured symptom tokens like:
    "anxiety and nervousness", "stomach bloating", "shortness of breath"

  Users type casual phrases like:
    "anxiety stress can't sleep", "stomach hurts acid", "hard to breathe"

  These vocabularies don't overlap well, so TF-IDF treats unknown words
  as noise and falls back to the most common class ("General").

THE FIX:
  1. Expand user synonyms → training vocabulary tokens
  2. This lets TF-IDF find real signal in user input
  3. The model then predicts the correct category

USAGE:
  from symptom_normalizer import normalize_input

  text = normalize_input("i have anxiety and can't sleep, feeling stressed")
  # → "anxiety and nervousness insomnia restlessness emotional symptoms"
"""

# ── Synonym map: user word/phrase → one or more training symptom tokens ───────
#
# Keys   : words or short phrases a user might type (lowercase)
# Values : space-separated string of matching training symptom column names
#          (exact column names from the dataset)
#
# Design rules:
#   - Map to the CLOSEST training token(s), not necessarily exact
#   - Prefer more specific tokens over generic ones
#   - Multiple mappings help TF-IDF find stronger signal

SYNONYM_MAP = {

    # ── RESPIRATORY SYNONYMS ───────────────────────────────────────────────────
    "runny nose":           "nasal congestion coryza",
    "stuffy nose":          "nasal congestion sinus congestion",
    "blocked nose":         "nasal congestion sinus congestion",
    "nose blocked":         "nasal congestion sinus congestion",
    "congestion":           "nasal congestion sinus congestion congestion in chest",
    "sneezing":             "sneezing coryza",
    "cold":                 "nasal congestion coryza sore throat",
    "coughing":             "cough coughing up sputum",
    "mucus":                "coughing up sputum pus in sputum nasal congestion",
    "phlegm":               "coughing up sputum pus in sputum",
    "breathless":           "shortness of breath difficulty breathing",
    "breathlessness":       "shortness of breath difficulty breathing",
    "short of breath":      "shortness of breath difficulty breathing",
    "hard to breathe":      "shortness of breath difficulty breathing hurts to breath",
    "difficulty breathing": "difficulty breathing shortness of breath",
    "can't breathe":        "difficulty breathing shortness of breath",
    "wheezing":             "wheezing abnormal breathing sounds",
    "chest tightness":      "chest tightness congestion in chest",
    "chest congestion":     "congestion in chest chest tightness",
    "sore throat":          "sore throat throat irritation throat redness",
    "throat pain":          "sore throat throat irritation",
    "throat irritation":    "throat irritation sore throat drainage in throat",
    "hoarseness":           "hoarse voice difficulty speaking",
    "hoarse":               "hoarse voice difficulty speaking",
    "flu":                  "fever chills fatigue cough nasal congestion flu-like syndrome",
    "influenza":            "fever chills fatigue cough nasal congestion flu-like syndrome",
    "sinusitis":            "painful sinuses sinus congestion nasal congestion",
    "sinus pain":           "painful sinuses sinus congestion",
    "nosebleed":            "nosebleed",
    "snoring":              "apnea abnormal breathing sounds",
    "sleep apnea":          "apnea",

    # ── DIGESTIVE SYNONYMS ─────────────────────────────────────────────────────
    "stomach pain":         "sharp abdominal pain upper abdominal pain burning abdominal pain",
    "stomach ache":         "sharp abdominal pain upper abdominal pain",
    "tummy ache":           "sharp abdominal pain lower abdominal pain",
    "belly pain":           "sharp abdominal pain lower abdominal pain",
    "abdominal pain":       "sharp abdominal pain upper abdominal pain lower abdominal pain",
    "acidity":              "heartburn burning abdominal pain regurgitation",
    "acid reflux":          "heartburn burning abdominal pain regurgitation",
    "heartburn":            "heartburn burning chest pain burning abdominal pain",
    "bloating":             "stomach bloating abdominal distention flatulence",
    "gas":                  "flatulence stomach bloating",
    "burping":              "regurgitation flatulence",
    "nausea":               "nausea vomiting",
    "vomiting":             "vomiting nausea",
    "throwing up":          "vomiting nausea",
    "diarrhea":             "diarrhea discharge in stools",
    "loose stools":         "diarrhea discharge in stools changes in stool appearance",
    "constipation":         "constipation",
    "indigestion":          "heartburn burning abdominal pain upper abdominal pain",
    "digestion problem":    "heartburn upper abdominal pain flatulence",
    "jaundice":             "jaundice",
    "yellow skin":          "jaundice",
    "blood in stool":       "blood in stool rectal bleeding melena",
    "rectal bleeding":      "rectal bleeding blood in stool",
    "bowel issues":         "diarrhea constipation changes in stool appearance",
    "ulcer":                "burning abdominal pain sharp abdominal pain heartburn",

    # ── MENTAL / PSYCHOLOGICAL SYNONYMS ───────────────────────────────────────
    "anxiety":              "anxiety and nervousness fears and phobias restlessness",
    "anxious":              "anxiety and nervousness restlessness",
    "stress":               "anxiety and nervousness emotional symptoms restlessness",
    "stressed":             "anxiety and nervousness emotional symptoms",
    "panic":                "anxiety and nervousness palpitations breathing fast",
    "panic attack":         "anxiety and nervousness palpitations shortness of breath",
    "depression":           "depression low self-esteem emotional symptoms",
    "depressed":            "depression low self-esteem",
    "sad":                  "depression emotional symptoms low self-esteem",
    "hopeless":             "depression low self-esteem emotional symptoms",
    "mood swings":          "emotional symptoms depression excessive anger",
    "irritable":            "premenstrual tension or irritability temper problems excessive anger",
    "anger":                "excessive anger temper problems hostile behavior",
    "aggressive":           "hostile behavior excessive anger antisocial behavior",
    "can't sleep":          "insomnia sleepiness",
    "insomnia":             "insomnia",
    "sleeplessness":        "insomnia",
    "trouble sleeping":     "insomnia",
    "sleep problems":       "insomnia sleepiness sleepwalking",
    "nightmares":           "nightmares insomnia",
    "hallucinations":       "delusions or hallucinations",
    "paranoia":             "delusions or hallucinations anxiety and nervousness",
    "ocd":                  "obsessions and compulsions",
    "compulsive":           "obsessions and compulsions",
    "ptsd":                 "nightmares anxiety and nervousness fears and phobias",
    "phobia":               "fears and phobias anxiety and nervousness",
    "memory loss":          "disturbance of memory",
    "forgetful":            "disturbance of memory",
    "confusion":            "disturbance of memory delirium",
    "low confidence":       "low self-esteem emotional symptoms",

    # ── MUSCULOSKELETAL SYNONYMS ───────────────────────────────────────────────
    "joint pain":           "joint pain joint stiffness or tightness joint swelling",
    "back pain":            "back pain low back pain back stiffness or tightness",
    "lower back pain":      "low back pain low back stiffness or tightness back pain",
    "neck pain":            "neck pain neck stiffness or tightness",
    "shoulder pain":        "shoulder pain shoulder stiffness or tightness",
    "knee pain":            "knee pain knee swelling knee stiffness or tightness",
    "hip pain":             "hip pain hip stiffness or tightness",
    "muscle pain":          "muscle pain muscle stiffness or tightness muscle weakness",
    "muscle ache":          "muscle pain ache all over muscle stiffness or tightness",
    "body ache":            "ache all over muscle pain lower body pain",
    "body pain":            "ache all over muscle pain lower body pain",
    "stiffness":            "stiffness all over joint stiffness or tightness muscle stiffness or tightness",
    "swollen joint":        "joint swelling joint pain",
    "arthritis":            "joint pain joint swelling joint stiffness or tightness bones are painful",
    "cramps":               "cramps and spasms muscle cramps, contractures, or spasms leg cramps or spasms",
    "leg cramps":           "leg cramps or spasms cramps and spasms",
    "weakness":             "weakness muscle weakness focal weakness",
    "numbness":             "loss of sensation paresthesia",
    "tingling":             "paresthesia loss of sensation",
    "paralysis":            "focal weakness problems with movement",
    "spasm":                "muscle cramps, contractures, or spasms cramps and spasms",
    "wrist pain":           "wrist pain wrist stiffness or tightness",
    "elbow pain":           "elbow pain elbow stiffness or tightness",
    "ankle pain":           "ankle pain ankle swelling ankle stiffness or tightness",
    "foot pain":            "foot or toe pain foot or toe stiffness or tightness",
    "posture":              "posture problems back pain",
    "scoliosis":            "posture problems back pain",
    "sprain":               "joint pain swelling",
    "fracture":             "bones are painful",

    # ── SKIN SYNONYMS ─────────────────────────────────────────────────────────
    "rash":                 "skin rash skin irritation abnormal appearing skin",
    "skin rash":            "skin rash skin irritation abnormal appearing skin",
    "itching":              "itching of skin skin irritation",
    "itchy skin":           "itching of skin skin irritation",
    "itch":                 "itching of skin skin irritation",
    "acne":                 "acne or pimples skin oiliness",
    "pimples":              "acne or pimples skin growth",
    "dry skin":             "skin dryness, peeling, scaliness, or roughness",
    "scaly skin":           "skin dryness, peeling, scaliness, or roughness",
    "peeling skin":         "skin dryness, peeling, scaliness, or roughness",
    "blisters":             "skin lesion skin rash",
    "hives":                "skin rash allergic reaction skin swelling",
    "swelling":             "skin swelling peripheral edema",
    "eczema":               "itching of skin skin rash skin dryness, peeling, scaliness, or roughness",
    "psoriasis":            "skin rash skin lesion skin dryness, peeling, scaliness, or roughness",
    "warts":                "warts skin growth",
    "moles":                "skin moles change in skin mole size or color",
    "hair loss":            "too little hair irregular appearing scalp",
    "dandruff":             "dry or flaky scalp itchy scalp",
    "fungal":               "itching of skin skin rash",
    "infection":            "skin on arm or hand looks infected skin on leg or foot looks infected",
    "wound":                "skin pain skin swelling",
    "bruise":               "skin pain abnormal appearing skin",
    "yellowing skin":       "jaundice",
    "redness":              "skin irritation skin rash eye redness",

    # ── GENERAL / SYSTEMIC SYNONYMS ───────────────────────────────────────────
    "fever":                "fever chills feeling hot and cold",
    "high temperature":     "fever feeling hot",
    "temperature":          "fever chills",
    "chills":               "chills feeling cold",
    "fatigue":              "fatigue weakness",
    "tired":                "fatigue sleepiness",
    "tiredness":            "fatigue sleepiness weakness",
    "exhausted":            "fatigue weakness",
    "dizziness":            "dizziness fainting",
    "dizzy":                "dizziness fainting",
    "lightheaded":          "dizziness fainting",
    "headache":             "headache frontal headache",
    "head pain":            "headache frontal headache",
    "migraine":             "headache frontal headache",
    "loss of appetite":     "decreased appetite",
    "no appetite":          "decreased appetite",
    "weight loss":          "recent weight loss underweight",
    "weight gain":          "weight gain",
    "sweating":             "sweating",
    "night sweats":         "sweating feeling hot",
    "pale":                 "pallor",
    "swollen lymph nodes":  "swollen lymph nodes",
    "lymph nodes":          "swollen lymph nodes",
    "feeling sick":         "feeling ill nausea",
    "unwell":               "feeling ill",
    "malaise":              "feeling ill fatigue weakness",
    "palpitations":         "palpitations irregular heartbeat increased heart rate",
    "heart racing":         "palpitations increased heart rate",
    "chest pain":           "sharp chest pain chest tightness burning chest pain",
    "urination":            "frequent urination painful urination",
    "frequent urination":   "frequent urination polyuria",
    "thirst":               "thirst",
    "urinary":              "frequent urination painful urination retention of urine",
}


def normalize_input(user_text: str) -> str:
    """
    Expands a user's free-form symptom text by replacing known
    synonyms with their training-vocabulary equivalents.

    Args:
        user_text : Raw input from the user, e.g. "i have anxiety and stress"

    Returns:
        Expanded string containing both original tokens and training tokens,
        e.g. "anxiety and nervousness fears and phobias restlessness stress"
    """
    text_lower = user_text.lower().strip()
    expanded_tokens = []

    # First pass: check multi-word phrases (longer matches first)
    phrases_sorted = sorted(SYNONYM_MAP.keys(), key=len, reverse=True)
    replaced = set()

    for phrase in phrases_sorted:
        if phrase in text_lower and phrase not in replaced:
            expanded_tokens.append(SYNONYM_MAP[phrase])
            replaced.add(phrase)

    # Second pass: add original words that aren't stopwords
    # This preserves any training vocabulary already in the user's text
    stopwords = {
        "i", "have", "am", "a", "the", "my", "me", "is", "are", "and",
        "or", "with", "some", "feel", "feeling", "been", "get", "got",
        "very", "really", "so", "too", "bit", "little", "lot", "been",
        "since", "days", "weeks", "ago", "also", "still", "always", "can",
        "cant", "don't", "do", "not", "of", "in", "on", "at", "to", "for",
    }
    for word in text_lower.split():
        word_clean = word.strip(".,!?;:'\"()")
        if word_clean and word_clean not in stopwords:
            expanded_tokens.append(word_clean)

    result = " ".join(expanded_tokens)
    return result


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("fever cough headache runny nose",      "Respiratory"),
        ("stomach pain acidity bloating nausea", "Digestive"),
        ("joint pain back pain stiffness",        "Musculoskeletal"),
        ("anxiety stress sleeplessness",          "Mental"),
        ("rash itching dry skin",                 "Skin"),
        ("fatigue weakness tired",                "General"),
    ]

    print("Symptom Normalizer — expansion preview")
    print("=" * 60)
    for text, expected in tests:
        expanded = normalize_input(text)
        print(f"  Input    : {text}")
        print(f"  Expected : {expected}")
        print(f"  Expanded : {expanded}")
        print()