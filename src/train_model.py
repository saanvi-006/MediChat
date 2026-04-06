"""
MediChat - train_model.py
--------------------------
Trains a TF-IDF + Logistic Regression pipeline to predict broad health
categories from symptom text.

KEY CHANGE from previous version:
  Training text is now passed through normalize_input() so the vectoriser
  learns from expanded, normalised tokens — exactly matching what the
  normaliser produces at inference time. This closes the training/inference
  vocabulary gap that caused most predictions to fall back to "General".

Input  : data/processed/final_dataset.csv  (columns: text, category)
Output : models/model.pkl
         models/vectorizer.pkl
"""

import os
import sys
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Make sure symptom_normalizer is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symptom_normalizer import normalize_input

# ── 1. Config ──────────────────────────────────────────────────────────────────

INPUT_PATH      = "data/processed/final_dataset.csv"
MODEL_PATH      = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# TF-IDF: unigrams + bigrams, moderate vocab size
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE  = (1, 2)

# Logistic Regression: class_weight="balanced" handles any remaining imbalance
LR_C        = 2.0    # slightly higher C = less regularisation = more expressive
LR_MAX_ITER = 1000


# ── 2. Load & normalise data ───────────────────────────────────────────────────

def load_data(path: str) -> tuple[list[str], list[str]]:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)

    required = {"text", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing columns: {missing}")

    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip().astype(bool)]

    print(f"[INFO] {len(df)} samples | {df['category'].nunique()} categories")
    print(df["category"].value_counts().to_string())

    # ── CRITICAL: normalise training text the same way inference text will be ──
    print("[INFO] Normalising training text via symptom_normalizer…")
    df["text_norm"] = df["text"].apply(normalize_input)

    return df["text_norm"].tolist(), df["category"].tolist()


# ── 3. Train ───────────────────────────────────────────────────────────────────

def train(X: list[str], y: list[str]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    print("[INFO] Fitting TF-IDF vectoriser…")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,          # log(1+tf) scaling
        min_df=2,                   # ignore tokens that appear only once
        analyzer="word",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("[INFO] Training Logistic Regression…")
    model = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
        class_weight="balanced",    # handles any residual imbalance
        solver="lbfgs",
        multi_class="multinomial",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\n[INFO] Classification Report (test set):")
    print(classification_report(y_test, y_pred))

    return vectorizer, model


# ── 4. Save artefacts ──────────────────────────────────────────────────────────

def save_artefacts(vectorizer, model) -> None:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Vectoriser saved → {VECTORIZER_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved      → {MODEL_PATH}")


# ── 5. Smoke test ──────────────────────────────────────────────────────────────

def smoke_test(vectorizer, model) -> None:
    """
    Tests with casual, free-form user inputs — exactly what the chatbot
    will receive. Each input is normalised before prediction.
    """
    samples = [
        # (user_input,                          expected_category)
        ("fever cough headache runny nose",      "Respiratory"),
        ("stomach pain acidity bloating nausea", "Digestive"),
        ("joint pain back pain stiffness",        "Musculoskeletal"),
        ("anxiety stress sleeplessness",          "Mental"),
        ("rash itching dry skin",                 "Skin"),
        ("fatigue weakness tired",                "General"),
        # Extra real-world style inputs
        ("i have a sore throat and runny nose",   "Respiratory"),
        ("my stomach hurts and i feel like vomiting", "Digestive"),
        ("i feel anxious and cant sleep at night",     "Mental"),
        ("my lower back and knees are very painful",   "Musculoskeletal"),
        ("i have a rash on my arm that keeps itching", "Skin"),
    ]

    print("\n[INFO] Smoke-test predictions:")
    print("-" * 65)
    all_correct = 0
    for text, expected in samples:
        norm_text = normalize_input(text)
        vec       = vectorizer.transform([norm_text])
        pred      = model.predict(vec)[0]
        proba     = model.predict_proba(vec).max() * 100
        correct   = "✅" if pred == expected else "❌"
        if pred == expected:
            all_correct += 1
        print(f"  {correct} Input    : {text}")
        print(f"     Expected : {expected}")
        print(f"     Got      : {pred}  ({proba:.1f}% confidence)")
        print()

    print(f"[INFO] Smoke test: {all_correct}/{len(samples)} correct\n")


# ── 6. Public predict function (used by chatbot backend) ──────────────────────

def predict(user_input: str, vectorizer, model) -> dict:
    """
    Predict the health category for a free-form user symptom input.

    Args:
        user_input : Raw text from the user
        vectorizer : Fitted TfidfVectorizer (loaded from pkl)
        model      : Fitted LogisticRegression (loaded from pkl)

    Returns:
        dict with keys:
            "category"    : predicted category string
            "confidence"  : float, 0–100
            "all_probs"   : dict of {category: probability}
    """
    norm_text = normalize_input(user_input)
    vec       = vectorizer.transform([norm_text])
    pred      = model.predict(vec)[0]
    proba_arr = model.predict_proba(vec)[0]
    classes   = model.classes_

    return {
        "category":   pred,
        "confidence": round(float(proba_arr.max()) * 100, 1),
        "all_probs":  {c: round(float(p) * 100, 1) for c, p in zip(classes, proba_arr)},
    }


# ── 7. Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_data(INPUT_PATH)
    vectorizer, model = train(X, y)
    save_artefacts(vectorizer, model)
    smoke_test(vectorizer, model)
    print("[DONE] Training complete. Models ready for MediChat backend.")