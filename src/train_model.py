import os
import sys
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symptom_normalizer import normalize_input

INPUT_PATH = "data/processed/final_dataset.csv"
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip().astype(bool)]

    print(f"[INFO] Loaded {len(df)} samples")
    print(df["category"].value_counts())
    return df


def augment_conversational_data(df):
    templates = [
        "i have {}",
        "i feel {}",
        "i am having {}",
    ]

    augmented = []
    for _, row in df.iterrows():
        for t in templates:
            augmented.append({
                "text": t.format(row["text"]),
                "category": row["category"]
            })

    df_aug = pd.DataFrame(augmented)
    print(f"[INFO] Added {len(df_aug)} augmented samples (train only)")
    return pd.concat([df, df_aug], ignore_index=True)


def normalize_data(df):
    df["text_norm"] = df["text"].apply(normalize_input)
    return df


def train(df):
    X = df["text"]
    y = df["category"]

    # ✅ SPLIT FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    train_df = pd.DataFrame({"text": X_train, "category": y_train})
    test_df  = pd.DataFrame({"text": X_test,  "category": y_test})

    # ✅ AUGMENT ONLY TRAIN
    train_df = augment_conversational_data(train_df)

    # Normalize
    train_df = normalize_data(train_df)
    test_df  = normalize_data(test_df)

    # ✅ TRIGRAMS ADDED
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,3),
        min_df=1,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(train_df["text_norm"])
    X_test_vec  = vectorizer.transform(test_df["text_norm"])

    model = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced"
    )

    print("[INFO] Training model...")
    model.fit(X_train_vec, train_df["category"])

    y_pred = model.predict(X_test_vec)

    print("\n[REALISTIC REPORT]")
    print(classification_report(test_df["category"], y_pred))

    print("\n[CONFUSION MATRIX]")
    print(confusion_matrix(test_df["category"], y_pred))

    return vectorizer, model


def save(vectorizer, model):
    os.makedirs("models", exist_ok=True)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("[INFO] Model saved")


def real_world_test(vectorizer, model):
    print("\n[REAL-WORLD TEST]")

    samples = [
        "i feel weak and tired and have body ache",
        "i have body ache and fever",
        "my stomach hurts after eating",
        "i feel anxious and can't sleep",
        "my skin is itchy and red",
        "i have cough and sore throat",
        "i keep on vomiting and my stomach hurts badly",
        "my eyes are hurting and i have a headache"
    ]

    for text in samples:
        norm = normalize_input(text)
        vec = vectorizer.transform([norm])
        pred = model.predict(vec)[0]
        conf = model.predict_proba(vec).max() * 100

        print(f"\nInput: {text}")
        print(f"Prediction: {pred} ({conf:.1f}%)")


if __name__ == "__main__":
    df = load_data(INPUT_PATH)
    vectorizer, model = train(df)
    save(vectorizer, model)
    real_world_test(vectorizer, model)
    print("\n[DONE] Training complete")