"""
MediChat - preprocess.py
------------------------
Maps diseases → 6 meaningful categories, then balances the dataset.

Input  : data/processed/cleaned_dataset.csv  (columns: text, disease)
Output : data/processed/final_dataset.csv    (columns: text, category)

Categories (6 total):
    Respiratory | Digestive | Mental | Musculoskeletal | Skin | General
"""

import os
import pandas as pd

# ── 1. Disease → Category map ─────────────────────────────────────────────────
#
# Built from the exact disease names in the Kaggle dataset.
# Only 6 categories are used — matches the chatbot guidance goal.
# General is the intentional fallback for vague/systemic conditions.

DISEASE_CATEGORY_MAP = {

    # ── RESPIRATORY ────────────────────────────────────────────────────────────
    "asthma":                                              "Respiratory",
    "acute bronchiolitis":                                 "Respiratory",
    "acute bronchitis":                                    "Respiratory",
    "acute bronchospasm":                                  "Respiratory",
    "acute respiratory distress syndrome (ards)":          "Respiratory",
    "acute sinusitis":                                     "Respiratory",
    "atelectasis":                                         "Respiratory",
    "chronic obstructive pulmonary disease (copd)":        "Respiratory",
    "chronic sinusitis":                                   "Respiratory",
    "common cold":                                         "Respiratory",
    "croup":                                               "Respiratory",
    "cystic fibrosis":                                     "Respiratory",
    "emphysema":                                           "Respiratory",
    "empyema":                                             "Respiratory",
    "flu":                                                 "Respiratory",
    "interstitial lung disease":                           "Respiratory",
    "laryngitis":                                          "Respiratory",
    "lung cancer":                                         "Respiratory",
    "lung contusion":                                      "Respiratory",
    "pharyngitis":                                         "Respiratory",
    "pleural effusion":                                    "Respiratory",
    "pneumoconiosis":                                      "Respiratory",
    "pneumonia":                                           "Respiratory",
    "pneumothorax":                                        "Respiratory",
    "pulmonary congestion":                                "Respiratory",
    "pulmonary embolism":                                  "Respiratory",
    "pulmonary eosinophilia":                              "Respiratory",
    "pulmonary fibrosis":                                  "Respiratory",
    "pulmonary hypertension":                              "Respiratory",
    "sarcoidosis":                                         "Respiratory",
    "strep throat":                                        "Respiratory",
    "tonsillitis":                                         "Respiratory",
    "tonsillar hypertrophy":                               "Respiratory",
    "tracheitis":                                          "Respiratory",
    "tuberculosis":                                        "Respiratory",
    "whooping cough":                                      "Respiratory",
    "abscess of the lung":                                 "Respiratory",
    "abscess of the pharynx":                              "Respiratory",
    "abscess of nose":                                     "Respiratory",
    "seasonal allergies (hay fever)":                      "Respiratory",
    "nasal polyp":                                         "Respiratory",
    "deviated nasal septum":                               "Respiratory",
    "nose disorder":                                       "Respiratory",
    "vocal cord polyp":                                    "Respiratory",
    "foreign body in the throat":                          "Respiratory",
    "foreign body in the nose":                            "Respiratory",
    "peritonsillar abscess":                               "Respiratory",
    "herpangina":                                          "Respiratory",
    "obstructive sleep apnea (osa)":                       "Respiratory",

    # ── DIGESTIVE ──────────────────────────────────────────────────────────────
    "achalasia":                                           "Digestive",
    "acute pancreatitis":                                  "Digestive",
    "alcoholic liver disease":                             "Digestive",
    "appendicitis":                                        "Digestive",
    "celiac disease":                                      "Digestive",
    "cholecystitis":                                       "Digestive",
    "choledocholithiasis":                                 "Digestive",
    "chronic constipation":                                "Digestive",
    "chronic pancreatitis":                                "Digestive",
    "chronic ulcer":                                       "Digestive",
    "cirrhosis":                                           "Digestive",
    "colonic polyp":                                       "Digestive",
    "colorectal cancer":                                   "Digestive",
    "crohn disease":                                       "Digestive",
    "diverticulitis":                                      "Digestive",
    "diverticulosis":                                      "Digestive",
    "dumping syndrome":                                    "Digestive",
    "esophageal cancer":                                   "Digestive",
    "esophageal varices":                                  "Digestive",
    "esophagitis":                                         "Digestive",
    "foreign body in the gastrointestinal tract":          "Digestive",
    "gallstone":                                           "Digestive",
    "gastritis":                                           "Digestive",
    "gastroduodenal ulcer":                                "Digestive",
    "gastroesophageal reflux disease (gerd)":              "Digestive",
    "gastrointestinal hemorrhage":                         "Digestive",
    "gastroparesis":                                       "Digestive",
    "hemorrhoids":                                         "Digestive",
    "hepatic encephalopathy":                              "Digestive",
    "hepatitis due to a toxin":                            "Digestive",
    "hiatal hernia":                                       "Digestive",
    "ileus":                                               "Digestive",
    "indigestion":                                         "Digestive",
    "infectious gastroenteritis":                          "Digestive",
    "intestinal cancer":                                   "Digestive",
    "intestinal disease":                                  "Digestive",
    "intestinal malabsorption":                            "Digestive",
    "intestinal obstruction":                              "Digestive",
    "intussusception":                                     "Digestive",
    "ischemia of the bowel":                               "Digestive",
    "irritable bowel syndrome":                            "Digestive",
    "lactose intolerance":                                 "Digestive",
    "liver cancer":                                        "Digestive",
    "liver disease":                                       "Digestive",
    "meckel diverticulum":                                 "Digestive",
    "nonalcoholic liver disease (nash)":                   "Digestive",
    "noninfectious gastroenteritis":                       "Digestive",
    "pancreatic cancer":                                   "Digestive",
    "peritonitis":                                         "Digestive",
    "pyloric stenosis":                                    "Digestive",
    "rectal disorder":                                     "Digestive",
    "stomach cancer":                                      "Digestive",
    "stricture of the esophagus":                          "Digestive",
    "ulcerative colitis":                                  "Digestive",
    "viral hepatitis":                                     "Digestive",
    "volvulus":                                            "Digestive",
    "zenker diverticulum":                                 "Digestive",
    "abdominal hernia":                                    "Digestive",
    "inguinal hernia":                                     "Digestive",
    "anal fissure":                                        "Digestive",
    "anal fistula":                                        "Digestive",
    "ascending cholangitis":                               "Digestive",
    "acute fatty liver of pregnancy (aflp)":               "Digestive",
    "hirschsprung disease":                                "Digestive",
    "persistent vomiting of unknown cause":                "Digestive",
    "perirectal infection":                                "Digestive",
    "pilonidal cyst":                                      "Digestive",
    "mucositis":                                           "Digestive",

    # ── MENTAL ─────────────────────────────────────────────────────────────────
    "acute stress reaction":                               "Mental",
    "adjustment reaction":                                 "Mental",
    "alcohol abuse":                                       "Mental",
    "alcohol intoxication":                                "Mental",
    "alcohol withdrawal":                                  "Mental",
    "alzheimer disease":                                   "Mental",
    "anxiety":                                             "Mental",
    "asperger syndrome":                                   "Mental",
    "attention deficit hyperactivity disorder (adhd)":     "Mental",
    "autism":                                              "Mental",
    "bipolar disorder":                                    "Mental",
    "conduct disorder":                                    "Mental",
    "conversion disorder":                                 "Mental",
    "delirium":                                            "Mental",
    "dementia":                                            "Mental",
    "depression":                                          "Mental",
    "dissociative disorder":                               "Mental",
    "drug abuse":                                          "Mental",
    "drug abuse (barbiturates)":                           "Mental",
    "drug abuse (cocaine)":                                "Mental",
    "drug abuse (methamphetamine)":                        "Mental",
    "drug abuse (opioids)":                                "Mental",
    "drug withdrawal":                                     "Mental",
    "dysthymic disorder":                                  "Mental",
    "eating disorder":                                     "Mental",
    "factitious disorder":                                 "Mental",
    "huntington disease":                                  "Mental",
    "impulse control disorder":                            "Mental",
    "lewy body dementia":                                  "Mental",
    "marijuana abuse":                                     "Mental",
    "neurosis":                                            "Mental",
    "obsessive compulsive disorder (ocd)":                 "Mental",
    "oppositional disorder":                               "Mental",
    "panic attack":                                        "Mental",
    "panic disorder":                                      "Mental",
    "personality disorder":                                "Mental",
    "post-traumatic stress disorder (ptsd)":               "Mental",
    "postpartum depression":                               "Mental",
    "primary insomnia":                                    "Mental",
    "psychosexual disorder":                               "Mental",
    "psychotic disorder":                                  "Mental",
    "schizophrenia":                                       "Mental",
    "sleepwalking":                                        "Mental",
    "social phobia":                                       "Mental",
    "somatization disorder":                               "Mental",
    "substance-related mental disorder":                   "Mental",
    "tourette syndrome":                                   "Mental",
    "wernicke korsakoff syndrome":                         "Mental",
    "narcolepsy":                                          "Mental",
    "smoking or tobacco addiction":                        "Mental",
    "developmental disability":                            "Mental",
    "down syndrome":                                       "Mental",
    "fetal alcohol syndrome":                              "Mental",
    "edward syndrome":                                     "Mental",

    # ── MUSCULOSKELETAL ────────────────────────────────────────────────────────
    "adhesive capsulitis of the shoulder":                 "Musculoskeletal",
    "ankylosing spondylitis":                              "Musculoskeletal",
    "arthritis of the hip":                                "Musculoskeletal",
    "avascular necrosis":                                  "Musculoskeletal",
    "bone cancer":                                         "Musculoskeletal",
    "bone disorder":                                       "Musculoskeletal",
    "bone spur of the calcaneous":                         "Musculoskeletal",
    "brachial neuritis":                                   "Musculoskeletal",
    "bunion":                                              "Musculoskeletal",
    "bursitis":                                            "Musculoskeletal",
    "carpal tunnel syndrome":                              "Musculoskeletal",
    "cervical disorder":                                   "Musculoskeletal",
    "chondromalacia of the patella":                       "Musculoskeletal",
    "chronic back pain":                                   "Musculoskeletal",
    "chronic knee pain":                                   "Musculoskeletal",
    "complex regional pain syndrome":                      "Musculoskeletal",
    "de quervain disease":                                 "Musculoskeletal",
    "degenerative disc disease":                           "Musculoskeletal",
    "dislocation of the ankle":                            "Musculoskeletal",
    "dislocation of the elbow":                            "Musculoskeletal",
    "dislocation of the finger":                           "Musculoskeletal",
    "dislocation of the foot":                             "Musculoskeletal",
    "dislocation of the hip":                              "Musculoskeletal",
    "dislocation of the knee":                             "Musculoskeletal",
    "dislocation of the patella":                          "Musculoskeletal",
    "dislocation of the shoulder":                         "Musculoskeletal",
    "dislocation of the vertebra":                         "Musculoskeletal",
    "dislocation of the wrist":                            "Musculoskeletal",
    "fibromyalgia":                                        "Musculoskeletal",
    "flat feet":                                           "Musculoskeletal",
    "fracture of the ankle":                               "Musculoskeletal",
    "fracture of the arm":                                 "Musculoskeletal",
    "fracture of the facial bones":                        "Musculoskeletal",
    "fracture of the finger":                              "Musculoskeletal",
    "fracture of the foot":                                "Musculoskeletal",
    "fracture of the hand":                                "Musculoskeletal",
    "fracture of the jaw":                                 "Musculoskeletal",
    "fracture of the leg":                                 "Musculoskeletal",
    "fracture of the neck":                                "Musculoskeletal",
    "fracture of the patella":                             "Musculoskeletal",
    "fracture of the pelvis":                              "Musculoskeletal",
    "fracture of the rib":                                 "Musculoskeletal",
    "fracture of the shoulder":                            "Musculoskeletal",
    "fracture of the skull":                               "Musculoskeletal",
    "fracture of the vertebra":                            "Musculoskeletal",
    "gout":                                                "Musculoskeletal",
    "hammer toe":                                          "Musculoskeletal",
    "hemarthrosis":                                        "Musculoskeletal",
    "herniated disk":                                      "Musculoskeletal",
    "juvenile rheumatoid arthritis":                       "Musculoskeletal",
    "knee ligament or meniscus tear":                      "Musculoskeletal",
    "lateral epicondylitis (tennis elbow)":                "Musculoskeletal",
    "lumbago":                                             "Musculoskeletal",
    "muscle spasm":                                        "Musculoskeletal",
    "muscular dystrophy":                                  "Musculoskeletal",
    "myasthenia gravis":                                   "Musculoskeletal",
    "myositis":                                            "Musculoskeletal",
    "nerve impingement near the shoulder":                 "Musculoskeletal",
    "osteoarthritis":                                      "Musculoskeletal",
    "osteochondroma":                                      "Musculoskeletal",
    "osteochondrosis":                                     "Musculoskeletal",
    "osteomyelitis":                                       "Musculoskeletal",
    "osteoporosis":                                        "Musculoskeletal",
    "pain disorder affecting the neck":                    "Musculoskeletal",
    "plantar fasciitis":                                   "Musculoskeletal",
    "polymyalgia rheumatica":                              "Musculoskeletal",
    "reactive arthritis":                                  "Musculoskeletal",
    "restless leg syndrome":                               "Musculoskeletal",
    "rheumatoid arthritis":                                "Musculoskeletal",
    "rotator cuff injury":                                 "Musculoskeletal",
    "sciatica":                                            "Musculoskeletal",
    "scoliosis":                                           "Musculoskeletal",
    "septic arthritis":                                    "Musculoskeletal",
    "spinal stenosis":                                     "Musculoskeletal",
    "spondylitis":                                         "Musculoskeletal",
    "spondylolisthesis":                                   "Musculoskeletal",
    "spondylosis":                                         "Musculoskeletal",
    "sprain or strain":                                    "Musculoskeletal",
    "tendinitis":                                          "Musculoskeletal",
    "thoracic outlet syndrome":                            "Musculoskeletal",
    "tietze syndrome":                                     "Musculoskeletal",
    "torticollis":                                         "Musculoskeletal",
    "trigger finger (finger disorder)":                    "Musculoskeletal",
    "chronic pain disorder":                               "Musculoskeletal",
    "chronic rheumatic fever":                             "Musculoskeletal",
    "joint effusion":                                      "Musculoskeletal",
    "ganglion cyst":                                       "Musculoskeletal",
    "temporomandibular joint disorder":                    "Musculoskeletal",
    "jaw disorder":                                        "Musculoskeletal",
    "injury of the ankle":                                 "Musculoskeletal",
    "injury to the arm":                                   "Musculoskeletal",
    "injury to the finger":                                "Musculoskeletal",
    "injury to the hand":                                  "Musculoskeletal",
    "injury to the hip":                                   "Musculoskeletal",
    "injury to the knee":                                  "Musculoskeletal",
    "injury to the leg":                                   "Musculoskeletal",
    "injury to the shoulder":                              "Musculoskeletal",
    "injury to the spinal cord":                           "Musculoskeletal",
    "open wound due to trauma":                            "Musculoskeletal",
    "open wound from surgical incision":                   "Musculoskeletal",
    "open wound of the abdomen":                           "Musculoskeletal",
    "open wound of the arm":                               "Musculoskeletal",
    "open wound of the back":                              "Musculoskeletal",
    "open wound of the cheek":                             "Musculoskeletal",
    "open wound of the chest":                             "Musculoskeletal",
    "open wound of the ear":                               "Musculoskeletal",
    "open wound of the eye":                               "Musculoskeletal",
    "open wound of the face":                              "Musculoskeletal",
    "open wound of the finger":                            "Musculoskeletal",
    "open wound of the foot":                              "Musculoskeletal",
    "open wound of the hand":                              "Musculoskeletal",
    "open wound of the head":                              "Musculoskeletal",
    "open wound of the jaw":                               "Musculoskeletal",
    "open wound of the knee":                              "Musculoskeletal",
    "open wound of the lip":                               "Musculoskeletal",
    "open wound of the mouth":                             "Musculoskeletal",
    "open wound of the neck":                              "Musculoskeletal",
    "open wound of the nose":                              "Musculoskeletal",
    "open wound of the shoulder":                          "Musculoskeletal",
    "crushing injury":                                     "Musculoskeletal",
    "injury to internal organ":                            "Musculoskeletal",
    "injury to the abdomen":                               "Musculoskeletal",
    "injury to the face":                                  "Musculoskeletal",
    "injury to the trunk":                                 "Musculoskeletal",
    "pain after an operation":                             "Musculoskeletal",

    # ── SKIN ───────────────────────────────────────────────────────────────────
    "acanthosis nigricans":                                "Skin",
    "acne":                                                "Skin",
    "actinic keratosis":                                   "Skin",
    "alopecia":                                            "Skin",
    "atrophic skin condition":                             "Skin",
    "burn":                                                "Skin",
    "callus":                                              "Skin",
    "cellulitis or abscess of mouth":                      "Skin",
    "cold sore":                                           "Skin",
    "contact dermatitis":                                  "Skin",
    "decubitus ulcer":                                     "Skin",
    "dermatitis due to sun exposure":                      "Skin",
    "diaper rash":                                         "Skin",
    "drug reaction":                                       "Skin",
    "dyshidrosis":                                         "Skin",
    "eczema":                                              "Skin",
    "erythema multiforme":                                 "Skin",
    "frostbite":                                           "Skin",
    "fungal infection of the hair":                        "Skin",
    "fungal infection of the skin":                        "Skin",
    "hemangioma":                                          "Skin",
    "hidradenitis suppurativa":                            "Skin",
    "impetigo":                                            "Skin",
    "infection of open wound":                             "Skin",
    "intertrigo (skin condition)":                         "Skin",
    "itching of unknown cause":                            "Skin",
    "kaposi sarcoma":                                      "Skin",
    "lice":                                                "Skin",
    "lichen planus":                                       "Skin",
    "lichen simplex":                                      "Skin",
    "lipoma":                                              "Skin",
    "melanoma":                                            "Skin",
    "molluscum contagiosum":                               "Skin",
    "necrotizing fasciitis":                               "Skin",
    "onychomycosis":                                       "Skin",
    "pemphigus":                                           "Skin",
    "pityriasis rosea":                                    "Skin",
    "psoriasis":                                           "Skin",
    "pyogenic skin infection":                             "Skin",
    "rosacea":                                             "Skin",
    "scabies":                                             "Skin",
    "scar":                                                "Skin",
    "sebaceous cyst":                                      "Skin",
    "seborrheic dermatitis":                               "Skin",
    "seborrheic keratosis":                                "Skin",
    "shingles (herpes zoster)":                            "Skin",
    "skin cancer":                                         "Skin",
    "skin disorder":                                       "Skin",
    "skin pigmentation disorder":                          "Skin",
    "skin polyp":                                          "Skin",
    "soft tissue sarcoma":                                 "Skin",
    "sporotrichosis":                                      "Skin",
    "viral warts":                                         "Skin",
    "warts":                                               "Skin",
    "athlete's foot":                                      "Skin",
    "paronychia":                                          "Skin",
    "hematoma":                                            "Skin",
    "insect bite":                                         "Skin",
    "envenomation from spider or animal bite":             "Skin",
    "gas gangrene":                                        "Skin",
    "viral exanthem":                                      "Skin",
    "ingrown toe nail":                                    "Skin",
}

# Everything not in the map → General (intentional fallback)
DEFAULT_CATEGORY = "General"

# ── Target samples per category (balanced training) ───────────────────────────
# General is capped to prevent it overwhelming the model.
# Other categories are capped at 5000 to keep training fast.
CATEGORY_SAMPLE_CAP = 5000


def map_category(disease: str) -> str:
    return DISEASE_CATEGORY_MAP.get(disease.strip().lower(), DEFAULT_CATEGORY)


def preprocess(
    input_path: str  = "data/processed/cleaned_dataset.csv",
    output_path: str = "data/processed/final_dataset.csv",
) -> pd.DataFrame:

    print(f"[INFO] Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    required = {"text", "disease"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[ERROR] Missing columns: {missing}")

    print(f"[INFO] Loaded {len(df)} rows.")

    # Map diseases → categories
    df["category"] = df["disease"].apply(map_category)

    # Warn about unmapped diseases
    unmapped = df[df["category"] == DEFAULT_CATEGORY]["disease"].unique()
    if len(unmapped):
        print(f"[WARN] {len(unmapped)} disease(s) fell back to '{DEFAULT_CATEGORY}'.")

    # Keep only model inputs
    final_df = df[["text", "category"]].copy()
    final_df = final_df[final_df["text"].str.strip().astype(bool)]

    # ── Balance: cap every category at CATEGORY_SAMPLE_CAP ───────────────────
    balanced_parts = []
    for cat, group in final_df.groupby("category"):
        if len(group) > CATEGORY_SAMPLE_CAP:
            group = group.sample(n=CATEGORY_SAMPLE_CAP, random_state=42)
        balanced_parts.append(group)

    final_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"[INFO] Saved {len(final_df)} rows → {output_path}")
    print("\n[INFO] Category distribution:")
    print(final_df["category"].value_counts().to_string())

    return final_df


if __name__ == "__main__":
    preprocess()