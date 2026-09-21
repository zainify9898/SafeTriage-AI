import pandas as pd
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =====================================================
# SETTINGS
# =====================================================

FILE = "data/processed/pilot_50_translated_review.csv"

MODEL_DIR = "models/xlmr_pilot/final"

LABEL_COLUMN = "triage_acuity"

TEXT_COLUMNS = {
    "English": "chief_complaint_raw",
    "Urdu": "urdu_text",
    "Roman Urdu": "roman_urdu_text",
    "Code-Mixed": "code_mixed_text"
}


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading dataset...")

df = pd.read_csv(FILE)

print("Dataset loaded.")
print("Total rows:", len(df))

print("\nAvailable columns:")
print(df.columns.tolist())


# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================

print("\nChecking language columns...\n")

for language, column in TEXT_COLUMNS.items():

    if column in df.columns:
        print("OK:", language, "->", column)
    else:
        print("MISSING:", language, "->", column)


# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading trained XLM-R model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("Model loaded successfully.")


# =====================================================
# LABEL MAPPING
# =====================================================

original_labels = sorted(
    df[LABEL_COLUMN].dropna().unique()
)

label2id = {
    label: i
    for i, label in enumerate(original_labels)
}

print("\nOriginal labels:", original_labels)


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_text(text):

    inputs = tokenizer(
        str(text),
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    return predicted_id, confidence


# =====================================================
# EVALUATE LANGUAGE
# =====================================================

results = []


for language, column in TEXT_COLUMNS.items():

    print("\n")
    print("========================================")
    print("TESTING:", language)
    print("========================================")

    # Check column
    if column not in df.columns:

        print(
            "SKIPPED:",
            column,
            "does not exist."
        )

        continue

    temp = df.dropna(
        subset=[
            column,
            LABEL_COLUMN
        ]
    ).copy()

    temp = temp[
        temp[column]
        .astype(str)
        .str.strip()
        != ""
    ]

    print(
        "Valid records:",
        len(temp)
    )

    if len(temp) == 0:

        print("No text available.")
        continue

    true_labels = []
    predicted_labels = []
    confidences = []

    # ---------------------------------------------
    # Predictions
    # ---------------------------------------------

    for _, row in temp.iterrows():

        true_label = row[
            LABEL_COLUMN
        ]

        true_id = label2id[
            true_label
        ]

        predicted_id, confidence = predict_text(
            row[column]
        )

        true_labels.append(
            true_id
        )

        predicted_labels.append(
            predicted_id
        )

        confidences.append(
            confidence
        )

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    mean_confidence = np.mean(
        confidences
    )

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Macro-F1:",
        round(macro_f1, 4)
    )

    print(
        "Mean Confidence:",
        round(mean_confidence, 4)
    )

    # ---------------------------------------------
    # Store result
    # ---------------------------------------------

    results.append({
        "Language": language,
        "Records": len(temp),
        "Accuracy": round(accuracy, 4),
        "Macro-F1": round(macro_f1, 4),
        "Mean Confidence": round(
            mean_confidence,
            4
        )
    })


# =====================================================
# FINAL TABLE
# =====================================================

print("\n\n")
print("========================================")
print("FINAL MULTILINGUAL RESULTS")
print("========================================")

if len(results) == 0:

    print(
        "\nNO RESULTS WERE GENERATED."
    )

else:

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # Save CSV
    results_df.to_csv(
        "results/multilingual_evaluation.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        "results/multilingual_evaluation.csv"
    )

print("\nFinished.")