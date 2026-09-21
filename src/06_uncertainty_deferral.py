import os
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# FILES
# =========================

FILE = "data/processed/pilot_50_translated_review.csv"
MODEL_PATH = "models/xlmr_pilot/final"

TEXT_COLUMN = "chief_complaint_raw"
LABEL_COLUMN = "triage_acuity"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(FILE)

df = df.dropna(
    subset=[TEXT_COLUMN, LABEL_COLUMN]
).copy()

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print("Total records:", len(df))


# =========================
# LOAD MODEL
# =========================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully!")


# =========================
# LABELS
# =========================

labels = sorted(
    df[LABEL_COLUMN].unique()
)

print("Labels:", labels)


# =========================
# PREDICTIONS
# =========================

predictions = []
confidences = []

print("\nRunning predictions...\n")

for text in df[TEXT_COLUMN]:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=-1
        )

    predicted_index = prediction.item()

    # Convert model index back to actual triage label
    predicted_label = labels[predicted_index]

    predictions.append(predicted_label)
    confidences.append(confidence.item())


# Add results

df["predicted_triage"] = predictions
df["confidence"] = confidences


# =========================
# CONFIDENCE SUMMARY
# =========================

print("\n==============================")
print("CONFIDENCE SUMMARY")
print("==============================")

print(df["confidence"].describe())


# =========================
# DEFERRAL TABLE
# =========================

thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

results = []

for threshold in thresholds:

    ai_cases = (
        df["confidence"] >= threshold
    ).sum()

    human_cases = (
        df["confidence"] < threshold
    ).sum()

    total = len(df)

    ai_percentage = (
        ai_cases / total
    ) * 100

    human_percentage = (
        human_cases / total
    ) * 100

    # Accuracy of AI on accepted cases

    accepted = df[
        df["confidence"] >= threshold
    ]

    if len(accepted) > 0:

        accuracy = (
            accepted["predicted_triage"]
            == accepted[LABEL_COLUMN]
        ).mean()

    else:

        accuracy = np.nan

    results.append({

        "Threshold": threshold,

        "AI Cases": ai_cases,

        "Human Review": human_cases,

        "AI %": round(
            ai_percentage, 2
        ),

        "Human Review %": round(
            human_percentage, 2
        ),

        "AI Accuracy": round(
            accuracy, 4
        ) if not np.isnan(accuracy)
        else None
    })


# =========================
# FINAL TABLE
# =========================

result_df = pd.DataFrame(results)

print("\n")
print("==============================================")
print("UNCERTAINTY + HUMAN DEFERRAL TABLE")
print("==============================================")

print(
    result_df.to_string(index=False)
)


# =========================
# SAVE
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

result_df.to_csv(
    "results/uncertainty_deferral.csv",
    index=False
)

df.to_csv(
    "results/uncertainty_row_predictions.csv",
    index=False
)

print("\n==============================================")
print("FILES SAVED")
print("==============================================")

print("results/uncertainty_deferral.csv")
print("results/uncertainty_row_predictions.csv")

print("\nDONE!")