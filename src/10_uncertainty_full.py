import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# =========================
# PATHS
# =========================
MODEL_PATH = "models/xlmr_full"
DATA_PATH = "data/processed/pilot_50_translated_review.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", len(df))

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
model.to("cpu")

print("Model loaded successfully.")

# =========================
# LABEL MAPPING
# =========================
id2label = model.config.id2label

def convert_label(pred_id):
    label = id2label[pred_id]

    # Our model labels are 1,2,3,4,5
    return int(label)

# =========================
# PREDICTION FUNCTION
# =========================
def get_predictions(texts):

    predictions = []
    confidences = []

    for text in texts:

        inputs = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities).item()

        predictions.append(convert_label(pred_id))
        confidences.append(confidence)

    return predictions, confidences


# =========================
# LANGUAGE EVALUATION
# =========================
languages = [
    ("chief_complaint_raw", "English"),
    ("urdu_text", "Urdu"),
    ("roman_urdu_text", "Roman Urdu"),
    ("code_mixed_text", "Code-Mixed")
]

all_results = []

# Confidence thresholds
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

for column, language in languages:

    print("\n" + "=" * 60)
    print(language)
    print("=" * 60)

    temp = df[[column, "triage_acuity"]].copy()

    temp[column] = temp[column].fillna("").astype(str)
    temp = temp[temp[column].str.strip() != ""]

    actual = temp["triage_acuity"].astype(int).tolist()

    predictions, confidences = get_predictions(
        temp[column].tolist()
    )

    # =========================
    # THRESHOLD ANALYSIS
    # =========================
    for threshold in thresholds:

        ai_indices = [
            i for i, confidence in enumerate(confidences)
            if confidence >= threshold
        ]

        human_indices = [
            i for i, confidence in enumerate(confidences)
            if confidence < threshold
        ]

        ai_cases = len(ai_indices)
        human_cases = len(human_indices)

        # Accuracy only for cases handled by AI
        if ai_cases > 0:

            ai_actual = [
                actual[i]
                for i in ai_indices
            ]

            ai_predictions = [
                predictions[i]
                for i in ai_indices
            ]

            ai_accuracy = accuracy_score(
                ai_actual,
                ai_predictions
            )

        else:
            ai_accuracy = float("nan")

        ai_percentage = (
            ai_cases / len(actual) * 100
        )

        human_percentage = (
            human_cases / len(actual) * 100
        )

        result = {
            "Language": language,
            "Threshold": threshold,
            "Total Cases": len(actual),
            "AI Cases": ai_cases,
            "Human Review": human_cases,
            "AI %": ai_percentage,
            "Human Review %": human_percentage,
            "AI Accuracy": ai_accuracy
        }

        all_results.append(result)

        print(
            f"Threshold {threshold:.2f} | "
            f"AI: {ai_cases} ({ai_percentage:.1f}%) | "
            f"Human: {human_cases} ({human_percentage:.1f}%) | "
            f"AI Accuracy: "
            f"{ai_accuracy if ai_cases > 0 else 'N/A'}"
        )


# =========================
# FINAL TABLE
# =========================
results_df = pd.DataFrame(all_results)

print("\n")
print("=" * 90)
print("UNCERTAINTY + HUMAN DEFERRAL RESULTS")
print("=" * 90)

print(results_df.to_string(index=False))

# =========================
# SAVE
# =========================
results_df.to_csv(
    "results/uncertainty_full_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/uncertainty_full_results.csv")