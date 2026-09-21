import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

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
print("Columns:")
print(df.columns.tolist())

# =========================
# LOAD MODEL
# =========================
print("\nLoading XLM-R model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

device = torch.device("cpu")
model.to(device)

print("Model loaded successfully.")

# =========================
# LABEL MAPPING
# =========================
id2label = model.config.id2label

print("\nModel labels:")
print(id2label)

# Convert model output index -> original triage label
def get_original_label(pred_id):
    label = id2label[str(pred_id)] if str(pred_id) in id2label else id2label[pred_id]

    # LABEL_0 -> 1
    if isinstance(label, str) and label.startswith("LABEL_"):
        return int(label.split("_")[1]) + 1

    return int(label)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_language(text_column, language_name):

    if text_column not in df.columns:
        print(f"\n{language_name}: column not found")
        return None

    temp = df[[text_column, "triage_acuity"]].copy()

    temp[text_column] = temp[text_column].fillna("").astype(str)
    temp = temp[temp[text_column].str.strip() != ""]

    if len(temp) == 0:
        print(f"\n{language_name}: no usable records")
        return None

    predictions = []
    confidences = []

    print(f"\nEvaluating {language_name}...")
    print("Records:", len(temp))

    for text in temp[text_column]:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)

        pred_id = torch.argmax(probabilities, dim=-1).item()
        confidence = torch.max(probabilities).item()

        predictions.append(get_original_label(pred_id))
        confidences.append(confidence)

    actual = temp["triage_acuity"].astype(int).tolist()

    accuracy = accuracy_score(actual, predictions)

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    mean_confidence = sum(confidences) / len(confidences)

    print(
        f"{language_name}: "
        f"Accuracy={accuracy:.4f}, "
        f"Macro-F1={macro_f1:.4f}, "
        f"Mean Confidence={mean_confidence:.4f}"
    )

    return {
        "Language": language_name,
        "Records": len(temp),
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Mean Confidence": mean_confidence
    }


# =========================
# RUN 4 LANGUAGES
# =========================

results = []

languages = [
    ("chief_complaint_raw", "English"),
    ("urdu_text", "Urdu"),
    ("roman_urdu_text", "Roman Urdu"),
    ("code_mixed_text", "Code-Mixed")
]

for column, language in languages:

    result = evaluate_language(column, language)

    if result is not None:
        results.append(result)

# =========================
# RESULTS TABLE
# =========================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL MULTILINGUAL EVALUATION")
print("=" * 70)

print(results_df.to_string(index=False))

# =========================
# SAVE RESULTS
# =========================

results_df.to_csv(
    "results/multilingual_full_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/multilingual_full_results.csv")