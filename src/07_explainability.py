import os
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# =========================
# SETTINGS
# =========================

FILE = "data/processed/pilot_50_translated_review.csv"

MODEL_PATH = "models/xlmr_pilot/final"

TEXT_COLUMN = "chief_complaint_raw"

MAX_LENGTH = 128


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(FILE)

df = df.dropna(
    subset=[TEXT_COLUMN]
).copy()

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

print("Total records:", len(df))


# =========================
# LOAD MODEL
# =========================

print("\nLoading XLM-R model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully!")


# =========================
# LABEL MAPPING
# =========================

print("\nModel labels:")

print(model.config.id2label)


# =========================
# EXPLAIN ONE TEXT
# =========================

def explain_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

    predicted_id = logits.argmax(
        dim=-1
    ).item()

    confidence = probabilities[
        0, predicted_id
    ].item()

    predicted_label = model.config.id2label[
        predicted_id
    ]

    # Get tokens
    token_ids = inputs["input_ids"][0]

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    # Simple token importance:
    # use input embedding magnitude as a
    # basic interpretability signal
    embeddings = model.get_input_embeddings()(
        token_ids
    )

    importance = torch.norm(
        embeddings,
        dim=1
    ).detach().cpu().numpy()

    # Remove special tokens
    special_tokens = {
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token
    }

    explanation = []

    for token, score in zip(
        tokens,
        importance
    ):

        if token in special_tokens:
            continue

        explanation.append(
            {
                "token": token,
                "importance": float(score)
            }
        )

    explanation = sorted(
        explanation,
        key=lambda x: x["importance"],
        reverse=True
    )

    return (
        predicted_label,
        confidence,
        explanation
    )


# =========================
# RUN ON FIRST 10 CASES
# =========================

all_results = []

print("\n==============================")
print("XAI RESULTS")
print("==============================")


for i, text in enumerate(
    df[TEXT_COLUMN].head(10)
):

    print("\n--------------------------------")
    print("Case:", i + 1)
    print("Text:", text)

    (
        prediction,
        confidence,
        explanation
    ) = explain_text(text)

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        round(confidence, 4)
    )

    print("\nTop important tokens:")

    top_tokens = explanation[:10]

    for item in top_tokens:

        print(
            item["token"],
            "->",
            round(
                item["importance"],
                4
            )
        )

        all_results.append(
            {
                "case": i + 1,
                "text": text,
                "prediction": prediction,
                "confidence": confidence,
                "token": item["token"],
                "importance": item["importance"]
            }
        )


# =========================
# SAVE RESULTS
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

output_file = (
    "results/"
    "xai_token_importance.csv"
)

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n==============================")
print("XAI COMPLETE")
print("==============================")

print(
    "Saved:",
    output_file
)