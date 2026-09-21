import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "models/xlmr_full"
DATA_PATH = "data/processed/pilot_50_translated_review.csv"
OUTPUT_PATH = "results/xai_full_results.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", len(df))

# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
model.to("cpu")

print("Model loaded successfully.")

# =========================================================
# LABEL MAPPING
# =========================================================
id2label = model.config.id2label

print("Labels:", id2label)


def convert_label(pred_id):
    label = id2label[pred_id]
    return int(label)


# =========================================================
# GRADIENT-BASED TOKEN ATTRIBUTION
# =========================================================
def explain_text(text):

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get embedding layer
    embedding_layer = model.get_input_embeddings()

    # Convert token IDs to embeddings
    embeddings = embedding_layer(input_ids)

    # We need gradients
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    # Forward pass using embeddings
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask
    )

    logits = outputs.logits

    # Predicted class
    pred_id = torch.argmax(logits, dim=-1).item()

    # Probability
    probabilities = torch.softmax(logits, dim=-1)
    confidence = probabilities[0, pred_id].item()

    # Select predicted class score
    target_score = logits[0, pred_id]

    # Clear old gradients
    model.zero_grad()

    # Backpropagation
    target_score.backward()

    # Gradient × input attribution
    attribution = (
        embeddings.grad * embeddings
    ).sum(dim=-1).squeeze(0)

    # Absolute importance
    importance = attribution.abs().detach().numpy()

    # Tokens
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)
    )

    # Attention mask
    mask = attention_mask.squeeze(0).numpy()

    token_results = []

    for token, score, valid in zip(
        tokens,
        importance,
        mask
    ):

        if valid == 0:
            continue

        # Ignore special tokens
        if token in ["<s>", "</s>", "<pad>"]:
            continue

        token_results.append(
            (token, float(score))
        )

    # Sort by importance
    token_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return (
        convert_label(pred_id),
        confidence,
        token_results
    )


# =========================================================
# SELECT EXAMPLES
# =========================================================
examples = [
    ("English", "chief_complaint_raw"),
    ("Urdu", "urdu_text"),
    ("Roman Urdu", "roman_urdu_text"),
    ("Code-Mixed", "code_mixed_text")
]

results = []

# =========================================================
# RUN XAI
# =========================================================
for language, column in examples:

    print("\n" + "=" * 70)
    print(language)
    print("=" * 70)

    temp = df[[column, "triage_acuity"]].copy()

    temp[column] = temp[column].fillna("").astype(str)
    temp = temp[temp[column].str.strip() != ""]

    # Take first 3 examples from each language
    sample = temp.head(3)

    for index, row in sample.iterrows():

        text = row[column]
        actual_label = int(row["triage_acuity"])

        try:

            prediction, confidence, token_results = explain_text(
                text
            )

            # Top 10 tokens
            top_tokens = token_results[:10]

            token_names = [
                x[0] for x in top_tokens
            ]

            token_scores = [
                round(x[1], 6)
                for x in top_tokens
            ]

            print("\nText:")
            print(text)

            print("Actual label:", actual_label)
            print("Predicted label:", prediction)
            print("Confidence:", round(confidence, 4))

            print("\nTop important tokens:")

            for token, score in top_tokens:
                print(
                    f"{token:20s} {score:.6f}"
                )

            results.append({
                "Language": language,
                "Text": text,
                "Actual Label": actual_label,
                "Predicted Label": prediction,
                "Confidence": confidence,
                "Top Tokens": ", ".join(token_names),
                "Token Scores": ", ".join(
                    map(str, token_scores)
                )
            })

        except Exception as e:

            print("ERROR:", e)


# =========================================================
# SAVE RESULTS
# =========================================================
results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("XAI RESULTS")
print("=" * 70)

if len(results_df) > 0:
    print(
        results_df[
            [
                "Language",
                "Actual Label",
                "Predicted Label",
                "Confidence",
                "Top Tokens"
            ]
        ].to_string(index=False)
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

else:
    print("No XAI results generated.")