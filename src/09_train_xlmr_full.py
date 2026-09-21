import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ==================================================
# SETTINGS
# ==================================================

TRAIN_FILE = "data/raw/train.csv"
COMPLAINT_FILE = "data/raw/chief_complaints.csv"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

OUTPUT_DIR = "models/xlmr_full"

TEXT_COLUMN = "chief_complaint_raw"
LABEL_COLUMN = "triage_acuity"

MAX_LENGTH = 128

# Fast version
MAX_SAMPLES = 8000


# ==================================================
# START
# ==================================================

print("\n==============================")
print("FAST XLM-R TRAINING")
print("==============================")


# ==================================================
# LOAD DATA
# ==================================================

train_df = pd.read_csv(TRAIN_FILE)
complaints_df = pd.read_csv(COMPLAINT_FILE)

print("Train rows:", len(train_df))
print("Complaint rows:", len(complaints_df))


# ==================================================
# MERGE
# ==================================================

df = train_df.merge(
    complaints_df,
    on="patient_id",
    how="inner"
)

print("Merged rows:", len(df))


# ==================================================
# CLEAN
# ==================================================

df = df.dropna(
    subset=[
        TEXT_COLUMN,
        LABEL_COLUMN
    ]
).copy()

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

df = df[
    df[TEXT_COLUMN] != ""
].copy()

print("After cleaning:", len(df))


# ==================================================
# LABELS
# ==================================================

original_labels = sorted(
    df[LABEL_COLUMN].unique().tolist()
)

print("\nOriginal labels:")
print(original_labels)


label_to_id = {
    label: i
    for i, label in enumerate(original_labels)
}

id_to_label = {
    i: str(label)
    for i, label in enumerate(original_labels)
}

df["label"] = df[LABEL_COLUMN].map(
    label_to_id
)


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

print("\nClass distribution:")

print(
    df[LABEL_COLUMN]
    .value_counts()
    .sort_index()
)


# ==================================================
# TAKE 8000 STRATIFIED SAMPLES
# ==================================================

if len(df) > MAX_SAMPLES:

    df, _ = train_test_split(
        df,
        train_size=MAX_SAMPLES,
        random_state=42,
        stratify=df["label"]
    )

    df = df.reset_index(drop=True)


print("\nSamples selected:", len(df))


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training:", len(train_data))
print("Testing :", len(test_data))


# ==================================================
# HUGGING FACE DATASETS
# ==================================================

train_dataset = Dataset.from_pandas(
    train_data[
        [TEXT_COLUMN, "label"]
    ],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_data[
        [TEXT_COLUMN, "label"]
    ],
    preserve_index=False
)


# ==================================================
# TOKENIZER
# ==================================================

print("\n==============================")
print("LOADING XLM-R TOKENIZER")
print("==============================")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ==================================================
# MODEL
# ==================================================

print("\n==============================")
print("LOADING XLM-R MODEL")
print("==============================")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(original_labels),
    id2label=id_to_label,
    label2id={
        str(label): i
        for i, label in enumerate(original_labels)
    }
)

print("Model loaded!")


# ==================================================
# METRICS
# ==================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# ==================================================
# TRAINING SETTINGS
# ==================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=1,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_steps=100,

    report_to="none"
)


# ==================================================
# TRAINER
# ==================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ==================================================
# TRAIN
# ==================================================

print("\n==============================")
print("TRAINING STARTED")
print("==============================")

trainer.train()


# ==================================================
# EVALUATION
# ==================================================

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

results = trainer.evaluate()

print("\nAccuracy:",
      results.get("eval_accuracy"))

print("Macro-F1:",
      results.get("eval_macro_f1"))


# ==================================================
# SAVE MODEL
# ==================================================

print("\n==============================")
print("SAVING MODEL")
print("==============================")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ==================================================
# SAVE RESULTS
# ==================================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df = pd.DataFrame([
    {
        "model": "XLM-R",
        "samples": len(df),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "epochs": 1,
        "accuracy": results.get(
            "eval_accuracy"
        ),
        "macro_f1": results.get(
            "eval_macro_f1"
        )
    }
])

results_df.to_csv(
    "results/xlmr_full_results.csv",
    index=False
)


# ==================================================
# DONE
# ==================================================

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(
    "Model saved at:",
    OUTPUT_DIR
)

print(
    "Results saved at:",
    "results/xlmr_full_results.csv"
)

print("\nDONE!")