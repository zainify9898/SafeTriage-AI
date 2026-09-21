import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# =========================================================
# SETTINGS
# =========================================================

FILE = "data/processed/pilot_50_translated_review.csv"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TEXT_COLUMN = "chief_complaint_raw"
LABEL_COLUMN = "triage_acuity"

OUTPUT_DIR = "models/xlmr_pilot"


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading dataset...\n")

df = pd.read_csv(FILE)

df = df.dropna(
    subset=[
        TEXT_COLUMN,
        LABEL_COLUMN
    ]
)

print("Total records:", len(df))


# =========================================================
# LABEL ENCODING
# =========================================================

# Example:
# Original triage labels = 2, 3, 4, 5
#
# XLM-R needs numerical labels:
# 2 -> 0
# 3 -> 1
# 4 -> 2
# 5 -> 3

original_labels = sorted(
    df[LABEL_COLUMN].unique()
)

label2id = {
    label: i
    for i, label in enumerate(original_labels)
}

id2label = {
    i: str(label)
    for label, i in label2id.items()
}

df["label"] = df[LABEL_COLUMN].map(label2id)

print("Original classes:", original_labels)

print("Label mapping:")
print(label2id)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))


# =========================================================
# TOKENIZER
# =========================================================

print("\nLoading XLM-R tokenizer...\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# DATASET CLASS
# =========================================================

class TriageDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe):

        self.texts = (
            dataframe[TEXT_COLUMN]
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label"]
            .tolist()
        )

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


# Create datasets

train_dataset = TriageDataset(
    train_df
)

test_dataset = TriageDataset(
    test_df
)


# =========================================================
# LOAD XLM-R MODEL
# =========================================================

print("\nLoading XLM-R model...\n")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(original_labels),
    id2label=id2label,
    label2id={
        str(k): v
        for k, v in label2id.items()
    }
)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, true_labels = eval_pred

    predicted_labels = np.argmax(
        logits,
        axis=-1
    )

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

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# =========================================================
# TRAINING SETTINGS
# =========================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=5,

    report_to="none",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True
)


# =========================================================
# TRAINER
# =========================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# =========================================================
# TRAIN
# =========================================================

print("\n========================================")
print("STARTING XLM-R TRAINING")
print("========================================\n")

trainer.train()


# =========================================================
# PREDICTION
# =========================================================

print("\n========================================")
print("RUNNING FINAL PREDICTION")
print("========================================\n")

prediction_output = trainer.predict(
    test_dataset
)

predicted_labels = np.argmax(
    prediction_output.predictions,
    axis=-1
)

true_labels = prediction_output.label_ids


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# =========================================================
# MACRO F1
# =========================================================

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


# =========================================================
# RESULTS
# =========================================================

print("\n========================================")
print("XLM-R PILOT RESULTS")
print("========================================")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Macro-F1:",
    round(macro_f1, 4)
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

# Encoded labels:
# 0, 1, 2, 3 ...

report_labels = list(
    range(len(original_labels))
)

# Display original triage labels:
# 2, 3, 4, 5 ...

report_names = [
    str(label)
    for label in original_labels
]

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=report_labels,
        target_names=report_names,
        zero_division=0
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

print("\n========================================")
print("SAVING MODEL")
print("========================================\n")

FINAL_DIR = f"{OUTPUT_DIR}/final"

trainer.save_model(
    FINAL_DIR
)

tokenizer.save_pretrained(
    FINAL_DIR
)


# =========================================================
# DONE
# =========================================================

print("========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print(
    "Location:",
    FINAL_DIR
)

print("\nDone!")