import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW / "train.csv")
complaints = pd.read_csv(RAW / "chief_complaints.csv")

df = train.merge(
    complaints[["patient_id", "chief_complaint_raw"]],
    on="patient_id",
    how="inner"
)

required = ["patient_id", "chief_complaint_raw", "triage_acuity"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=["chief_complaint_raw", "triage_acuity"]).copy()

# Reproducible pilot: stratified sample where possible.
from sklearn.model_selection import train_test_split

pilot, _ = train_test_split(
    df,
    train_size=min(50, len(df)),
    stratify=df["triage_acuity"] if df["triage_acuity"].nunique() > 1 else None,
    random_state=42
)

pilot = pilot.sort_values("patient_id")
pilot["urdu_text"] = ""
pilot["roman_urdu_text"] = ""
pilot["code_mixed_text"] = ""
pilot["translation_status"] = "PENDING_HUMAN_REVIEW"

pilot.to_csv(OUT / "pilot_50_translation_template.csv", index=False)

print(f"Joined records: {len(df)}")
print(f"Pilot records: {len(pilot)}")
print("Created data/processed/pilot_50_translation_template.csv")
