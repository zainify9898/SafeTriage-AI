import pandas as pd
from pathlib import Path

src = Path("data/processed/pilot_50_translation_template.csv")
out = Path("data/processed/pilot_50_annotation_sheet.csv")

df = pd.read_csv(src)

cols = [
    "patient_id",
    "chief_complaint_raw",
    "triage_acuity",
    "urdu_text",
    "roman_urdu_text",
    "code_mixed_text",
    "translation_status"
]

df[cols].to_csv(out, index=False)
print(f"Created: {out}")
print("Fill the three language columns carefully. Preserve the clinical meaning.")
