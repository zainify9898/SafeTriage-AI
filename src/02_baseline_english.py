import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

DATA = Path("data/processed/pilot_50_translation_template.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["chief_complaint_raw", "triage_acuity"])

X_train, X_test, y_train, y_test = train_test_split(
    df["chief_complaint_raw"].astype(str),
    df["triage_acuity"],
    test_size=0.25,
    stratify=df["triage_acuity"],
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Macro-F1:", round(f1_score(y_test, pred, average="macro"), 4))
print("\nClassification report:\n")
print(classification_report(y_test, pred, zero_division=0))

joblib.dump(model, MODEL_DIR / "english_tfidf_logreg.joblib")
print("\nSaved model to models/english_tfidf_logreg.joblib")
