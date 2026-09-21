# Multilingual Patient Triage Pilot

Research project:
Towards Safe Multilingual Patient Triage: An Explainable and Uncertainty-Aware AI Framework for English, Urdu, and Roman Urdu

## Important
Triagegeist is a fully synthetic emergency-department dataset. Do not describe it as real Pakistani patient data. The Kaggle license permits academic research/education but prohibits redistribution outside Kaggle without permission.

## Dataset
Download the Triagegeist files from Kaggle after accepting its rules. Put:
- train.csv
- chief_complaints.csv
inside `data/raw/`

This project starts with a 50-case pilot. It does NOT invent experimental results.

## Pilot workflow
1. Load and join the English complaint text with triage labels.
2. Select a reproducible 50-case pilot.
3. Create a translation template for Urdu, Roman Urdu and code-mixed variants.
4. Train a simple TF-IDF + Logistic Regression baseline on English text.
5. Evaluate with accuracy and macro-F1.
6. Later add XLM-R, SHAP, calibration and selective prediction.

## Safety
This is an academic decision-support experiment, not a diagnostic or treatment system.
