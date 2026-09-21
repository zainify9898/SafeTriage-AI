# SafeTriage-AI

### Towards Safe Multilingual Patient Triage: An Explainable and Uncertainty-Aware AI Framework for English, Urdu, Roman Urdu, and Code-Mixed Text

SafeTriage-AI is an academic research prototype for exploring multilingual AI-assisted patient triage across English, Urdu, Roman Urdu, and code-mixed clinical text.

The framework combines:

- Multilingual text classification
- Explainable AI (XAI)
- Confidence-based uncertainty estimation
- Human-review deferral
- A Streamlit research prototype

> **Important:** SafeTriage-AI is a research prototype and is not a diagnostic, treatment, or clinically validated system. It is designed for academic experimentation and decision-support research only.

---

## Research Objective

The project investigates whether language variation affects AI-based triage classification and whether explainability and uncertainty estimation can help identify cases that may require human review.

### Research Questions

1. How does language variation across English, Urdu, Roman Urdu, and code-mixed patient messages affect AI-based triage performance?
2. Can explainable AI identify textual factors associated with multilingual triage predictions?
3. Can uncertainty estimation identify cases that should be deferred for human review?

---

## System Overview

The SafeTriage-AI workflow is:

Patient Message  
↓  
Language Detection  
↓  
XLM-R Based Triage Classification  
↓  
Triage Level 1–5  
↓  
Confidence Estimation  
↓  
Explainability Analysis  
↓  
Human Review if Confidence is Below Threshold

The system is intended as an experimental decision-support framework rather than an autonomous clinical decision-maker.

---

## Dataset

The main benchmark used in this project is **Triagegeist**, a fully synthetic emergency-department triage dataset.

The dataset contains structured emergency-department information and free-text chief complaints with triage acuity labels.

### Important Dataset Note

Triagegeist is **synthetic data** and must not be described as real Pakistani patient data.

The multilingual evaluation in this project uses English chief complaints together with manually prepared Urdu, Roman Urdu, and code-mixed variants for research experimentation.

The project does not claim to represent naturally occurring Pakistani clinical language data.

### Dataset Setup

The original dataset is obtained separately from Kaggle.

Place the required files inside:

```text
data/raw/
├── train.csv
└── chief_complaints.csv