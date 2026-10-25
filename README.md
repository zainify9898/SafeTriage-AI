# SafeTriage-AI

**A research prototype for multilingual triage-text classification, confidence-based deferral, and exploratory model explanation.**

> **Safety notice:** This project is for academic experimentation only. It is not clinically validated and must not be used to diagnose, treat, or decide care priority for real patients. Low or high model confidence is not clinical certainty. Human clinicians must make real-world triage decisions.

## Current capabilities

- XLM-RoBERTa sequence-classification inference for experimental acuity levels 1–5.
- Language-condition heuristic for English, Urdu script, Roman Urdu, and code-mixed text.
- Single-message inference and four-language side-by-side comparison in Streamlit.
- Confidence display and configurable low-confidence human-review flag (default threshold: 0.70).
- Local word-deletion perturbation analysis that measures changes in the predicted class probability. This is a sensitivity diagnostic, not a causal explanation.
- Pilot-case evaluation from a locally supplied CSV and CSV export of comparison results.
- Utility unit tests that run with a small dummy model and do not download the trained weights.

## Repository structure

```text
SafeTriage-AI/
├── app.py
├── requirements.txt
├── src/
│   └── safetriage_utils.py
├── tests/
│   └── test_safetriage_utils.py
├── models/
│   └── xlmr_full/            # model/tokenizer files tracked with Git LFS
├── data/
│   └── processed/            # small pilot template/review CSVs
└── results/                  # experiment tables and figures
```

## Dataset and evidence

The main benchmark is **Triagegeist**, a fully synthetic emergency-department dataset. It must not be described as real Pakistani patient data. The multilingual pilot uses manually prepared/translated variants of English chief complaints; it does not establish performance on naturally occurring clinical Urdu, Roman Urdu, or code-mixed messages.

Reported repository pilot results are limited and language-dependent. In the saved 50-case pilot table, accuracy was 0.740 for English, 0.438 for Urdu, 0.396 for Roman Urdu, and 0.604 for code-mixed text. These small pilot results are exploratory and are not evidence of clinical effectiveness. A 0.70 confidence threshold is an experimental deferral rule, not a clinically validated safety threshold.

## Setup

Use Python 3.10 or 3.11 in a virtual environment where possible.

### 1. Clone the repository and fetch the full model weights

Git LFS is required because the XLM-R weight file is large.

```bash
git lfs install
git clone https://github.com/zainify9898/SafeTriage-AI.git
cd SafeTriage-AI
git lfs pull
```

If the repository is already cloned, run `git lfs pull` inside the repository directory. Confirm that `models/xlmr_full/model.safetensors` is the full model file, not a small text file containing a Git LFS pointer.

### 2. Install dependencies

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the environment's Python executable directly or follow your local Python policy; do not disable system security settings just for this project.

### 3. Run the app

From the repository root:

```bash
streamlit run app.py
```

The first run can take time and memory because the model weights are large. If model loading reports a Git LFS pointer, fetch the weights with `git lfs pull`.

## Run tests

The unit tests use a small dummy model and do not load the 1.1 GB trained checkpoint.

```bash
python -m unittest discover -s tests -v
```

## Model outputs and interpretation

The application shows the top predicted level, its softmax confidence, a probability distribution, and whether confidence falls below the configured threshold. Confidence is not calibration and must not be interpreted as the probability that a clinical decision is correct. The perturbation table reports how removing selected words changes the probability of the original predicted class; the result can be affected by unnatural perturbed text and is not a causal explanation.

## Data and licensing

Obtain the original dataset directly from its official host and follow its terms. Do not commit the original dataset, identifiable patient information, or files you are not licensed to redistribute. Check the dataset's current licence before sharing derived data or using this repository commercially.

## Known limitations

- Synthetic training labels and manually prepared language variants limit generalization.
- Small pilot sets do not support strong claims about multilingual safety or fairness.
- The heuristic language detector may misclassify short or code-mixed messages.
- The confidence threshold has not been clinically calibrated.
- Perturbation explanations are exploratory, not causal.
- The trained checkpoint is large and requires Git LFS.
- No real-world external validation or prospective clinical study has been performed.

## Citation and contact

If using this code or results, cite the project repository and the original dataset according to their terms. Add a formal citation once the project has a stable release or DOI.
