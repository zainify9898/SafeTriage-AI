"""Reusable SafeTriage-AI language and prediction utilities."""
import re
import torch

MAX_LENGTH = 128
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    text_lower = text.lower()

    # --------------------------------------------------------
    # Urdu script detection
    # --------------------------------------------------------

    urdu_script = any(
        "\u0600" <= char <= "\u06ff"
        for char in text
    )

    # --------------------------------------------------------
    # Roman Urdu vocabulary
    # --------------------------------------------------------

    roman_urdu_words = {
        "hai", "hain", "ho", "hona", "hota",
        "hoti", "hoon", "hun",
        "mein", "main", "mera", "meri",
        "mere", "mujhe", "mujhy",
        "aap", "ap", "apka", "apki",
        "apke", "ka", "ki", "ke",
        "ko", "se", "sy",
        "aur", "ya", "ye", "yeh",
        "woh", "wo",
        "raha", "rahi", "rahe",
        "hua", "hui", "hue",
        "shuru", "band", "dard",
        "kya", "kyun", "kyu",
        "nahi", "nahin",
        "bohat", "bahut",
        "acha", "achha",
        "sath", "saath",
        "andar", "bahar",
        "aaj", "kal",
        "subha", "shaam", "raat",
        "din", "ghanta",
        "ghante", "ghanton",
        "thora", "zyada",
        "wala", "wali", "wale",
        "laga", "lagi",
        "kr", "kar", "karo",
        "krna", "hy", "he"
    }

    # --------------------------------------------------------
    # English vocabulary
    # --------------------------------------------------------

    english_words = {
        "the", "and", "is", "are",
        "was", "were", "has", "have",
        "had", "pain", "nose",
        "runny", "fever", "headache",
        "cough", "chest", "breathing",
        "infection", "today",
        "yesterday", "tomorrow",
        "hours", "hour", "days",
        "day", "blood", "pressure",
        "vomiting", "vomit",
        "nausea", "heart",
        "stomach", "wound",
        "injury", "swelling",
        "severe", "mild",
        "moderate", "acute",
        "symptoms", "symptom",
        "started", "starting",
        "with", "without",
        "from", "to", "in", "on"
    }

    words = set(
        re.findall(
            r"[a-zA-Z]+",
            text_lower
        )
    )

    roman_count = len(
        words.intersection(
            roman_urdu_words
        )
    )

    english_count = len(
        words.intersection(
            english_words
        )
    )

    # Urdu
    if urdu_script:

        if english_count > 0:

            return "Code-Mixed"

        return "Urdu"

    # Roman Urdu + English
    if roman_count >= 2 and english_count >= 1:

        return "Code-Mixed"

    # Roman Urdu
    if roman_count >= 2:

        return "Roman Urdu"

    # English
    if english_count >= 2:

        return "English"

    # Fallback
    try:

        detected = detect(text)

        if detected == "en":
            return "English"

        if detected == "ur":
            return "Urdu"

    except Exception:
        pass

    return "Unknown"

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_triage(
    text,
    tokenizer,
    model
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_index
    ].item()

    # --------------------------------------------------------
    # Label mapping
    # --------------------------------------------------------

    id2label = model.config.id2label

    label = id2label.get(
        predicted_index,
        str(predicted_index + 1)
    )

    if label.startswith("LABEL_"):

        try:

            triage_level = (
                int(
                    label.replace(
                        "LABEL_",
                        ""
                    )
                ) + 1
            )

        except Exception:

            triage_level = (
                predicted_index + 1
            )

    else:

        try:

            triage_level = int(label)

        except Exception:

            triage_level = (
                predicted_index + 1
            )

    return (
        triage_level,
        confidence,
        probabilities
    )


def explain_text_perturbation(text, tokenizer, model, top_n=10):
    """Return local word-deletion sensitivity scores for the predicted class.

    A positive score means removing the word lowered the original predicted
    class probability. This is a perturbation diagnostic, not a causal
    explanation. At most top_n word occurrences are evaluated for responsiveness.
    """
    text = str(text or "").strip()
    if not text:
        return []

    base_level, base_confidence, base_probabilities = predict_triage(
        text, tokenizer, model
    )
    target_index = base_level - 1
    tokens = list(re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", text, flags=re.UNICODE))
    if not tokens:
        return []

    # Use evenly spaced token positions if the input is longer than top_n.
    if top_n > 0 and len(tokens) > top_n:
        positions = sorted({
            round(i * (len(tokens) - 1) / (top_n - 1))
            for i in range(top_n)
        }) if top_n > 1 else [len(tokens) // 2]
        tokens = [tokens[i] for i in positions]

    baseline_probability = float(base_probabilities[target_index].item())
    rows = []
    for match in tokens:
        perturbed = (text[:match.start()] + " " + text[match.end():]).strip()
        if not perturbed:
            continue
        _, _, perturbed_probabilities = predict_triage(
            perturbed, tokenizer, model
        )
        changed_probability = float(perturbed_probabilities[target_index].item())
        rows.append({
            "Word": match.group(0),
            "Original class probability": round(baseline_probability, 4),
            "Probability after removal": round(changed_probability, 4),
            "Probability drop": round(baseline_probability - changed_probability, 4),
        })

    rows.sort(key=lambda row: abs(row["Probability drop"]), reverse=True)
    return rows
