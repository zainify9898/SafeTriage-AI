import streamlit as st
import torch
import pandas as pd
import os
from pathlib import Path

from src.safetriage_utils import detect_language, predict_triage, explain_text_perturbation

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/xlmr_full"

PILOT_DATA_PATH = (
    "data/processed/pilot_50_translated_review.csv"
)

CONFIDENCE_THRESHOLD = 0.70

MAX_LENGTH = 128


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SafeTriage-AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .research-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f4f6f8;
        border-left: 5px solid #4c78a8;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        background-color: #f7f7f7;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    weights_file = Path(MODEL_PATH) / "model.safetensors"
    if weights_file.exists() and weights_file.stat().st_size < 1024:
        try:
            header = weights_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            header = ""
        if header.startswith("version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                "Model weights are a Git LFS pointer, not the actual weights. "
                "Install Git LFS and run 'git lfs pull' in the repository, "
                "or place the complete model.safetensors file in models/xlmr_full."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# ============================================================
# PILOT DATA LOADING
# ============================================================

@st.cache_data
def load_pilot_data():

    if not os.path.exists(PILOT_DATA_PATH):

        return None

    try:

        return pd.read_csv(
            PILOT_DATA_PATH
        )

    except Exception:

        return None


# ============================================================
# TRIAGE DESCRIPTION
# ============================================================

triage_descriptions = {

    1: "Highest acuity category in the experimental dataset.",

    2: "High acuity category requiring urgent assessment.",

    3: "Moderate acuity category requiring timely assessment.",

    4: "Lower acuity category in the experimental dataset.",

    5: "Lowest acuity category in the experimental dataset."
}


# ============================================================
# APP HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏥 SafeTriage-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable and Uncertainty-Aware Multilingual '
    'Patient Triage Research Prototype'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="research-box">
    ⚠️ <b>Research Prototype Only</b><br>
    This system is designed for academic research and
    experimentation. It does not provide medical diagnosis,
    treatment advice, or replace qualified healthcare
    professionals.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔬 SafeTriage-AI")

    st.write(
        "**Model:** XLM-RoBERTa"
    )

    st.write(
        "**Task:** Triage Classification"
    )

    st.write(
        "**Confidence Threshold:** 70%"
    )

    st.divider()

    st.subheader("🌐 Languages")

    st.write("🇬🇧 English")
    st.write("🇵🇰 Urdu")
    st.write("🔤 Roman Urdu")
    st.write("🌐 Code-Mixed")

    st.divider()

    st.subheader("📊 Research Components")

    st.write("✓ Multilingual Evaluation")
    st.write("✓ Confidence Estimation")
    st.write("✓ Human Deferral")
    st.write("✓ Explainability")
    st.write("✓ Comparative Analysis")

    st.divider()

    st.caption(
        "Academic research prototype."
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

try:

    tokenizer, model = load_model()

    model_loaded = True

except Exception as error:

    model_loaded = False

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(str(error))


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Analyze Message",
        "🌐 4-Language Comparison",
        "🧪 Pilot Cases",
        "ℹ️ Research Info"
    ]
)


# ============================================================
# TAB 1 — SINGLE MESSAGE
# ============================================================

with tab1:

    st.header(
        "🔍 Analyze Patient Message"
    )

    st.write(
        "Enter a synthetic/demo patient message. "
        "The system automatically estimates the language "
        "and predicts the experimental triage category."
    )

    message = st.text_area(
        "Patient Message",
        placeholder=(
            "Example:\n"
            "Naak band hona aur runny nose, "
            "aaj se shuru hua hai."
        ),
        height=140,
        key="single_message"
    )

    analyze_button = st.button(
        "🔍 Analyze",
        use_container_width=True
    )

    if analyze_button:

        if not model_loaded:

            st.error(
                "Model available nahi hai."
            )

        elif not message.strip():

            st.warning(
                "Please enter a message."
            )

        else:

            language = detect_language(
                message
            )

            (
                level,
                confidence,
                probabilities
            ) = predict_triage(
                message,
                tokenizer,
                model
            )

            st.divider()

            # ------------------------------------------------
            # Main metrics
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "🌐 Language",
                language
            )

            c2.metric(
                "🏷️ Triage",
                f"Level {level}"
            )

            c3.metric(
                "🎯 Confidence",
                f"{confidence * 100:.1f}%"
            )

            st.subheader(
                f"Predicted Triage Level {level}"
            )

            st.write(
                triage_descriptions.get(
                    level,
                    "Experimental triage category."
                )
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.subheader(
                "🎯 Model Confidence"
            )

            st.progress(
                min(confidence, 1.0)
            )

            # ------------------------------------------------
            # Human review
            # ------------------------------------------------

            st.subheader(
                "👨‍⚕️ Human Review Decision"
            )

            if confidence < CONFIDENCE_THRESHOLD:

                st.error(
                    "⚠️ HUMAN REVIEW RECOMMENDED"
                )

                st.write(
                    f"Confidence "
                    f"({confidence * 100:.1f}%) "
                    f"is below the experimental "
                    f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold."
                )

            else:

                st.success(
                    "Confidence is above the "
                    "experimental threshold."
                )

                st.warning(
                    "Human oversight is still recommended."
                )

            # ------------------------------------------------
            # Probability distribution
            # ------------------------------------------------

            st.subheader(
                "📈 Probability Distribution"
            )

            probability_data = pd.DataFrame(
                {
                    "Probability": [
                        float(x)
                        for x in probabilities
                    ]
                },
                index=[
                    f"Level {i + 1}"
                    for i in range(
                        len(probabilities)
                    )
                ]
            )

            st.bar_chart(
                probability_data
            )

            # ------------------------------------------------
            # Input summary
            # ------------------------------------------------

            st.subheader(
                "📝 Input Summary"
            )

            st.write(
                f"**Detected Language:** {language}"
            )

            st.write(
                f"**Predicted Level:** {level}"
            )

            st.write(
                f"**Confidence:** "
                f"{confidence * 100:.2f}%"
            )

            st.write(
                f"**Human Review:** "
                f"{'Required' if confidence < CONFIDENCE_THRESHOLD else 'Not triggered'}"
            )

            st.code(message)

            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.subheader(
                "🔎 Explainability Note"
            )

            with st.spinner("Running local word-perturbation analysis..."):
                explanation = explain_text_perturbation(
                    message, tokenizer, model, top_n=10
                )

            if explanation:
                st.caption(
                    "Local sensitivity analysis: each word is removed in turn "
                    "and the change in the original predicted class probability is measured."
                )
                st.dataframe(
                    pd.DataFrame(explanation),
                    use_container_width=True,
                    hide_index=True
                )
                st.info(
                    "These are exploratory perturbation signals, not causal explanations "
                    "and not evidence that a word clinically caused a triage level."
                )
            else:
                st.info("No word-level perturbation result was available for this input.")


# ============================================================
# TAB 2 — FOUR LANGUAGE COMPARISON
# ============================================================

with tab2:

    st.header(
        "🌐 Four-Language Comparison"
    )

    st.write(
        "Enter equivalent versions of the same clinical "
        "case in four language conditions."
    )

    st.info(
        "Keep the clinical meaning approximately the same "
        "across all four versions for a controlled comparison."
    )

    col1, col2 = st.columns(2)

    with col1:

        english = st.text_area(
            "🇬🇧 English",
            placeholder=(
                "Nasal congestion and runny nose "
                "started today."
            ),
            height=110,
            key="comparison_english"
        )

        urdu = st.text_area(
            "🇵🇰 Urdu",
            placeholder=(
                "ناک بند ہونا اور ناک بہنا، "
                "آج سے شروع ہوا ہے۔"
            ),
            height=110,
            key="comparison_urdu"
        )

    with col2:

        roman_urdu = st.text_area(
            "🔤 Roman Urdu",
            placeholder=(
                "Naak band hona aur naak behna, "
                "aaj se shuru hua hai."
            ),
            height=110,
            key="comparison_roman"
        )

        code_mixed = st.text_area(
            "🌐 Code-Mixed",
            placeholder=(
                "Naak band hona aur runny nose, "
                "aaj se shuru hua hai."
            ),
            height=110,
            key="comparison_mixed"
        )

    compare_button = st.button(
        "🌐 Compare Four Languages",
        use_container_width=True
    )

    if compare_button:

        if not model_loaded:
            st.error("The model is unavailable. Resolve the model-loading error shown above before comparing languages.")
        else:
            texts = {
    
                "English": english,
    
                "Urdu": urdu,
    
                "Roman Urdu": roman_urdu,
    
                "Code-Mixed": code_mixed
            }
    
            results = []
    
            for language, text in texts.items():
    
                if not text.strip():
    
                    continue
    
                (
                    level,
                    confidence,
                    probabilities
                ) = predict_triage(
                    text,
                    tokenizer,
                    model
                )
    
                results.append(
                    {
                        "Language": language,
    
                        "Triage Level": level,
    
                        "Confidence (%)": round(
                            confidence * 100,
                            2
                        ),
    
                        "Human Review":
                            "Yes"
                            if confidence < CONFIDENCE_THRESHOLD
                            else "No"
                    }
                )
    
            if results:
    
                results_df = pd.DataFrame(
                    results
                )
    
                st.subheader(
                    "📊 Comparison Results"
                )
    
                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )
    
                # ------------------------------------------------
                # Confidence chart
                # ------------------------------------------------
    
                st.subheader(
                    "📈 Confidence by Language"
                )
    
                confidence_chart = (
                    results_df[
                        [
                            "Language",
                            "Confidence (%)"
                        ]
                    ]
                    .set_index("Language")
                )
    
                st.bar_chart(
                    confidence_chart
                )
    
                # ------------------------------------------------
                # Triage chart
                # ------------------------------------------------
    
                st.subheader(
                    "🏷️ Triage Level by Language"
                )
    
                triage_chart = (
                    results_df[
                        [
                            "Language",
                            "Triage Level"
                        ]
                    ]
                    .set_index("Language")
                )
    
                st.bar_chart(
                    triage_chart
                )
    
                # ------------------------------------------------
                # Download
                # ------------------------------------------------
    
                csv_data = results_df.to_csv(
                    index=False
                )
    
                st.download_button(
                    "📥 Download Comparison CSV",
                    data=csv_data,
                    file_name="four_language_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
                st.info(
                    "This comparison is descriptive and "
                    "research-oriented. Differences between "
                    "languages should not be interpreted as "
                    "clinical superiority."
                )
    
            else:
    
                st.warning(
                    "Enter at least one language version."
                )
    

# ============================================================
# TAB 3 — PILOT CASES
# ============================================================

with tab3:

    st.header(
        "🧪 Research Pilot Cases"
    )

    st.write(
        "This section uses the translated pilot cases "
        "already prepared for the research experiment."
    )

    if not model_loaded:
        st.warning("Pilot inference is disabled because the model could not be loaded. Resolve the model-loading error shown above.")
    else:
        pilot_df = load_pilot_data()
    
        if pilot_df is None:
    
            st.warning(
                "Pilot dataset not found at:"
            )
    
            st.code(
                PILOT_DATA_PATH
            )
    
        else:
    
            st.success(
                f"Pilot dataset loaded: "
                f"{len(pilot_df)} records"
            )
    
            # ----------------------------------------------------
            # Case selection
            # ----------------------------------------------------
    
            if "patient_id" in pilot_df.columns:
    
                selected_id = st.selectbox(
                    "Select Pilot Case",
                    pilot_df["patient_id"].tolist()
                )
    
                selected_row = pilot_df[
                    pilot_df["patient_id"]
                    == selected_id
                ].iloc[0]
    
            else:
    
                selected_index = st.selectbox(
                    "Select Pilot Case",
                    range(len(pilot_df))
                )
    
                selected_row = pilot_df.iloc[
                    selected_index
                ]
    
            # ----------------------------------------------------
            # Show source case
            # ----------------------------------------------------
    
            st.subheader(
                "📋 Selected Case"
            )
    
            if "chief_complaint_raw" in selected_row:
    
                st.write(
                    "**Original English Complaint:**"
                )
    
                st.info(
                    str(
                        selected_row[
                            "chief_complaint_raw"
                        ]
                    )
                )
    
            if "triage_acuity" in selected_row:
    
                st.write(
                    f"**Actual Dataset Triage:** "
                    f"Level {selected_row['triage_acuity']}"
                )
    
            elif "triage_level" in selected_row:
    
                st.write(
                    f"**Actual Dataset Triage:** "
                    f"Level {selected_row['triage_level']}"
                )
    
            # ----------------------------------------------------
            # Evaluate available languages
            # ----------------------------------------------------
    
            pilot_languages = {
    
                "English": "chief_complaint_raw",
    
                "Urdu": "urdu_text",
    
                "Roman Urdu": "roman_urdu_text",
    
                "Code-Mixed": "code_mixed_text"
            }
    
            pilot_results = []
    
            for language, column in pilot_languages.items():
    
                if column not in pilot_df.columns:
    
                    continue
    
                text = str(
                    selected_row[column]
                )
    
                if text == "nan" or not text.strip():
    
                    continue
    
                (
                    predicted_level,
                    confidence,
                    probabilities
                ) = predict_triage(
                    text,
                    tokenizer,
                    model
                )
    
                pilot_results.append(
                    {
                        "Language": language,
    
                        "Triage Level":
                            predicted_level,
    
                        "Confidence (%)":
                            round(
                                confidence * 100,
                                2
                            ),
    
                        "Human Review":
                            "Yes"
                            if confidence < CONFIDENCE_THRESHOLD
                            else "No"
                    }
                )
    
            if pilot_results:
    
                pilot_results_df = pd.DataFrame(
                    pilot_results
                )
    
                st.subheader(
                    "🌐 Model Performance Across Languages"
                )
    
                st.dataframe(
                    pilot_results_df,
                    use_container_width=True,
                    hide_index=True
                )
    
                # ------------------------------------------------
                # Show texts
                # ------------------------------------------------
    
                st.subheader(
                    "📝 Case Texts"
                )
    
                for language, column in pilot_languages.items():
    
                    if column in pilot_df.columns:
    
                        value = str(
                            selected_row[column]
                        )
    
                        if value != "nan" and value.strip():
    
                            with st.expander(
                                f"{language} Version"
                            ):
    
                                st.write(value)
    
                # ------------------------------------------------
                # Download
                # ------------------------------------------------
    
                pilot_csv = pilot_results_df.to_csv(
                    index=False
                )
    
                st.download_button(
                    "📥 Download Pilot Case Result",
                    data=pilot_csv,
                    file_name="pilot_case_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    

# ============================================================
# TAB 4 — RESEARCH INFORMATION
# ============================================================

with tab4:

    st.header(
        "ℹ️ About the Research Prototype"
    )

    st.subheader(
        "🎯 Research Goal"
    )

    st.write(
        "The project investigates whether multilingual "
        "patient-triage predictions change across English, "
        "Urdu, Roman Urdu, and Code-Mixed text."
    )

    st.subheader(
        "🔬 Research Components"
    )

    research_components = pd.DataFrame(
        {
            "Component": [
                "Multilingual Classification",
                "Language Detection",
                "Confidence Estimation",
                "Human Deferral",
                "Explainability",
                "Cross-Language Evaluation"
            ],

            "Purpose": [
                "Predict experimental triage category",
                "Identify language condition",
                "Estimate model certainty",
                "Flag lower-confidence cases",
                "Study model attribution signals",
                "Compare language effects"
            ]
        }
    )

    st.dataframe(
        research_components,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "⚙️ Model"
    )

    st.write(
        "**Underlying model:** XLM-RoBERTa"
    )

    st.write(
        "**Training experiment:** 800-sample "
        "experimental training subset"
    )

    st.write(
        "**Evaluation:** English, Urdu, "
        "Roman Urdu and Code-Mixed pilot conditions"
    )

    st.subheader(
        "⚠️ Important Limitations"
    )

    limitations = [
        "This is a research prototype, not a clinical system.",
        "The multilingual pilot contains manually created/translated text.",
        "The model was trained primarily using English complaint text.",
        "The multilingual evaluation is a cross-lingual transfer experiment.",
        "The pilot dataset is small.",
        "Confidence should not be interpreted as clinical certainty.",
        "Human review is recommended for low-confidence cases.",
        "Real patient identifiable information should not be entered."
    ]

    for item in limitations:

        st.write(
            f"• {item}"
        )

    st.subheader(
        "📌 Research Interpretation"
    )

    st.info(
        "The purpose of this application is to demonstrate "
        "the experimental framework and visualize research "
        "results. It should not be used to make real-world "
        "medical decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SafeTriage-AI | Multilingual Patient Triage "
    "Research Prototype | XLM-RoBERTa"
)