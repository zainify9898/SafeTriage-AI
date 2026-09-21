import pandas as pd

# ============================================
# LOAD RESULTS
# ============================================

multi = pd.read_csv(
    "results/multilingual_full_results.csv"
)

uncertainty = pd.read_csv(
    "results/uncertainty_full_results.csv"
)

xai = pd.read_csv(
    "results/xai_full_results.csv"
)

# ============================================
# PRINT FINAL RESULTS
# ============================================

print("\n")
print("=" * 80)
print("FINAL RESEARCH RESULTS")
print("=" * 80)

# --------------------------------------------
# 1. MODEL PERFORMANCE
# --------------------------------------------

print("\n1. MULTILINGUAL PERFORMANCE")
print("-" * 80)

print(
    multi[
        [
            "Language",
            "Records",
            "Accuracy",
            "Macro-F1",
            "Mean Confidence"
        ]
    ].to_string(index=False)
)

# --------------------------------------------
# 2. UNCERTAINTY
# --------------------------------------------

print("\n2. UNCERTAINTY / HUMAN DEFERRAL")
print("-" * 80)

# Show threshold 0.70 as one reference point
threshold_70 = uncertainty[
    uncertainty["Threshold"] == 0.70
]

print(
    threshold_70[
        [
            "Language",
            "AI Cases",
            "Human Review",
            "AI %",
            "Human Review %",
            "AI Accuracy"
        ]
    ].to_string(index=False)
)

# --------------------------------------------
# 3. XAI
# --------------------------------------------

print("\n3. XAI SUMMARY")
print("-" * 80)

xai_summary = (
    xai
    .groupby("Language")
    .agg(
        Cases=("Language", "count"),
        Correct=("Actual Label", lambda x: 0)
    )
    .reset_index()
)

# Calculate correct predictions
xai["Correct"] = (
    xai["Actual Label"]
    == xai["Predicted Label"]
)

xai_summary = (
    xai
    .groupby("Language")
    .agg(
        Cases=("Language", "count"),
        Correct=("Correct", "sum")
    )
    .reset_index()
)

xai_summary["Correct %"] = (
    xai_summary["Correct"]
    / xai_summary["Cases"]
    * 100
)

print(
    xai_summary.to_string(index=False)
)

# ============================================
# SAVE FINAL TABLES
# ============================================

multi.to_csv(
    "results/final_multilingual_table.csv",
    index=False
)

threshold_70.to_csv(
    "results/final_deferral_threshold_070.csv",
    index=False
)

xai_summary.to_csv(
    "results/final_xai_summary.csv",
    index=False
)

print("\n")
print("=" * 80)
print("FINAL SUMMARY FILES CREATED")
print("=" * 80)

print("results/final_multilingual_table.csv")
print("results/final_deferral_threshold_070.csv")
print("results/final_xai_summary.csv")

print("\nAll final research tables are ready.")