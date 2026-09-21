import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# FILE PATHS
# =========================================================
MULTI_PATH = "results/multilingual_full_results.csv"
UNCERTAINTY_PATH = "results/uncertainty_full_results.csv"

# =========================================================
# LOAD RESULTS
# =========================================================
multi = pd.read_csv(MULTI_PATH)
uncertainty = pd.read_csv(UNCERTAINTY_PATH)

print("Results loaded successfully.")

# =========================================================
# GRAPH 1: MULTILINGUAL PERFORMANCE
# =========================================================

languages = multi["Language"]

x = range(len(languages))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width/2 for i in x],
    multi["Accuracy"],
    width=width,
    label="Accuracy"
)

plt.bar(
    [i + width/2 for i in x],
    multi["Macro-F1"],
    width=width,
    label="Macro-F1"
)

plt.xticks(list(x), languages)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Language")
plt.title("Multilingual Triage Performance")
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/figure1_multilingual_performance.png",
    dpi=300
)

plt.show()

print("Figure 1 saved.")

# =========================================================
# GRAPH 2: AI VS HUMAN REVIEW
# =========================================================

for language in uncertainty["Language"].unique():

    temp = uncertainty[
        uncertainty["Language"] == language
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        temp["Threshold"],
        temp["AI %"],
        marker="o",
        label="AI Cases (%)"
    )

    plt.plot(
        temp["Threshold"],
        temp["Human Review %"],
        marker="o",
        label="Human Review (%)"
    )

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Percentage of Cases")
    plt.ylim(0, 105)

    plt.title(
        f"AI vs Human Review: {language}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = (
        "results/deferral_"
        + language.lower().replace(" ", "_").replace("-", "_")
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()

    print("Saved:", filename)

print("\nAll graphs generated successfully.")