import os
import pandas as pd


# =========================
# FILE PATHS
# =========================

MULTILINGUAL_FILE = "results/multilingual_evaluation.csv"
UNCERTAINTY_FILE = "results/uncertainty_deferral.csv"
XAI_FILE = "results/xai_token_importance.csv"


# =========================
# LOAD MULTILINGUAL RESULTS
# =========================

print("\n==============================")
print("MULTILINGUAL RESULTS")
print("==============================")

if os.path.exists(MULTILINGUAL_FILE):

    multilingual = pd.read_csv(
        MULTILINGUAL_FILE
    )

    print(
        multilingual.to_string(
            index=False
        )
    )

else:

    print(
        "Multilingual result file not found."
    )


# =========================
# LOAD UNCERTAINTY RESULTS
# =========================

print("\n==============================")
print("UNCERTAINTY / DEFERRAL")
print("==============================")

if os.path.exists(UNCERTAINTY_FILE):

    uncertainty = pd.read_csv(
        UNCERTAINTY_FILE
    )

    print(
        uncertainty.to_string(
            index=False
        )
    )

else:

    print(
        "Uncertainty result file not found."
    )


# =========================
# LOAD XAI RESULTS
# =========================

print("\n==============================")
print("XAI RESULTS")
print("==============================")

if os.path.exists(XAI_FILE):

    xai = pd.read_csv(
        XAI_FILE
    )

    print(
        "Total XAI rows:",
        len(xai)
    )

    print(
        "\nFirst 20 XAI rows:"
    )

    print(
        xai.head(20).to_string(
            index=False
        )
    )

else:

    print(
        "XAI result file not found."
    )


# =========================
# SAVE FINAL SUMMARY
# =========================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    "results/final_results_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MULTILINGUAL TRIAGE RESULTS\n"
    )

    if os.path.exists(
        MULTILINGUAL_FILE
    ):

        f.write(
            multilingual.to_string(
                index=False
            )
        )

    f.write(
        "\n\nUNCERTAINTY / HUMAN DEFERRAL\n"
    )

    if os.path.exists(
        UNCERTAINTY_FILE
    ):

        f.write(
            uncertainty.to_string(
                index=False
            )
        )

    f.write(
        "\n\nXAI RESULTS\n"
    )

    if os.path.exists(XAI_FILE):

        f.write(
            xai.head(20).to_string(
                index=False
            )
        )


print("\n==============================")
print("FINAL SUMMARY CREATED")
print("==============================")

print(
    "results/final_results_summary.txt"
)