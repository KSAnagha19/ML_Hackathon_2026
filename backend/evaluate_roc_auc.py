
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

VALIDATION_FILE = RESULTS / "validation_dataset.csv"

MODEL_FILES = {
    "HistGradientBoosting": "hist_gradient_boosting.joblib",
    "XGBoost": "xgboost.joblib",
    "RandomForest": "random_forest.joblib",
    "ExtraTrees": "extra_trees.joblib",
    "MLP": "mlp.joblib",
    "LogisticRegression": "logistic_regression.joblib",
}


def main():

    print("\n========== PATHSNAP ==========")
    print("STEP 44: ROC-AUC EVALUATION")
    print("--------------------------------")

    validation = pd.read_csv(VALIDATION_FILE)

    if "label" not in validation.columns:
        raise ValueError(
            "Validation dataset needs a label column."
        )

    y_true = validation["label"].astype(int)

    if y_true.nunique() != 2:
        raise ValueError(
            "ROC-AUC requires both positive "
            "and negative validation labels."
        )

    results = []

    plt.figure(figsize=(9, 7))

    for name, filename in MODEL_FILES.items():

        path = MODELS / filename

        if not path.exists():
            print(f"SKIPPED {name}: {filename} not found")
            continue

        bundle = joblib.load(path)

        if isinstance(bundle, dict):
            model = bundle["model"]
            features = bundle["features"]
        else:
            print(
                f"SKIPPED {name}: "
                "model bundle has no feature list"
            )
            continue

        missing = [
            feature for feature in features
            if feature not in validation.columns
        ]

        if missing:
            print(
                f"SKIPPED {name}: missing features {missing}"
            )
            continue

        X = validation[features]

        probabilities = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(
            y_true,
            probabilities
        )

        fpr, tpr, thresholds = roc_curve(
            y_true,
            probabilities
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={auc:.4f})"
        )

        results.append({
            "model": name,
            "roc_auc": auc
        })

        print(f"{name}: ROC-AUC = {auc:.6f}")

    if not results:
        raise RuntimeError(
            "No saved model bundles were evaluated. "
            "Check model filenames."
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier (AUC=0.5000)"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("PathSnap - ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    RESULTS.mkdir(parents=True, exist_ok=True)

    plot_file = RESULTS / "roc_auc_comparison.png"

    plt.savefig(
        plot_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "roc_auc",
        ascending=False
    )

    csv_file = RESULTS / "roc_auc_comparison.csv"

    results_df.to_csv(
        csv_file,
        index=False
    )

    print("\n========== FINAL ROC-AUC RESULTS ==========")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "roc_auc": "{:.6f}".format
            }
        )
    )

    print("\nSaved:", csv_file)
    print("Saved:", plot_file)

    print("\nSTEP 44 COMPLETED!")


if __name__ == "__main__":
    main()
