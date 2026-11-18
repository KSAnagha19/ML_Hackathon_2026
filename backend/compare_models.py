
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
MODELS = ROOT / "models"

VALIDATION_FILE = RESULTS / "validation_dataset.csv"

MODEL_FILES = {
    "Original_HistGradientBoosting":
        MODELS / "hist_gradient_boosting.joblib",

    "Tuned_HistGradientBoosting":
        MODELS / "hist_gradient_boosting_tuned.joblib"
}


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def calculate_top1(df, probabilities, label_col):
    trip_col = find_column(
        df,
        ["trip_id", "case_id", "trip"],
        "trip identifier"
    )

    gps_col = find_column(
        df,
        [
            "gps_index",
            "gps_idx",
            "point_index",
            "observation_index"
        ],
        "GPS observation identifier"
    )

    evaluation = pd.DataFrame({
        "trip_id": df[trip_col].to_numpy(),
        "gps_index": df[gps_col].to_numpy(),
        "label": df[label_col].to_numpy(),
        "probability": probabilities
    })

    selected = (
        evaluation.groupby(
            ["trip_id", "gps_index"],
            sort=False
        )["probability"].idxmax()
    )

    return float(
        evaluation.loc[selected, "label"].mean()
    )


def evaluate_model(name, path, validation, label_col):
    print(f"\n========== {name} ==========")

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    bundle = joblib.load(path)

    if isinstance(bundle, dict):
        model = bundle["model"]
        features = bundle["features"]
    else:
        raise ValueError(
            "Expected a model bundle with "
            "'model' and 'features' keys."
        )

    missing = [
        feature for feature in features
        if feature not in validation.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = validation[features]
    y = validation[label_col].astype(int)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            y, predictions
        ),
        "precision": precision_score(
            y, predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y, predictions,
            zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions,
            average="macro",
            zero_division=0
        ),
        "log_loss": log_loss(
            y, probabilities,
            labels=[0, 1]
        ),
        "top1_accuracy": calculate_top1(
            validation,
            probabilities,
            label_col
        )
    }

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.6f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return metrics


def main():
    print("\n========== PATHSNAP ==========")
    print("STEP 24: ORIGINAL VS TUNED MODEL")
    print("--------------------------------")

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    label_col = find_column(
        validation,
        ["label", "is_correct", "target", "y"],
        "target label"
    )

    trip_col = find_column(
        validation,
        ["trip_id", "case_id", "trip"],
        "trip identifier"
    )

    print("Validation rows:", len(validation))
    print(
        "Validation trips:",
        validation[trip_col].nunique()
    )

    results = []

    for name, path in MODEL_FILES.items():
        metrics = evaluate_model(
            name,
            path,
            validation,
            label_col
        )

        results.append(metrics)

    comparison = pd.DataFrame(results)

    print("\n========== FINAL COMPARISON ==========")
    print(
        comparison.to_string(index=False)
    )

    original = comparison.iloc[0]
    tuned = comparison.iloc[1]

    print("\n========== DIFFERENCE (TUNED - ORIGINAL) ==========")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "log_loss",
        "top1_accuracy"
    ]:
        difference = (
            tuned[metric] - original[metric]
        )

        print(
            f"{metric}: {difference:+.6f}"
        )

    print("\nNote: Higher is better for most metrics.")
    print("For log loss, LOWER is better.")

    RESULTS.mkdir(
        parents=True, exist_ok=True
    )

    output = (
        RESULTS /
        "original_vs_tuned_comparison.csv"
    )

    comparison.to_csv(
        output,
        index=False
    )

    print("\nSaved:", output)
    print("\nSTEP 24 COMPLETED!")


if __name__ == "__main__":
    main()
