
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = RESULTS_DIR / "train_dataset.csv"
VAL_PATH = RESULTS_DIR / "validation_dataset.csv"

FEATURE_COLUMNS = [
    "candidate_rank",
    "distance_m",
    "road_length_m",
    "heading_difference",
    "heading_valid",
    "projection_fraction",
    "gps_step_distance_m"
]


def top1_accuracy(df, probabilities):
    scored = df[
        ["trip_id", "gps_index", "label"]
    ].copy()

    scored["probability"] = probabilities

    best_indices = (
        scored.groupby(
            ["trip_id", "gps_index"]
        )["probability"].idxmax()
    )

    selected = scored.loc[best_indices]

    return float(selected["label"].mean())


def main():
    print("\n========== PATHSNAP ==========")
    print("STEP 13: MLP NEURAL NETWORK TRAINING")
    print("--------------------------------")

    train_df = pd.read_csv(
        TRAIN_PATH,
        dtype={"trip_id": str}
    )

    val_df = pd.read_csv(
        VAL_PATH,
        dtype={"trip_id": str}
    )

    train_trips = set(train_df["trip_id"])
    val_trips = set(val_df["trip_id"])

    overlap = train_trips.intersection(val_trips)

    if overlap:
        raise ValueError(
            f"Data leakage: overlapping trips {overlap}"
        )

    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Training trips:", len(train_trips))
    print("Validation trips:", len(val_trips))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"].astype(int)

    print("\nFeatures:", FEATURE_COLUMNS)

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42
            )
        )
    ])

    print("\nTraining MLP Neural Network...")
    print("This may take several minutes.")

    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    print("Training completed!")

    start = perf_counter()
    probabilities = model.predict_proba(X_val)[:, 1]
    inference_time = perf_counter() - start

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": "MLP",
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "training_time_seconds": float(training_time),
        "inference_time_seconds": float(inference_time),
        "accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "precision": float(
            precision_score(
                y_val, predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_val, predictions,
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            )
        ),
        "log_loss": float(
            log_loss(
                y_val, probabilities,
                labels=[0, 1]
            )
        ),
        "top1_accuracy": float(
            top1_accuracy(
                val_df, probabilities
            )
        )
    }

    print("\n========== VALIDATION RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val, predictions,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_val, predictions,
        labels=[0, 1]
    )

    print("\nConfusion matrix:")
    print(cm)

    classifier = model.named_steps["classifier"]

    print("\nMLP training information:")
    print("Iterations:", classifier.n_iter_)
    print("Final training loss:", classifier.loss_)
    print("Best validation score:", classifier.best_validation_score_)

    metrics["iterations"] = int(classifier.n_iter_)
    metrics["final_training_loss"] = float(classifier.loss_)
    metrics["best_internal_validation_score"] = float(
        classifier.best_validation_score_
    )

    model_path = MODELS_DIR / "mlp.joblib"

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS
        },
        model_path,
        compress=3
    )

    metrics_path = RESULTS_DIR / "mlp_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    cm_df.to_csv(
        RESULTS_DIR / "mlp_confusion.csv"
    )

    # Save actual training loss history.
    loss_df = pd.DataFrame({
        "iteration": range(
            1, len(classifier.loss_curve_) + 1
        ),
        "training_loss": classifier.loss_curve_
    })

    loss_df.to_csv(
        RESULTS_DIR / "mlp_training_loss.csv",
        index=False
    )

    # Save actual internal validation accuracy history.
    validation_scores = getattr(
        classifier,
        "validation_scores_",
        None
    )

    if validation_scores is not None:
        pd.DataFrame({
            "iteration": range(
                1, len(validation_scores) + 1
            ),
            "validation_accuracy": validation_scores
        }).to_csv(
            RESULTS_DIR / "mlp_internal_validation.csv",
            index=False
        )

    print("\nSaved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved training loss history.")

    print("\nSTEP 13 COMPLETED!")


if __name__ == "__main__":
    main()
