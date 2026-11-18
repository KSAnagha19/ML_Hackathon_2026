
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

FEATURES = [
    "candidate_rank",
    "distance_m",
    "road_length_m",
    "heading_difference",
    "heading_valid",
    "projection_fraction",
    "gps_step_distance_m"
]


def top1_accuracy(df, probabilities):

    temp = df[
        ["trip_id", "gps_index", "label"]
    ].copy()

    temp["probability"] = probabilities

    best = temp.groupby(
        ["trip_id", "gps_index"]
    )["probability"].idxmax()

    return float(
        temp.loc[best, "label"].mean()
    )


def evaluate(model, df):

    X = df[FEATURES]
    y = df["label"].astype(int)

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "macro_f1": f1_score(
            y, pred,
            average="macro",
            zero_division=0
        ),
        "log_loss": log_loss(
            y, prob, labels=[0, 1]
        ),
        "top1_accuracy": top1_accuracy(
            df, prob
        )
    }


def main():

    print("\n========== PATHSNAP ==========")
    print("STEP 38: XGBOOST TRAINING")
    print("--------------------------------")

    train = pd.read_csv(
        RESULTS / "train_dataset.csv"
    )

    validation = pd.read_csv(
        RESULTS / "validation_dataset.csv"
    )

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Features:", len(FEATURES))

    # Check trip separation.
    train_trips = set(
        train["trip_id"].astype(str)
    )

    validation_trips = set(
        validation["trip_id"].astype(str)
    )

    overlap = train_trips & validation_trips

    if overlap:
        raise ValueError(
            f"Train/validation trip overlap: {overlap}"
        )

    X_train = train[FEATURES]
    y_train = train["label"].astype(int)

    model = XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed!")

    print("\nEvaluating validation data...")

    metrics = evaluate(
        model,
        validation
    )

    print("\n========== XGBOOST RESULTS ==========")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.6f}"
        )

    MODELS.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODELS / "xgboost.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        model_path
    )

    results_path = (
        RESULTS / "xgboost_validation_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\nSaved model:", model_path)
    print("Saved results:", results_path)

    print("\nSTEP 38 COMPLETED!")


if __name__ == "__main__":
    main()
