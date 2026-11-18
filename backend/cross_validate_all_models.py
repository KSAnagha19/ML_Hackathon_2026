
from pathlib import Path
from time import perf_counter
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FEATURES = [
    "candidate_rank",
    "distance_m",
    "road_length_m",
    "heading_difference",
    "heading_valid",
    "projection_fraction",
    "gps_step_distance_m"
]


def create_models():
    return {
        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=42
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),

        "ExtraTrees":
            ExtraTreesClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),

        "LogisticRegression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=42
                ))
            ]),

        "MLP":
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
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
                ))
            ])
    }


def top1_accuracy(df, probabilities):
    scored = df[
        ["trip_id", "gps_index", "label"]
    ].copy()

    scored["probability"] = probabilities

    best_indices = scored.groupby(
        ["trip_id", "gps_index"]
    )["probability"].idxmax()

    return float(
        scored.loc[best_indices, "label"].mean()
    )


def main():
    print("\n========== PATHSNAP ==========")
    print("STEP 15: FIVE-MODEL GROUPED CV")
    print("--------------------------------")

    train_df = pd.read_csv(
        RESULTS / "train_dataset.csv",
        dtype={"trip_id": str}
    )

    val_df = pd.read_csv(
        RESULTS / "validation_dataset.csv",
        dtype={"trip_id": str}
    )

    data = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    if data["trip_id"].nunique() != 80:
        raise ValueError(
            "Expected exactly 80 training trips."
        )

    if data.duplicated(
        subset=["trip_id", "gps_index", "arc_id"]
    ).any():
        raise ValueError(
            "Duplicate trip/GPS/arc rows found."
        )

    X = data[FEATURES]
    y = data["label"].astype(int)
    groups = data["trip_id"]

    cv = GroupKFold(n_splits=5)

    # Create fold indices once, shared by all models.
    folds = list(cv.split(X, y, groups))

    models = create_models()

    print("Total rows:", len(data))
    print("Unique trips:", groups.nunique())
    print("Models:", len(models))
    print("Folds:", len(folds))
    print("Total model fits:", len(models) * len(folds))

    output_path = RESULTS / "all_models_cv_results.csv"

    # Resume completed fits if the script is interrupted.
    if output_path.exists():
        previous = pd.read_csv(output_path)
        results = previous.to_dict("records")
        print("Previously saved fits:", len(results))
    else:
        results = []

    completed = {
        (str(row["model"]), int(row["fold"]))
        for row in results
    }

    for model_name in models:
        print(f"\n========== {model_name} ==========")

        for fold_number, (train_idx, val_idx) in enumerate(
            folds, start=1
        ):
            if (model_name, fold_number) in completed:
                print(f"Fold {fold_number}: already completed")
                continue

            print(f"\nFold {fold_number}/5")

            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])

            assert train_groups.isdisjoint(val_groups)

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = create_models()[model_name]

            start = perf_counter()
            model.fit(X_train, y_train)
            training_time = perf_counter() - start

            start = perf_counter()
            probabilities = model.predict_proba(
                X_val
            )[:, 1]
            inference_time = perf_counter() - start

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            row = {
                "model": model_name,
                "fold": fold_number,
                "training_trips": len(train_groups),
                "validation_trips": len(val_groups),
                "training_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "accuracy": float(
                    accuracy_score(y_val, predictions)
                ),
                "macro_f1": float(
                    f1_score(
                        y_val,
                        predictions,
                        average="macro",
                        zero_division=0
                    )
                ),
                "log_loss": float(
                    log_loss(
                        y_val,
                        probabilities,
                        labels=[0, 1]
                    )
                ),
                "top1_accuracy": top1_accuracy(
                    data.iloc[val_idx],
                    probabilities
                ),
                "training_time_seconds": float(
                    training_time
                ),
                "inference_time_seconds": float(
                    inference_time
                )
            }

            results.append(row)

            # Save after every completed fold.
            pd.DataFrame(results).to_csv(
                output_path,
                index=False
            )

            print(
                "Macro F1:",
                round(row["macro_f1"], 4)
            )
            print(
                "Top-1 Accuracy:",
                round(row["top1_accuracy"], 4)
            )
            print(
                "Log Loss:",
                round(row["log_loss"], 4)
            )
            print(
                "Training Time:",
                round(training_time, 2),
                "seconds"
            )

    results_df = pd.DataFrame(results)

    summary = results_df.groupby("model").agg(
        folds_completed=("fold", "nunique"),
        cv_macro_f1_mean=("macro_f1", "mean"),
        cv_macro_f1_std=("macro_f1", "std"),
        cv_top1_mean=("top1_accuracy", "mean"),
        cv_top1_std=("top1_accuracy", "std"),
        cv_log_loss_mean=("log_loss", "mean"),
        mean_training_seconds=(
            "training_time_seconds", "mean"
        ),
        mean_inference_seconds=(
            "inference_time_seconds", "mean"
        )
    ).reset_index()

    summary = summary.sort_values(
        "cv_top1_mean",
        ascending=False
    )

    summary.to_csv(
        RESULTS / "all_models_cv_summary.csv",
        index=False
    )

    print("\n========== FINAL MODEL COMPARISON ==========")
    print(summary.to_string(index=False))

    if (summary["folds_completed"] == 5).all():
        print("\nSTEP 15 COMPLETED!")
    else:
        print("\nSome folds are still missing.")


if __name__ == "__main__":
    main()
