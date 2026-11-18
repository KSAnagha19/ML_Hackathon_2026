
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    log_loss
)

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RESULTS / "train_dataset.csv"
VALIDATION_FILE = RESULTS / "validation_dataset.csv"

FEATURES = [
    "candidate_rank",
    "distance_m",
    "road_length_m",
    "heading_difference",
    "heading_valid",
    "projection_fraction",
    "gps_step_distance_m"
]

N_TRIALS = 20
N_SPLITS = 5
RANDOM_STATE = 42


def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Cannot find {description}. "
        f"Available columns: {list(df.columns)}"
    )


def load_datasets():
    print("Loading datasets...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    label_col = find_column(
        train,
        ["label", "is_correct", "target", "y"],
        "target label"
    )

    trip_col = find_column(
        train,
        ["trip_id", "case_id", "trip"],
        "trip identifier"
    )

    for name, df in [
        ("training", train),
        ("validation", validation)
    ]:
        required = FEATURES + [label_col, trip_col]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}"
            )

        if df[FEATURES].isna().any().any():
            raise ValueError(
                f"Missing feature values in {name}"
            )

    train_trips = set(train[trip_col].astype(str))
    validation_trips = set(
        validation[trip_col].astype(str)
    )

    if train_trips & validation_trips:
        raise ValueError(
            "DATA LEAKAGE: training and "
            "validation trips overlap!"
        )

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Training trips:", len(train_trips))
    print("Validation trips:", len(validation_trips))

    return train, validation, label_col, trip_col


def top1_accuracy(df, probabilities):
    """
    Select the highest-probability candidate
    for each GPS observation.
    """

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

    trip_col = find_column(
        df,
        ["trip_id", "case_id", "trip"],
        "trip identifier"
    )

    label_col = find_column(
        df,
        ["label", "is_correct", "target", "y"],
        "target label"
    )

    evaluation = pd.DataFrame({
        "trip_id": df[trip_col].to_numpy(),
        "gps_index": df[gps_col].to_numpy(),
        "label": df[label_col].to_numpy(),
        "probability": probabilities
    })

    selected_indices = (
        evaluation.groupby(
            ["trip_id", "gps_index"],
            sort=False
        )["probability"].idxmax()
    )

    selected = evaluation.loc[selected_indices]

    return float(
        selected["label"].mean()
    )


def make_model(params):
    return HistGradientBoostingClassifier(
        **params,
        random_state=RANDOM_STATE
    )


def main():
    print("\n========== PATHSNAP ==========")
    print("STEP 23: BAYESIAN OPTIMISATION")
    print("--------------------------------")

    train, validation, label_col, trip_col = (
        load_datasets()
    )

    X = train[FEATURES]
    y = train[label_col].astype(int)

    groups = train[trip_col].astype(str)

    if y.nunique() != 2:
        raise ValueError(
            "Expected binary candidate labels."
        )

    if groups.nunique() < N_SPLITS:
        raise ValueError(
            "Not enough trips for GroupKFold."
        )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    # Precompute trip-separated folds.
    folds = list(
        group_kfold.split(X, y, groups)
    )

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.02,
                0.2,
                log=True
            ),
            "max_iter": trial.suggest_int(
                "max_iter",
                100,
                350,
                step=50
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes",
                15,
                63
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                20,
                100
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization",
                1e-5,
                10.0,
                log=True
            )
        }

        scores = []

        for fold_no, (train_idx, val_idx) in enumerate(
            folds, start=1
        ):
            model = make_model(params)

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx]
            )

            probabilities = model.predict_proba(
                X.iloc[val_idx]
            )[:, 1]

            score = f1_score(
                y.iloc[val_idx],
                probabilities >= 0.5,
                average="macro",
                zero_division=0
            )

            scores.append(score)

            trial.report(
                float(np.mean(scores)),
                step=fold_no
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_histgb_tuning"
    )

    start_time = time.time()

    print(f"\nRunning {N_TRIALS} Bayesian trials...")
    print(f"Each trial uses {N_SPLITS} trip-grouped folds.")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    tuning_seconds = time.time() - start_time

    best_params = study.best_params

    print("\n========== BEST PARAMETERS ==========")

    print(
        json.dumps(
            best_params,
            indent=2
        )
    )

    print(
        "\nBest CV macro F1:",
        f"{study.best_value:.6f}"
    )

    print(
        "Tuning time:",
        f"{tuning_seconds:.2f} seconds"
    )

    print("\nTraining final tuned model...")

    final_model = make_model(best_params)

    final_model.fit(X, y)

    X_val = validation[FEATURES]
    y_val = validation[label_col].astype(int)

    probabilities = final_model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    validation_log_loss = log_loss(
        y_val,
        probabilities,
        labels=[0, 1]
    )

    top1 = top1_accuracy(
        validation,
        probabilities
    )

    print("\n========== HELD-OUT VALIDATION ==========")
    print("Macro F1:", f"{macro_f1:.6f}")
    print("Top-1 accuracy:", f"{top1:.6f}")
    print(
        "Log loss:",
        f"{validation_log_loss:.6f}"
    )

    model_path = (
        MODELS /
        "hist_gradient_boosting_tuned.joblib"
    )

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "best_params": best_params
        },
        model_path
    )

    trials_path = (
        RESULTS /
        "bayesian_optimization_trials.csv"
    )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    summary = {
        "model": "HistGradientBoosting_Tuned",
        "n_trials": N_TRIALS,
        "cv_folds": N_SPLITS,
        "best_cv_macro_f1": study.best_value,
        "validation_macro_f1": macro_f1,
        "validation_top1_accuracy": top1,
        "validation_log_loss": validation_log_loss,
        "tuning_time_seconds": tuning_seconds,
        "best_params": best_params
    }

    summary_path = (
        RESULTS /
        "bayesian_optimization_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n========== FILES SAVED ==========")
    print(model_path)
    print(trials_path)
    print(summary_path)

    print("\nSTEP 23 COMPLETED!")


if __name__ == "__main__":
    main()
