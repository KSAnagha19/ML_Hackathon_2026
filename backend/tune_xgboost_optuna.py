
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, f1_score


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

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
N_SPLITS = 3
RANDOM_STATE = 42


def top1_accuracy(df, probabilities):
    temp = df[
        ["trip_id", "gps_index", "label"]
    ].copy()

    temp["probability"] = probabilities

    best = temp.groupby(
        ["trip_id", "gps_index"],
        sort=False
    )["probability"].idxmax()

    return float(temp.loc[best, "label"].mean())


def make_model(params):
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def main():
    print("\n========== PATHSNAP ==========")
    print("STEP 24: XGBOOST OPTUNA TUNING")
    print("--------------------------------")

    train = pd.read_csv(
        RESULTS / "train_dataset.csv",
        dtype={"trip_id": str}
    )

    validation = pd.read_csv(
        RESULTS / "validation_dataset.csv",
        dtype={"trip_id": str}
    )

    required = FEATURES + [
        "trip_id", "gps_index", "label"
    ]

    for name, df in [
        ("training", train),
        ("validation", validation)
    ]:
        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}"
            )

    train_trips = set(train["trip_id"])
    val_trips = set(validation["trip_id"])

    if train_trips & val_trips:
        raise ValueError(
            "Training and validation trips overlap!"
        )

    X = train[FEATURES]
    y = train["label"].astype(int)
    groups = train["trip_id"]

    folds = list(
        GroupKFold(n_splits=N_SPLITS).split(
            X, y, groups
        )
    )

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Training trips:", len(train_trips))
    print("Validation trips:", len(val_trips))
    print("Features:", len(FEATURES))
    print("Optuna trials:", N_TRIALS)
    print("Grouped CV folds:", N_SPLITS)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 650, step=50
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 8
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.015, 0.15, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.65, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 1.0
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            "gamma": trial.suggest_float(
                "gamma", 0.0, 5.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-5, 5.0,
                log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.1, 20.0,
                log=True
            )
        }

        scores = []

        for fold_no, (tr_idx, va_idx) in enumerate(
            folds, start=1
        ):
            model = make_model(params)

            model.fit(
                X.iloc[tr_idx],
                y.iloc[tr_idx]
            )

            probabilities = model.predict_proba(
                X.iloc[va_idx]
            )[:, 1]

            score = top1_accuracy(
                train.iloc[va_idx],
                probabilities
            )

            scores.append(score)

            trial.report(
                float(np.mean(scores)),
                step=fold_no
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
        study_name="pathsnap_xgboost_optuna"
    )

    print("\nStarting Bayesian optimization...")
    start = time.time()

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=False
    )

    elapsed = time.time() - start

    print("\n========== BEST PARAMETERS ==========")
    print(json.dumps(
        study.best_params,
        indent=2
    ))

    print(
        "Best grouped CV Top-1:",
        f"{study.best_value * 100:.4f}%"
    )

    print("\nTraining tuned XGBoost...")

    model = make_model(study.best_params)
    model.fit(X, y)

    val_probabilities = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    val_predictions = (
        val_probabilities >= 0.5
    ).astype(int)

    val_top1 = top1_accuracy(
        validation,
        val_probabilities
    )

    val_f1 = f1_score(
        validation["label"],
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_logloss = log_loss(
        validation["label"],
        val_probabilities,
        labels=[0, 1]
    )

    print("\n========== VALIDATION RESULTS ==========")
    print(
        "Top-1 accuracy:",
        f"{val_top1 * 100:.4f}%"
    )
    print("Macro F1:", f"{val_f1:.6f}")
    print("Log loss:", f"{val_logloss:.6f}")

    model_path = (
        MODELS / "xgboost_tuned.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "best_params": study.best_params
        },
        model_path
    )

    trials_path = (
        RESULTS / "xgboost_optuna_trials.csv"
    )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False
    )

    summary = {
        "model": "XGBoost_Optuna_Tuned",
        "features": FEATURES,
        "n_trials": N_TRIALS,
        "cv_folds": N_SPLITS,
        "best_cv_top1_accuracy": study.best_value,
        "validation_top1_accuracy": val_top1,
        "validation_macro_f1": val_f1,
        "validation_log_loss": val_logloss,
        "tuning_time_seconds": elapsed,
        "best_params": study.best_params
    }

    summary_path = (
        RESULTS / "xgboost_optuna_summary.json"
    )

    with open(
        summary_path, "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n========== FILES SAVED ==========")
    print(model_path)
    print(trials_path)
    print(summary_path)

    print("\nSTEP 24 COMPLETED!")


if __name__ == "__main__":
    main()
