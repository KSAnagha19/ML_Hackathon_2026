
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score


# ==========================================
# STEP 32: FEATURE IMPORTANCE ANALYSIS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
MODELS = ROOT / "models"

RESULTS.mkdir(parents=True, exist_ok=True)

VALIDATION_FILE = RESULTS / "validation_dataset.csv"

MODEL_FILE = (
    MODELS / "hist_gradient_boosting.joblib"
)

SAMPLE_SIZE = 30000
N_REPEATS = 5
RANDOM_SEED = 42


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate(model, X, y):

    predictions = model.predict(X)

    score = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    return float(score)


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("\n========== PATHSNAP ==========")
    print("STEP 32: FEATURE IMPORTANCE")
    print("--------------------------------")

    # --------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------

    print("\nLoading validation dataset...")

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {VALIDATION_FILE}"
        )

    df = pd.read_csv(VALIDATION_FILE)

    print("Validation rows:", len(df))
    print("Validation columns:", len(df.columns))

    # --------------------------------------
    # LOAD TRAINED MODEL
    # --------------------------------------

    print("\nLoading trained ML model...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    features = list(bundle["features"])

    print("Model: HistGradientBoosting")
    print("Number of features:", len(features))

    print("\nFeatures used:")

    for index, feature in enumerate(
        features,
        start=1
    ):
        print(f"{index}. {feature}")

    # --------------------------------------
    # CHECK DATA
    # --------------------------------------

    required_columns = features + ["label"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------
    # SAMPLE VALIDATION DATA
    # --------------------------------------

    if len(df) > SAMPLE_SIZE:

        print(
            f"\nSampling {SAMPLE_SIZE} rows..."
        )

        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED
        ).copy()

    X = df[features].copy()
    y = df["label"].astype(int)

    print("Rows used:", len(X))
    print(
        "Missing feature values:",
        int(X.isnull().sum().sum())
    )

    # --------------------------------------
    # BASELINE MODEL PERFORMANCE
    # --------------------------------------

    print("\nEvaluating original model...")

    baseline_score = evaluate(
        model,
        X,
        y
    )

    print(
        "Baseline Macro F1:",
        f"{baseline_score:.6f}"
    )

    # --------------------------------------
    # PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------

    print("\nCalculating feature importance...")
    print("--------------------------------")

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    importance_results = []

    for feature in features:

        drops = []

        for repeat in range(N_REPEATS):

            X_permuted = X.copy()

            shuffled_values = rng.permutation(
                X_permuted[feature].to_numpy()
            )

            X_permuted[feature] = (
                shuffled_values
            )

            shuffled_score = evaluate(
                model,
                X_permuted,
                y
            )

            importance = (
                baseline_score - shuffled_score
            )

            drops.append(importance)

        mean_importance = float(
            np.mean(drops)
        )

        std_importance = float(
            np.std(drops)
        )

        importance_results.append({
            "feature": feature,
            "importance_mean": mean_importance,
            "importance_std": std_importance
        })

        print(
            f"{feature:28s} "
            f"{mean_importance:+.6f}"
        )

    # --------------------------------------
    # SORT FEATURES
    # --------------------------------------

    importance_df = pd.DataFrame(
        importance_results
    )

    importance_df = (
        importance_df.sort_values(
            by="importance_mean",
            ascending=False
        ).reset_index(drop=True)
    )

    importance_df.insert(
        0,
        "rank",
        range(1, len(importance_df) + 1)
    )

    # --------------------------------------
    # SAVE FEATURE RANKING
    # --------------------------------------

    csv_file = (
        RESULTS / "feature_importance.csv"
    )

    importance_df.to_csv(
        csv_file,
        index=False
    )

    # --------------------------------------
    # CREATE FEATURE IMPORTANCE GRAPH
    # --------------------------------------

    print("\nGenerating feature importance graph...")

    plot_df = importance_df.sort_values(
        by="importance_mean",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        capsize=4
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.xlabel(
        "Decrease in Macro F1 after shuffling"
    )

    plt.ylabel("Features")

    plt.title(
        "PathSnap - Feature Importance Analysis"
    )

    plt.tight_layout()

    graph_file = (
        RESULTS / "feature_importance.png"
    )

    plt.savefig(
        graph_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------

    print("\n========== FEATURE RANKING ==========")

    print(
        importance_df.to_string(
            index=False
        )
    )

    print("\n========== SUMMARY ==========")

    print(
        "Original features:",
        len(features)
    )

    print(
        "Highest-ranked feature:",
        importance_df.iloc[0]["feature"]
    )

    print(
        "Lowest-ranked feature:",
        importance_df.iloc[-1]["feature"]
    )

    print(
        "Baseline Macro F1:",
        f"{baseline_score:.6f}"
    )

    print("\nSaved CSV:", csv_file)
    print("Saved graph:", graph_file)

    print("\nSTEP 32 COMPLETED!")


if __name__ == "__main__":
    main()
