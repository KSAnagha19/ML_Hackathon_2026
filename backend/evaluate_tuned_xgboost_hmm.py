
from pathlib import Path
import time
import json

import joblib
import numpy as np
import pandas as pd

from candidate_generation import load_data
from road_network import RoadNetwork
from network_viterbi_decoder import network_viterbi_decode


ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
MODELS = ROOT / "models"

VALIDATION_FILE = RESULTS / "validation_dataset.csv"

ORIGINAL_MODEL_FILE = MODELS / "xgboost.joblib"
TUNED_MODEL_FILE = MODELS / "xgboost_tuned.joblib"

BETA = 20.0
TRANSITION_WEIGHT = 1.0


def build_sequence(trip_df, arcs):
    sequence = []

    for gps_index, group in trip_df.groupby(
        "gps_index", sort=True
    ):
        candidates = []

        for _, row in group.iterrows():
            arc_id = int(row["arc_id"])

            candidates.append({
                "gps_index": int(gps_index),
                "arc_id": arc_id,
                "start_node": int(arcs[arc_id, 0]),
                "end_node": int(arcs[arc_id, 1]),
                "fraction": float(
                    row["projection_fraction"]
                ),
                "road_length_m": float(
                    row["road_length_m"]
                ),
                "gps_step_distance_m": float(
                    row["gps_step_distance_m"]
                ),
                "probability": float(
                    row["probability"]
                ),
                "label": int(row["label"])
            })

        if candidates:
            sequence.append(candidates)

    return sequence


def prepare_data(validation, model_file):
    bundle = joblib.load(model_file)

    model = bundle["model"]
    features = bundle["features"]

    prepared = []

    trip_ids = sorted(
        validation["trip_id"].unique()
    )

    for number, trip_id in enumerate(
        trip_ids, start=1
    ):
        print(
            f"[{number}/{len(trip_ids)}] "
            f"Preparing trip {trip_id}",
            flush=True
        )

        trip_df = validation[
            validation["trip_id"] == trip_id
        ].copy()

        trip_df["probability"] = (
            model.predict_proba(
                trip_df[features]
            )[:, 1]
        )

        nodes, arcs, track = load_data(
            trip_id, "train"
        )

        network = RoadNetwork(nodes, arcs)

        sequence = build_sequence(
            trip_df, arcs
        )

        if sequence:
            prepared.append({
                "trip_id": trip_id,
                "network": network,
                "sequence": sequence
            })

    return prepared


def evaluate(prepared):
    baseline_correct = 0
    hmm_correct = 0
    total = 0

    start = time.perf_counter()

    for item in prepared:
        sequence = item["sequence"]

        baseline = [
            max(
                candidates,
                key=lambda c: c["probability"]
            )
            for candidates in sequence
        ]

        decoded = network_viterbi_decode(
            sequence,
            item["network"],
            beta=BETA,
            transition_weight=TRANSITION_WEIGHT
        )

        if len(decoded) != len(sequence):
            raise ValueError(
                f"Decoder length mismatch: "
                f"{item['trip_id']}"
            )

        for base, hmm in zip(
            baseline, decoded
        ):
            if hmm is None:
                continue

            baseline_correct += int(
                base["label"]
            )

            hmm_correct += int(
                hmm["label"]
            )

            total += 1

    elapsed = time.perf_counter() - start

    if total == 0:
        raise RuntimeError(
            "No GPS observations evaluated."
        )

    return {
        "xgboost_top1": baseline_correct / total,
        "hmm_top1": hmm_correct / total,
        "correct_hmm": hmm_correct,
        "gps_observations": total,
        "hmm_runtime_seconds": elapsed
    }


def main():
    print("\n======================================")
    print("PATHSNAP - TUNED XGBOOST + HMM")
    print("======================================")

    validation = pd.read_csv(
        VALIDATION_FILE,
        dtype={"trip_id": str}
    )

    validation["trip_id"] = (
        validation["trip_id"]
        .str.zfill(8)
    )

    print("\n========== ORIGINAL MODEL ==========")

    original_prepared = prepare_data(
        validation,
        ORIGINAL_MODEL_FILE
    )

    original = evaluate(original_prepared)

    print(
        "Original XGBoost Top-1:",
        f"{original['xgboost_top1']:.6%}"
    )

    print(
        "Original XGBoost + HMM:",
        f"{original['hmm_top1']:.6%}"
    )

    print("\n========== TUNED MODEL ==========")

    tuned_prepared = prepare_data(
        validation,
        TUNED_MODEL_FILE
    )

    tuned = evaluate(tuned_prepared)

    print(
        "Tuned XGBoost Top-1:",
        f"{tuned['xgboost_top1']:.6%}"
    )

    print(
        "Tuned XGBoost + HMM:",
        f"{tuned['hmm_top1']:.6%}"
    )

    improvement = (
        tuned["hmm_top1"]
        - original["hmm_top1"]
    ) * 100

    print("\n========== FINAL COMPARISON ==========")

    print(
        "Original XGBoost + HMM:",
        f"{original['hmm_top1']:.6%}"
    )

    print(
        "Tuned XGBoost + HMM:",
        f"{tuned['hmm_top1']:.6%}"
    )

    print(
        "Improvement:",
        f"{improvement:+.4f} percentage points"
    )

    results = {
        "beta": BETA,
        "transition_weight": TRANSITION_WEIGHT,
        "original": original,
        "tuned": tuned,
        "improvement_percentage_points": improvement
    }

    output = (
        RESULTS / "tuned_xgboost_hmm_comparison.json"
    )

    with open(
        output, "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", output)
    print("\nSTEP 25 COMPLETED!")


if __name__ == "__main__":
    main()
