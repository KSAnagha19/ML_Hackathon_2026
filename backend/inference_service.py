
from pathlib import Path
import sys
import time

import joblib
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent

# Allow existing ML scripts to use their original imports.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from candidate_generation import load_data
from feature_extraction import extract_features
from road_network import RoadNetwork
from network_viterbi_decoder import network_viterbi_decode


MODEL_PATH = ROOT / "models" / "xgboost_tuned.joblib"
STUDENT_TEST_DIR = ROOT / "student_test"

BETA = 10.0
TRANSITION_WEIGHT = 0.75

# Load the trained model once, not for every request.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = list(bundle["features"])


def get_available_trips():
    return sorted(
        folder.name
        for folder in STUDENT_TEST_DIR.iterdir()
        if folder.is_dir()
    )


def predict_trip(trip_id):
    start_time = time.perf_counter()

    trip_id = str(trip_id)

    if (
        len(trip_id) != 8
        or not trip_id.isdigit()
        or trip_id not in get_available_trips()
    ):
        raise ValueError(
            "Invalid student-test trip ID."
        )

    nodes, arcs, track = load_data(
        trip_id, "student_test"
    )

    candidate_df = extract_features(
        case_id=trip_id,
        split="student_test",
        k=5,
        radius=100
    )

    total_gps = len(track)

    if candidate_df.empty:
        return {
            "trip_id": trip_id,
            "model": "Optuna-tuned XGBoost + HMM",
            "total_gps_points": total_gps,
            "matched_points": 0,
            "coverage_percent": 0.0,
            "predictions": [],
            "runtime_seconds": round(
                time.perf_counter() - start_time, 3
            )
        }

    candidate_df["probability"] = (
        model.predict_proba(
            candidate_df[features]
        )[:, 1]
    )

    network = RoadNetwork(nodes, arcs)

    candidate_sequence = []

    for gps_index, group in candidate_df.groupby(
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
                )
            })

        if candidates:
            candidate_sequence.append(candidates)

    decoded = network_viterbi_decode(
        candidate_sequence,
        network,
        beta=BETA,
        transition_weight=TRANSITION_WEIGHT
    )

    if len(decoded) != len(candidate_sequence):
        raise RuntimeError(
            "HMM decoder returned an unexpected "
            "number of predictions."
        )

    predictions = []
    changed_count = 0
    decoded_count = 0

    for candidates, hmm in zip(
        candidate_sequence, decoded
    ):
        baseline = max(
            candidates,
            key=lambda item: item["probability"]
        )

        gps_index = int(baseline["gps_index"])

        if hmm is None:
            continue

        decoded_count += 1

        arc_id = int(hmm["arc_id"])

        changed_count += int(
            arc_id != int(baseline["arc_id"])
        )

        start_node = int(arcs[arc_id, 0])
        end_node = int(arcs[arc_id, 1])

        predictions.append({
            "gps_index": gps_index,
            "gps_lon": float(track[gps_index, 0]),
            "gps_lat": float(track[gps_index, 1]),
            "xgboost_arc_id": int(
                baseline["arc_id"]
            ),
            "hmm_arc_id": arc_id,
            "road_start": [
                float(nodes[start_node, 0]),
                float(nodes[start_node, 1])
            ],
            "road_end": [
                float(nodes[end_node, 0]),
                float(nodes[end_node, 1])
            ]
        })

    elapsed = time.perf_counter() - start_time

    return {
        "trip_id": trip_id,
        "model": "Optuna-tuned XGBoost + HMM",
        "hmm_beta": BETA,
        "transition_weight": TRANSITION_WEIGHT,
        "total_gps_points": total_gps,
        "points_with_candidates": len(candidate_sequence),
        "matched_points": decoded_count,
        "coverage_percent": round(
            decoded_count / total_gps * 100, 2
        ) if total_gps else 0.0,
        "changed_predictions": changed_count,
        "runtime_seconds": round(elapsed, 3),
        "predictions": predictions
    }
