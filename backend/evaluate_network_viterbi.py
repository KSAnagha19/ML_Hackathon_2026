
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from candidate_generation import (
    load_data,
    CandidateGenerator
)
from road_network import RoadNetwork
from viterbi_decoder import viterbi_decode
from network_viterbi_decoder import (
    network_viterbi_decode
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

TRIP_ID = "00000011"


def find_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Missing columns {names}. "
        f"Available: {list(df.columns)}"
    )


def main():

    print("\n========== PATHSNAP ==========")
    print("STEP 30: NETWORK-AWARE VITERBI")
    print("--------------------------------")

    validation = pd.read_csv(
        RESULTS / "validation_dataset.csv"
    )

    bundle = joblib.load(
        ROOT / "models"
        / "hist_gradient_boosting.joblib"
    )

    model = bundle["model"]
    features = bundle["features"]

    trip_col = find_column(
        validation,
        ["trip_id", "case_id", "trip"]
    )

    gps_col = find_column(
        validation,
        [
            "gps_index",
            "gps_idx",
            "point_index",
            "observation_index"
        ]
    )

    arc_col = find_column(
        validation,
        ["arc_id", "candidate_arc_id"]
    )

    label_col = find_column(
        validation,
        ["label", "is_correct", "target", "y"]
    )

    validation["_trip"] = (
        validation[trip_col]
        .astype(str)
        .str.zfill(8)
    )

    trip_df = validation[
        validation["_trip"] == TRIP_ID
    ].copy()

    if trip_df.empty:
        raise ValueError(
            f"Trip {TRIP_ID} not found"
        )

    trip_df["probability"] = (
        model.predict_proba(
            trip_df[features]
        )[:, 1]
    )

    nodes, arcs, track = load_data(
        TRIP_ID,
        "train"
    )

    generator = CandidateGenerator(
        nodes,
        arcs
    )

    network = RoadNetwork(
        nodes,
        arcs
    )

    sequence = []
    gps_indices = []

    for gps_index, group in trip_df.groupby(
        gps_col,
        sort=True
    ):

        candidates = []

        for _, row in group.iterrows():

            arc_id = int(row[arc_col])

            candidates.append({
                "gps_index": int(gps_index),
                "arc_id": arc_id,
                "start_node": int(
                    arcs[arc_id, 0]
                ),
                "end_node": int(
                    arcs[arc_id, 1]
                ),
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
                "label": int(row[label_col])
            })

        sequence.append(candidates)
        gps_indices.append(int(gps_index))

    print("Trip:", TRIP_ID)
    print("GPS observations:", len(sequence))
    print("Road nodes:", network.n_nodes)
    print("Road arcs:", network.n_arcs)

    baseline = [
        max(
            candidates,
            key=lambda c: c["probability"]
        )
        for candidates in sequence
    ]

    corrected = viterbi_decode(
        sequence,
        generator.project(nodes[:, :2]),
        beta=50.0,
        transition_weight=0.25
    )

    print("\nRunning network-aware decoder...")
    start = time.perf_counter()

    network_decoded = network_viterbi_decode(
        sequence,
        network,
        beta=50.0,
        transition_weight=0.25
    )

    elapsed = time.perf_counter() - start

    def accuracy(predictions):
        return float(np.mean([
            candidate["label"]
            for candidate in predictions
        ]))

    baseline_acc = accuracy(baseline)
    corrected_acc = accuracy(corrected)
    network_acc = accuracy(network_decoded)

    print("\n========== COMPARISON ==========")
    print(
        "Original ML:",
        f"{baseline_acc:.6f}"
    )
    print(
        "Corrected Viterbi:",
        f"{corrected_acc:.6f}"
    )
    print(
        "Network-aware Viterbi:",
        f"{network_acc:.6f}"
    )

    print(
        "Network vs corrected:",
        f"{network_acc-corrected_acc:+.6f}"
    )

    print(
        "Network decoding time:",
        f"{elapsed:.2f} seconds"
    )

    print(
        "Dijkstra cache:",
        network.cache_info()
    )

    output = RESULTS / (
        "network_viterbi_evaluation_"
        + TRIP_ID + ".csv"
    )

    pd.DataFrame({
        "trip_id": TRIP_ID,
        "gps_index": gps_indices,
        "baseline_arc_id": [
            c["arc_id"] for c in baseline
        ],
        "corrected_arc_id": [
            c["arc_id"] for c in corrected
        ],
        "network_arc_id": [
            c["arc_id"] for c in network_decoded
        ],
        "baseline_correct": [
            c["label"] for c in baseline
        ],
        "corrected_correct": [
            c["label"] for c in corrected
        ],
        "network_correct": [
            c["label"] for c in network_decoded
        ]
    }).to_csv(output, index=False)

    print("\nSaved:", output)
    print("\nSTEP 30 COMPLETED!")


if __name__ == "__main__":
    main()
