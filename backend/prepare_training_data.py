
from pathlib import Path
import numpy as np
import pandas as pd

from feature_extraction import extract_features
from candidate_generation import load_data

ROOT = Path(__file__).resolve().parent.parent

CASE_ID = "00000001"
SPLIT = "train"

# Maximum allowed distance for a confident label
MAX_LABEL_DISTANCE_M = 100.0

# Controls how strongly alignment prefers
# progressing along the reference route.
PROGRESS_PENALTY = 0.5


def load_reference_route(case_id, split):
    path = (
        ROOT / split / case_id /
        f"{case_id}.route"
    )
    return np.loadtxt(
        path, dtype=int, ndmin=1
    ).reshape(-1)


def build_alignment(nodes, arcs, track, route):
    """
    Align GPS points to ordered reference arcs.

    Each observation is assigned a non-decreasing
    position in the reference route.
    """

    if len(route) == 0:
        raise ValueError("Empty reference route")

    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError("Invalid reference arc ID")

    # Local projection in metres
    lon0 = float(np.mean(nodes[:, 0]))
    lat0 = float(np.mean(nodes[:, 1]))

    scale = np.array([
        111320 * np.cos(np.radians(lat0)),
        111320
    ])

    xy = (nodes[:, :2] - [lon0, lat0]) * scale
    gps_xy = (track[:, :2] - [lon0, lat0]) * scale

    reference_arcs = arcs[route]

    a = xy[reference_arcs[:, 0]]
    b = xy[reference_arcs[:, 1]]
    ab = b - a

    lengths_sq = np.sum(ab * ab, axis=1)

    # Cost matrix: GPS observations x route arcs
    costs = np.empty(
        (len(track), len(route)),
        dtype=np.float32
    )

    for i, point in enumerate(gps_xy):
        ap = point - a

        t = np.divide(
            np.sum(ap * ab, axis=1),
            lengths_sq,
            out=np.zeros(len(route)),
            where=lengths_sq > 0
        )

        t = np.clip(t, 0, 1)
        projections = a + t[:, None] * ab

        costs[i] = np.linalg.norm(
            projections - point,
            axis=1
        )

    # Dynamic programming with monotonic progress
    n, m = costs.shape

    dp = np.full(m, np.inf)
    dp[:] = costs[0]

    backpointers = np.zeros(
        (n, m), dtype=np.int32
    )

    for i in range(1, n):
        # Prefix minimum of:
        # dp_previous[j] - penalty*j
        # allowing previous position j <= current k.
        adjusted = (
            dp - PROGRESS_PENALTY * np.arange(m)
        )

        prefix_min = np.empty(m)
        prefix_arg = np.empty(m, dtype=np.int32)

        best = np.inf
        best_idx = 0

        for k in range(m):
            if adjusted[k] < best:
                best = adjusted[k]
                best_idx = k

            prefix_min[k] = best
            prefix_arg[k] = best_idx

        dp = (
            costs[i]
            + PROGRESS_PENALTY * np.arange(m)
            + prefix_min
        )

        backpointers[i] = prefix_arg

    positions = np.empty(n, dtype=np.int32)
    positions[-1] = int(np.argmin(dp))

    for i in range(n - 1, 0, -1):
        positions[i - 1] = (
            backpointers[i, positions[i]]
        )

    aligned_arcs = route[positions]

    aligned_distances = costs[
        np.arange(n), positions
    ]

    return aligned_arcs, aligned_distances


def prepare_training_data():
    print("\n========== PATHSNAP ==========")
    print("Step 7: Training Label Preparation")
    print("----------------------------------")

    nodes, arcs, track = load_data(
        CASE_ID, SPLIT
    )

    route = load_reference_route(
        CASE_ID, SPLIT
    )

    print("Reference route arcs:", len(route))
    print("GPS observations:", len(track))

    aligned_arcs, aligned_distances = (
        build_alignment(nodes, arcs, track, route)
    )

    df = extract_features(
        case_id=CASE_ID,
        split=SPLIT,
        k=5,
        radius=100
    )

    # Align reference arc to each GPS observation
    df["aligned_arc_id"] = df["gps_index"].map(
        lambda i: int(aligned_arcs[i])
    )

    df["alignment_distance_m"] = (
        df["gps_index"].map(
            lambda i: float(aligned_distances[i])
        )
    )

    df["label"] = (
        df["arc_id"] == df["aligned_arc_id"]
    ).astype(int)

    # Keep only observations where the
    # reference alignment is reasonably close.
    confident = (
        df["alignment_distance_m"]
        <= MAX_LABEL_DISTANCE_M
    )

    # A positive candidate must exist in the
    # generated candidate set.
    positive_per_gps = (
        df.groupby("gps_index")["label"]
        .transform("max")
    )

    usable = confident & (positive_per_gps == 1)

    labelled = df.loc[usable].copy()

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir / "training_sample.csv"
    )

    labelled.to_csv(output_path, index=False)

    print("\nTotal candidate rows:", len(df))
    print("Usable labelled rows:", len(labelled))
    print(
        "Usable GPS observations:",
        labelled["gps_index"].nunique()
    )
    print(
        "Positive labels:",
        int(labelled["label"].sum())
    )
    print(
        "Negative labels:",
        int((labelled["label"] == 0).sum())
    )
    print("\nSaved to:", output_path)

    print("\nFirst 10 labelled rows:")
    print(
        labelled[
            [
                "gps_index",
                "arc_id",
                "aligned_arc_id",
                "distance_m",
                "heading_difference",
                "label"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTraining label preparation complete!")


if __name__ == "__main__":
    prepare_training_data()
