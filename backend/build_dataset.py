
from pathlib import Path
import numpy as np
import pandas as pd

from candidate_generation import load_data
from feature_extraction import extract_features
from prepare_training_data import build_alignment, load_reference_route

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "trip_features"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MAX_LABEL_DISTANCE_M = 100.0
K = 5
RADIUS = 100


def process_trip(case_id):
    """Generate provisional labels for one trip."""

    cache_path = CACHE_DIR / f"{case_id}.csv"

    if cache_path.exists():
        print(f"Using cached trip {case_id}")
        return pd.read_csv(
            cache_path,
            dtype={"trip_id": str}
        )

    nodes, arcs, track = load_data(
        case_id, "train"
    )

    route = load_reference_route(
        case_id, "train"
    )

    aligned_arcs, aligned_distances = build_alignment(
        nodes, arcs, track, route
    )

    df = extract_features(
        case_id=case_id,
        split="train",
        k=K,
        radius=RADIUS
    )

    if df.empty:
        raise ValueError("No candidate roads generated")

    # Each GPS point receives one provisional
    # reference arc from route alignment.
    df["aligned_arc_id"] = aligned_arcs[
        df["gps_index"].to_numpy(dtype=int)
    ]

    df["alignment_distance_m"] = aligned_distances[
        df["gps_index"].to_numpy(dtype=int)
    ]

    df["label"] = (
        df["arc_id"] == df["aligned_arc_id"]
    ).astype(int)

    # Only retain GPS observations with:
    # 1. A reasonably close reference alignment
    # 2. At least one positive candidate
    confident = (
        df["alignment_distance_m"]
        <= MAX_LABEL_DISTANCE_M
    )

    has_positive = (
        df.groupby("gps_index")["label"]
        .transform("max") == 1
    )

    df = df.loc[
        confident & has_positive
    ].copy()

    df.to_csv(cache_path, index=False)

    return df


def main():
    trip_ids = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    )

    print("\n========== PATHSNAP ==========")
    print("STEP 8: FULL DATASET PREPARATION")
    print("--------------------------------")
    print("Training trips found:", len(trip_ids))

    if len(trip_ids) < 2:
        raise ValueError(
            "At least two trips are required"
        )

    # Fixed random seed for reproducibility
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(trip_ids)

    split_index = int(len(shuffled) * 0.8)

    train_ids = set(shuffled[:split_index])
    validation_ids = set(shuffled[split_index:])

    train_frames = []
    validation_frames = []
    stats = []
    failed_trips = []

    for number, case_id in enumerate(
        trip_ids, start=1
    ):
        print(
            f"\n[{number}/{len(trip_ids)}] "
            f"Processing trip {case_id}"
        )

        try:
            df = process_trip(case_id)

            if df.empty:
                raise ValueError(
                    "No usable labelled rows"
                )

            gps_count = df["gps_index"].nunique()
            positive_count = int(
                df["label"].sum()
            )

            print("Labelled rows:", len(df))
            print("Usable GPS points:", gps_count)
            print("Positive labels:", positive_count)

            stats.append({
                "trip_id": case_id,
                "split": (
                    "train" if case_id in train_ids
                    else "validation"
                ),
                "candidate_rows": len(df),
                "usable_gps_points": gps_count,
                "positive_labels": positive_count
            })

            if case_id in train_ids:
                train_frames.append(df)
            else:
                validation_frames.append(df)

        except Exception as error:
            print(
                f"ERROR in trip {case_id}: {error}"
            )
            failed_trips.append({
                "trip_id": case_id,
                "error": str(error)
            })

    if not train_frames or not validation_frames:
        raise RuntimeError(
            "Training or validation data is empty"
        )

    training_df = pd.concat(
        train_frames,
        ignore_index=True
    )

    validation_df = pd.concat(
        validation_frames,
        ignore_index=True
    )

    training_df.to_csv(
        RESULTS_DIR / "train_dataset.csv",
        index=False
    )

    validation_df.to_csv(
        RESULTS_DIR / "validation_dataset.csv",
        index=False
    )

    pd.DataFrame(stats).to_csv(
        RESULTS_DIR / "dataset_statistics.csv",
        index=False
    )

    pd.DataFrame(
        failed_trips,
        columns=["trip_id", "error"]
    ).to_csv(
        RESULTS_DIR / "failed_trips.csv",
        index=False
    )

    print("\n========== FINAL SUMMARY ==========")
    print("Training rows:", len(training_df))
    print("Validation rows:", len(validation_df))

    print(
        "Training trips:",
        training_df["trip_id"].nunique()
    )

    print(
        "Validation trips:",
        validation_df["trip_id"].nunique()
    )

    print(
        "Training positive labels:",
        int(training_df["label"].sum())
    )

    print(
        "Validation positive labels:",
        int(validation_df["label"].sum())
    )

    print(
        "Failed trips:",
        len(failed_trips)
    )

    print("\nFiles saved in results/")
    print("train_dataset.csv")
    print("validation_dataset.csv")
    print("dataset_statistics.csv")
    print("failed_trips.csv")

    print("\nSTEP 8 COMPLETED!")


if __name__ == "__main__":
    main()
