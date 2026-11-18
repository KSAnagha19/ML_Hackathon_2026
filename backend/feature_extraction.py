
from pathlib import Path
import numpy as np
import pandas as pd

from candidate_generation import (
    load_data,
    CandidateGenerator
)

ROOT = Path(__file__).resolve().parent.parent


def calculate_bearing(point1, point2):
    """
    Calculate compass bearing in degrees.
    Coordinates are [longitude, latitude].
    """
    lon1, lat1 = np.radians(point1[:2])
    lon2, lat2 = np.radians(point2[:2])

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))

    return float((bearing + 360) % 360)


def heading_difference(bearing1, bearing2):
    """Smallest angular difference (0 to 180 degrees)."""
    return abs((bearing1 - bearing2 + 180) % 360 - 180)


def extract_features(
    case_id="00000001",
    split="train",
    k=5,
    radius=100
):
    nodes, arcs, track = load_data(case_id, split)

    generator = CandidateGenerator(nodes, arcs)

    rows = []

    print(f"Processing {split}/{case_id}")
    print(f"GPS observations: {len(track)}")

    # GPS points must be in their recorded order.
    for i in range(len(track)):

        gps_point = track[i, :2]

        # Use neighboring observations to estimate
        # movement direction.
        if len(track) == 1:
            gps_bearing = 0.0
            gps_step_distance = 0.0
            heading_valid = 0

        else:
            if i == 0:
                p1 = track[i, :2]
                p2 = track[i + 1, :2]
            else:
                p1 = track[i - 1, :2]
                p2 = track[i, :2]

            xy = generator.project(
                np.asarray([p1, p2])
            )

            gps_step_distance = float(
                np.linalg.norm(xy[1] - xy[0])
            )

            heading_valid = int(
                gps_step_distance >= 3.0
            )

            gps_bearing = (
                calculate_bearing(p1, p2)
                if heading_valid else 0.0
            )

        candidates = generator.find_candidates(
            gps_point,
            k=k,
            radius=radius
        )

        for rank, candidate in enumerate(
            candidates, start=1
        ):
            arc_id = candidate["arc_id"]

            start_node = candidate["start_node"]
            end_node = candidate["end_node"]

            start_coord = nodes[start_node, :2]
            end_coord = nodes[end_node, :2]

            road_bearing = calculate_bearing(
                start_coord,
                end_coord
            )

            difference = (
                heading_difference(
                    gps_bearing,
                    road_bearing
                )
                if heading_valid else 180.0
            )

            road_length = float(
                generator.lengths[arc_id]
            )

            rows.append({
                "trip_id": case_id,
                "gps_index": i,
                "gps_lon": float(gps_point[0]),
                "gps_lat": float(gps_point[1]),
                "arc_id": arc_id,
                "candidate_rank": rank,
                "distance_m": candidate["distance_m"],
                "road_length_m": road_length,
                "gps_bearing": gps_bearing,
                "road_bearing": road_bearing,
                "heading_difference": difference,
                "heading_valid": heading_valid,
                "projection_fraction": candidate["fraction"],
                "gps_step_distance_m": gps_step_distance
            })

    df = pd.DataFrame(rows)

    print("Generated candidate rows:", len(df))

    return df


if __name__ == "__main__":

    df = extract_features(
        case_id="00000001",
        split="train",
        k=5,
        radius=100
    )

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "features_sample.csv"

    df.to_csv(output_path, index=False)

    print("\n========== FEATURE EXTRACTION ==========")
    print("Feature table shape:", df.shape)
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nSaved to:", output_path)
    print("\nFeature extraction completed!")
