
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent


def load_data(case_id="00000000", split="student_test"):
    folder = ROOT / split / case_id

    nodes = np.loadtxt(
        folder / f"{case_id}.nodes",
        ndmin=2
    )

    arcs = np.loadtxt(
        folder / f"{case_id}.arcs",
        dtype=int,
        ndmin=2
    )

    track = np.loadtxt(
        folder / f"{case_id}.track",
        ndmin=2
    )

    return nodes, arcs, track


class CandidateGenerator:
    def __init__(self, nodes, arcs):
        self.nodes = nodes
        self.arcs = arcs

        # Convert longitude/latitude to approximate
        # local coordinates in metres.
        self.lon0 = float(np.mean(nodes[:, 0]))
        self.lat0 = float(np.mean(nodes[:, 1]))

        self.scale_x = (
            111320 * np.cos(np.radians(self.lat0))
        )
        self.scale_y = 111320

        xy = self.project(nodes[:, :2])

        # Road segment endpoints
        self.starts = xy[arcs[:, 0]]
        self.ends = xy[arcs[:, 1]]

        # Segment midpoint spatial index
        midpoints = (self.starts + self.ends) / 2
        self.tree = cKDTree(midpoints)

        # Segment lengths
        self.lengths = np.linalg.norm(
            self.ends - self.starts,
            axis=1
        )

        self.max_half_length = (
            float(np.max(self.lengths)) / 2
        )

    def project(self, coordinates):
        coordinates = np.asarray(
            coordinates,
            dtype=float
        )

        x = (
            coordinates[..., 0] - self.lon0
        ) * self.scale_x

        y = (
            coordinates[..., 1] - self.lat0
        ) * self.scale_y

        return np.stack([x, y], axis=-1)

    def find_candidates(
        self,
        gps_point,
        k=5,
        radius=100
    ):
        point = self.project(
            np.asarray(gps_point[:2])
        )

        # Midpoint distance alone can miss long
        # road segments. Expand search by the
        # maximum segment half-length.
        search_radius = radius + self.max_half_length

        nearby_ids = self.tree.query_ball_point(
            point,
            r=search_radius
        )

        if not nearby_ids:
            return []

        nearby_ids = np.asarray(
            nearby_ids,
            dtype=int
        )

        a = self.starts[nearby_ids]
        b = self.ends[nearby_ids]

        ab = b - a
        ap = point - a

        squared_lengths = np.sum(ab * ab, axis=1)

        t = np.divide(
            np.sum(ap * ab, axis=1),
            squared_lengths,
            out=np.zeros_like(squared_lengths),
            where=squared_lengths > 0
        )

        t = np.clip(t, 0, 1)

        projected_points = a + t[:, None] * ab

        distances = np.linalg.norm(
            projected_points - point,
            axis=1
        )

        valid = np.where(distances <= radius)[0]

        if len(valid) == 0:
            return []

        nearest = valid[
            np.argsort(distances[valid])[:k]
        ]

        candidates = []

        for idx in nearest:
            arc_id = int(nearby_ids[idx])

            candidates.append({
                "arc_id": arc_id,
                "distance_m": round(
                    float(distances[idx]), 2
                ),
                "start_node": int(
                    self.arcs[arc_id, 0]
                ),
                "end_node": int(
                    self.arcs[arc_id, 1]
                ),
                "fraction": float(t[idx])
            })

        return candidates


if __name__ == "__main__":
    nodes, arcs, track = load_data()

    generator = CandidateGenerator(nodes, arcs)

    print("\n========== PATHSNAP ==========")
    print("Candidate Road Generation")
    print("--------------------------------")

    print("Road nodes:", len(nodes))
    print("Road segments:", len(arcs))
    print("GPS observations:", len(track))

    print("\nFirst GPS observation:")
    print(track[0])

    candidates = generator.find_candidates(
        track[0],
        k=5,
        radius=100
    )

    print("\nNearest candidate road segments:")

    for rank, candidate in enumerate(
        candidates, start=1
    ):
        print(
            f"{rank}. Arc {candidate['arc_id']} "
            f"| Distance: {candidate['distance_m']} m "
            f"| Nodes: "
            f"{candidate['start_node']} -> "
            f"{candidate['end_node']}"
        )

    print("\nCandidate generation completed!")
