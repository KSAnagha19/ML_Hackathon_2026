
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RoadNetwork:
    """
    Directed road graph built from PathSnap
    .nodes and .arcs files.

    Arc IDs are zero-based row indices.
    """

    def __init__(self, nodes, arcs):

        self.nodes = np.asarray(nodes, dtype=float)
        self.arcs = np.asarray(arcs, dtype=int)

        if self.nodes.ndim != 2 or self.nodes.shape[1] < 2:
            raise ValueError("Invalid nodes array")

        if self.arcs.ndim != 2 or self.arcs.shape[1] < 2:
            raise ValueError("Invalid arcs array")

        self.n_nodes = len(self.nodes)
        self.n_arcs = len(self.arcs)

        self.start_nodes = self.arcs[:, 0]
        self.end_nodes = self.arcs[:, 1]

        if (
            np.any(self.start_nodes < 0)
            or np.any(self.end_nodes < 0)
            or np.any(self.start_nodes >= self.n_nodes)
            or np.any(self.end_nodes >= self.n_nodes)
        ):
            raise ValueError("Invalid arc endpoint index")

        self.node_xy = self._project_nodes()

        self.arc_lengths = np.linalg.norm(
            self.node_xy[self.end_nodes]
            - self.node_xy[self.start_nodes],
            axis=1
        )

        # Keep only the shortest arc for each
        # directed node pair.
        self.arc_lookup = {}

        for arc_id, (u, v) in enumerate(
            self.arcs[:, :2]
        ):
            key = (int(u), int(v))

            old = self.arc_lookup.get(key)

            if (
                old is None
                or self.arc_lengths[arc_id]
                < self.arc_lengths[old]
            ):
                self.arc_lookup[key] = arc_id

        rows = []
        cols = []
        weights = []

        for (u, v), arc_id in self.arc_lookup.items():
            rows.append(u)
            cols.append(v)

            weights.append(
                max(
                    float(self.arc_lengths[arc_id]),
                    1e-6
                )
            )

        self.graph = csr_matrix(
            (
                weights,
                (rows, cols)
            ),
            shape=(self.n_nodes, self.n_nodes)
        )

    def _project_nodes(self):

        lon = self.nodes[:, 0]
        lat = self.nodes[:, 1]

        lon0 = float(np.mean(lon))
        lat0 = float(np.mean(lat))

        scale_x = (
            111320.0
            * np.cos(np.radians(lat0))
        )

        x = (lon - lon0) * scale_x
        y = (lat - lat0) * 111320.0

        return np.column_stack((x, y))

    @lru_cache(maxsize=256)
    def _shortest_paths_from(self, start_node):
        """
        Cache Dijkstra results by source node.

        Each cached result contains distances
        and predecessors to all other nodes.
        """

        return dijkstra(
            self.graph,
            directed=True,
            indices=int(start_node),
            return_predecessors=True
        )

    def shortest_path(self, start_node, end_node):

        start_node = int(start_node)
        end_node = int(end_node)

        if not (
            0 <= start_node < self.n_nodes
            and 0 <= end_node < self.n_nodes
        ):
            raise ValueError("Invalid node index")

        if start_node == end_node:
            return 0.0, [start_node], []

        distances, predecessors = (
            self._shortest_paths_from(start_node)
        )

        distance = float(distances[end_node])

        if not np.isfinite(distance):
            return float("inf"), [], []

        node_path = [end_node]
        current = end_node

        while current != start_node:

            previous = int(predecessors[current])

            if previous < 0:
                return float("inf"), [], []

            node_path.append(previous)
            current = previous

        node_path.reverse()

        arc_path = []

        for u, v in zip(
            node_path[:-1],
            node_path[1:]
        ):
            arc_id = self.arc_lookup.get(
                (int(u), int(v))
            )

            if arc_id is None:
                raise RuntimeError(
                    "Could not recover road arc"
                )

            arc_path.append(int(arc_id))

        return distance, node_path, arc_path

    def arc_to_arc_path(
        self,
        previous_arc_id,
        current_arc_id
    ):
        """
        Find the connecting road path between
        two directed candidate arcs.

        Does not include the candidate arcs
        themselves.
        """

        previous_arc_id = int(previous_arc_id)
        current_arc_id = int(current_arc_id)

        previous_end = int(
            self.end_nodes[previous_arc_id]
        )

        current_start = int(
            self.start_nodes[current_arc_id]
        )

        distance, _, arc_path = self.shortest_path(
            previous_end,
            current_start
        )

        return distance, arc_path

    def cache_info(self):
        return self._shortest_paths_from.cache_info()


def main():

    from candidate_generation import load_data

    print("\n========== PATHSNAP ==========")
    print("STEP 29: MULTI-EDGE PATH TEST")
    print("--------------------------------")

    trip_id = "00000011"

    nodes, arcs, track = load_data(
        trip_id,
        "train"
    )

    network = RoadNetwork(nodes, arcs)

    print("Trip:", trip_id)
    print("Road nodes:", network.n_nodes)
    print("Road arcs:", network.n_arcs)

    # Find a pair of nodes that requires
    # multiple arcs in the shortest path.
    test_result = None

    # Test selected source nodes.
    # This avoids an expensive all-pairs search.
    source_nodes = np.unique(
        network.start_nodes
    )[:30]

    for start_node in source_nodes:

        distances, predecessors = (
            network._shortest_paths_from(
                int(start_node)
            )
        )

        reachable = np.where(
            np.isfinite(distances)
            & (distances > 0)
        )[0]

        # Prefer farther reachable destinations
        # to increase the chance of a multi-edge path.
        candidates = reachable[
            np.argsort(
                distances[reachable]
            )[::-1][:30]
        ]

        for end_node in candidates:

            distance, node_path, arc_path = (
                network.shortest_path(
                    int(start_node),
                    int(end_node)
                )
            )

            if len(arc_path) >= 3:
                test_result = (
                    int(start_node),
                    int(end_node),
                    distance,
                    node_path,
                    arc_path
                )
                break

        if test_result is not None:
            break

    if test_result is None:
        print(
            "No multi-edge test path found "
            "among sampled nodes."
        )
        return

    (
        start_node,
        end_node,
        distance,
        node_path,
        arc_path
    ) = test_result

    print("\n========== PATH RESULT ==========")
    print("Start node:", start_node)
    print("End node:", end_node)
    print("Distance:", f"{distance:.2f} m")
    print("Path nodes:", len(node_path))
    print("Path arcs:", len(arc_path))
    print("First 10 arc IDs:", arc_path[:10])

    # Check that recovered arcs follow
    # the node path exactly.
    valid = (
        len(arc_path) == len(node_path) - 1
    )

    for i, arc_id in enumerate(arc_path):

        u = int(arcs[arc_id, 0])
        v = int(arcs[arc_id, 1])

        if (
            u != node_path[i]
            or v != node_path[i + 1]
        ):
            valid = False
            break

    print("Directed continuity:", valid)

    # Repeat query to demonstrate cache reuse.
    network.shortest_path(
        start_node,
        end_node
    )

    print("Cache statistics:", network.cache_info())

    if not valid:
        raise RuntimeError(
            "Shortest-path continuity test failed"
        )

    print("\nSTEP 29 COMPLETED!")


if __name__ == "__main__":
    main()
