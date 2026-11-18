
import numpy as np


def endpoint_transition_cost(
    previous,
    current,
    node_xy,
    gps_step_m,
    beta=50.0
):
    """
    Approximate transition cost between road candidates.

    Cases:
    1. Same directed road
    2. Directly connected directed roads
    3. Non-adjacent roads

    This is a baseline transition model, not
    a network shortest-path calculation.
    """

    prev_arc = int(previous["arc_id"])
    curr_arc = int(current["arc_id"])

    prev_fraction = float(previous["fraction"])
    curr_fraction = float(current["fraction"])

    prev_length = float(previous["road_length_m"])
    curr_length = float(current["road_length_m"])

    gps_step_m = max(float(gps_step_m), 0.0)

    beta = max(float(beta), 1e-6)

    # CASE 1: Same road
    if prev_arc == curr_arc:

        movement = (
            curr_fraction - prev_fraction
        ) * prev_length

        if movement >= -1e-6:

            discrepancy = abs(
                max(movement, 0.0) - gps_step_m
            )

            return min(
                discrepancy / beta,
                2.0
            )

        # Moving backward on a directed arc
        return 3.0

    # CASE 2: Directly connected roads
    if (
        int(previous["end_node"])
        == int(current["start_node"])
    ):

        remaining_distance = (
            1.0 - prev_fraction
        ) * prev_length

        progress_distance = (
            curr_fraction * curr_length
        )

        estimated_travel = (
            remaining_distance
            + progress_distance
        )

        discrepancy = abs(
            estimated_travel - gps_step_m
        )

        return min(
            discrepancy / beta,
            2.5
        )

    # CASE 3: Other transitions
    #
    # Connectivity is unknown without a
    # road-network shortest-path search.
    return 2.0


def decode_segment(
    sequence,
    node_xy,
    beta=50.0,
    transition_weight=0.25
):
    """
    Viterbi decoding for one continuous
    sequence of GPS observations.
    """

    if not sequence:
        return []

    scores = []
    backpointers = []

    # Initial emissions
    first_scores = np.array([
        np.log(
            np.clip(
                float(candidate["probability"]),
                1e-12,
                1.0
            )
        )
        for candidate in sequence[0]
    ])

    scores.append(first_scores)
    backpointers.append(None)

    # Forward dynamic programming
    for t in range(1, len(sequence)):

        previous_candidates = sequence[t - 1]
        current_candidates = sequence[t]

        current_scores = np.full(
            len(current_candidates),
            -np.inf
        )

        current_backpointers = np.zeros(
            len(current_candidates),
            dtype=int
        )

        for j, current in enumerate(
            current_candidates
        ):

            emission = np.log(
                np.clip(
                    float(current["probability"]),
                    1e-12,
                    1.0
                )
            )

            transition_scores = []

            for i, previous in enumerate(
                previous_candidates
            ):

                transition_cost = (
                    endpoint_transition_cost(
                        previous,
                        current,
                        node_xy,
                        current["gps_step_distance_m"],
                        beta
                    )
                )

                transition_score = (
                    scores[-1][i]
                    - transition_weight
                    * transition_cost
                )

                transition_scores.append(
                    transition_score
                )

            transition_scores = np.asarray(
                transition_scores
            )

            best_previous = int(
                np.argmax(transition_scores)
            )

            current_scores[j] = (
                transition_scores[best_previous]
                + emission
            )

            current_backpointers[j] = (
                best_previous
            )

        scores.append(current_scores)

        backpointers.append(
            current_backpointers
        )

    # Backtracking
    last_index = int(
        np.argmax(scores[-1])
    )

    selected_indices = [last_index]

    for t in range(
        len(sequence) - 1,
        0,
        -1
    ):

        last_index = int(
            backpointers[t][last_index]
        )

        selected_indices.append(
            last_index
        )

    selected_indices.reverse()

    return [
        sequence[t][selected_indices[t]]
        for t in range(len(sequence))
    ]


def viterbi_decode(
    candidate_sequence,
    node_xy,
    beta=50.0,
    transition_weight=0.25
):
    """
    Decode candidate sequences.

    Missing candidate groups are returned
    as None.

    Decoding restarts whenever original GPS
    indices are not consecutive.
    """

    if not candidate_sequence:
        return []

    decoded = []

    current_segment = []

    previous_gps_index = None

    def flush_segment():
        nonlocal current_segment

        if current_segment:

            selected = decode_segment(
                current_segment,
                node_xy,
                beta,
                transition_weight
            )

            decoded.extend(selected)

            current_segment = []

    for candidates in candidate_sequence:

        if not candidates:

            flush_segment()

            decoded.append(None)

            previous_gps_index = None

            continue

        current_gps_index = (
            candidates[0].get("gps_index")
        )

        if (
            previous_gps_index is not None
            and current_gps_index is not None
            and current_gps_index
            != previous_gps_index + 1
        ):
            flush_segment()

        current_segment.append(candidates)

        previous_gps_index = (
            current_gps_index
        )

    flush_segment()

    return decoded
