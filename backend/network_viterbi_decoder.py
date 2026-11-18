
import numpy as np


def network_transition_cost(
    previous,
    current,
    network,
    gps_step_m,
    beta=50.0
):
    """
    Transition cost based on the directed
    road-network shortest path.

    This is an experimental transition model.
    """

    beta = max(float(beta), 1e-6)
    gps_step_m = max(float(gps_step_m), 0.0)

    prev_arc = int(previous["arc_id"])
    curr_arc = int(current["arc_id"])

    prev_fraction = float(previous["fraction"])
    curr_fraction = float(current["fraction"])

    prev_length = float(previous["road_length_m"])
    curr_length = float(current["road_length_m"])

    # Same arc: direct forward movement.
    if prev_arc == curr_arc:
        movement = (
            curr_fraction - prev_fraction
        ) * prev_length

        if movement >= -1e-6:
            return min(
                abs(
                    max(movement, 0.0)
                    - gps_step_m
                ) / beta,
                3.0
            )

        # Backward movement on a directed arc
        # may require a loop through the network.
        # For this experiment, penalise it.
        return 3.0

    # Shortest directed connection between
    # previous arc's end and current arc's start.
    previous_end = int(
        previous["end_node"]
    )
    current_start = int(
        current["start_node"]
    )

    distance, _, _ = network.shortest_path(
        previous_end,
        current_start
    )

    if not np.isfinite(distance):
        # No directed connection exists.
        # Keep as a penalised fallback rather
        # than forcing an impossible graph path.
        return 4.0

    remaining = (
        1.0 - prev_fraction
    ) * prev_length

    progress = (
        curr_fraction * curr_length
    )

    estimated_travel = (
        remaining + distance + progress
    )

    discrepancy = abs(
        estimated_travel - gps_step_m
    )

    return min(
        discrepancy / beta,
        3.0
    )


def decode_segment(
    sequence,
    network,
    beta=50.0,
    transition_weight=0.25
):
    if not sequence:
        return []

    scores = np.array([
        np.log(
            np.clip(
                float(c["probability"]),
                1e-12,
                1.0
            )
        )
        for c in sequence[0]
    ])

    backpointers = []

    for t in range(1, len(sequence)):

        previous = sequence[t - 1]
        current = sequence[t]

        new_scores = np.full(
            len(current),
            -np.inf
        )

        pointers = np.zeros(
            len(current),
            dtype=int
        )

        for j, curr in enumerate(current):

            emission = np.log(
                np.clip(
                    float(curr["probability"]),
                    1e-12,
                    1.0
                )
            )

            best_score = -np.inf
            best_index = 0

            for i, prev in enumerate(previous):

                cost = network_transition_cost(
                    prev,
                    curr,
                    network,
                    curr["gps_step_distance_m"],
                    beta
                )

                score = (
                    scores[i]
                    - transition_weight * cost
                )

                if score > best_score:
                    best_score = score
                    best_index = i

            new_scores[j] = (
                best_score + emission
            )

            pointers[j] = best_index

        scores = new_scores
        backpointers.append(pointers)

    # Recover best candidate sequence.
    index = int(np.argmax(scores))
    indices = [index]

    for pointers in reversed(backpointers):
        index = int(pointers[index])
        indices.append(index)

    indices.reverse()

    return [
        sequence[t][indices[t]]
        for t in range(len(sequence))
    ]


def network_viterbi_decode(
    candidate_sequence,
    network,
    beta=50.0,
    transition_weight=0.25
):
    """
    Decode candidate groups.

    Restart when original GPS indices
    are not consecutive.
    """

    decoded = []
    segment = []
    previous_index = None

    def flush():
        nonlocal segment

        if segment:
            decoded.extend(
                decode_segment(
                    segment,
                    network,
                    beta,
                    transition_weight
                )
            )
            segment = []

    for candidates in candidate_sequence:

        if not candidates:
            flush()
            decoded.append(None)
            previous_index = None
            continue

        gps_index = int(
            candidates[0]["gps_index"]
        )

        if (
            previous_index is not None
            and gps_index != previous_index + 1
        ):
            flush()

        segment.append(candidates)
        previous_index = gps_index

    flush()
    return decoded
