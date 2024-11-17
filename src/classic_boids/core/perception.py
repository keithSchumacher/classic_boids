from .protocols import (
    InternalStateProtocol,
    # PerceptionFunctionProtocol,
    InputAlphabetProtocol,
    VectorType,
)
import numpy as np


def distance(position_i: VectorType, position_j: VectorType) -> float:
    """
    Distnace of Boid B_i from observed Boid B_j
    """
    return (position_i - position_j).norm()


def angular_offset(
    position_i: VectorType, position_j: VectorType, velocity_j: VectorType
) -> float:
    """Angular offset of Boid B_j from Boid B_i"""
    difference = position_i - position_j
    numerator = velocity_j.dot(difference)
    difference_magnitude = difference.norm()
    velocity_magnitude = velocity_j.norm()
    denominator = velocity_magnitude * difference_magnitude
    if difference_magnitude == 0.0:
        return 0.0
    if velocity_magnitude == 0.0:
        raise ValueError("Angular offset cannot be calculated if velocity is zero.")
    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be zero.")

    return np.arccos(numerator / denominator)


def separation_perception(
    input_alphabet: InputAlphabetProtocol, internal_state: InternalStateProtocol
) -> list[int]:
    """
    Identifies entities within a specified separation distance and field of view.

    Args:
    input_alphabet: Protocol providing positions and velocities of entities.
    internal_state: Protocol with internal state including ID, perception distance, and field of view.

    Returns:
    List of entity indices within the neighborhood meeting the criteria.
    """
    neighborhood = []
    separation_distance = internal_state.perception_distance.separation
    fov_angle = internal_state.perception_field_of_view.separation

    positions = input_alphabet.get_positions()

    # for idx, (position, velocity) in enumerate(zip(positions, velocities)):
    for idx, position in positions.items():
        if idx == internal_state.id:
            continue
        if (
            distance(position_i=position, position_j=internal_state.position)
            < separation_distance
            and angular_offset(
                position_i=position,
                position_j=internal_state.position,
                velocity_j=internal_state.velocity,
            )
            < fov_angle
        ):
            neighborhood.append(idx)

    return neighborhood


# TODO eventually this should return a touple of evaluated perception functions (ie neighborhoods)
# def process_perception(
#     perception_function: PerceptionFunctionProtocol,
#     input_alphabet: InputAlphabetProtocol,
# ) -> None:
#     perception_function(input_alphabet=input_alphabet)
