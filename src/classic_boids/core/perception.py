from typing import Literal
from .protocols import (
    InternalStateProtocol,
    InputAlphabetProtocol,
)
from .vector import distance, angular_offset


def perception(
    input_alphabet: InputAlphabetProtocol,
    internal_state: InternalStateProtocol,
    perception_type: Literal["separation", "alignment", "cohesion"],
) -> list[int]:
    """
    Identifies entities within a specified distance and field of view.

    Args:
    input_alphabet: Protocol providing positions and velocities of entities.
    internal_state: Protocol with internal state including ID, perception distance, and field of view.

    Returns:
    List of entity indices within the neighborhood meeting the criteria.
    """
    neighborhood = []
    perception_distance = internal_state.perception_distance[perception_type]
    fov_angle = internal_state.perception_field_of_view[perception_type]

    positions = input_alphabet.get_positions()

    # for idx, (position, velocity) in enumerate(zip(positions, velocities)):
    for idx, position in positions.items():
        if idx == internal_state.id:
            continue
        if (
            distance(position_i=position, position_j=internal_state.position)
            < perception_distance
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
