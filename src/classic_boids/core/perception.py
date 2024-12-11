from typing import Literal
from dataclasses import dataclass

# from classic_boids.core.protocols import Neighborhood
from .protocols import (
    InternalStateProtocol,
    InputAlphabetProtocol,
    BoidID,
    NeighborhoodProtocol,
    VectorType,
)
from .vector import distance, angular_offset


@dataclass
class Neighborhood(NeighborhoodProtocol):
    ids: list[BoidID]
    info: dict[BoidID, tuple[VectorType, VectorType]]


def perception(
    input_alphabet: InputAlphabetProtocol,
    internal_state: InternalStateProtocol,
    perception_type: Literal["separation", "alignment", "cohesion"],
) -> Neighborhood:
    """
    Identifies entities within a specified distance and field of view, returning a Neighborhood instance.

    Args:
        input_alphabet: Protocol providing positions and velocities of entities.
        internal_state: Protocol with internal state including ID, perception distance, and field of view.
        perception_type: The type of perception ("separation", "alignment", or "cohesion").

    Returns:
        A Neighborhood object containing:
          - ids: List of BoidID in the neighborhood
          - info: Dict mapping each BoidID to (position, velocity)
    """
    perception_distance = internal_state.perception_distance[perception_type]
    fov_angle = internal_state.perception_field_of_view[perception_type]

    positions = input_alphabet.get_positions()
    velocities = input_alphabet.get_velocities()

    neighborhood_ids = []

    for idx, position in positions.items():
        if BoidID(idx) == internal_state.id:
            continue
        dist = distance(position_i=position, position_j=internal_state.position)
        angle = angular_offset(
            position_i=position,
            position_j=internal_state.position,
            velocity_j=internal_state.velocity,
        )

        if dist < perception_distance and angle < fov_angle:
            neighborhood_ids.append(BoidID(idx))

    # Build the info dictionary
    neighborhood_info = {
        boid_id: (positions[int(boid_id)], velocities[int(boid_id)])
        for boid_id in neighborhood_ids
    }

    return Neighborhood(ids=neighborhood_ids, info=neighborhood_info)


# TODO eventually this should return a touple of evaluated perception functions (ie neighborhoods)
# def process_perception(
#     perception_function: PerceptionFunctionProtocol,
#     input_alphabet: InputAlphabetProtocol,
# ) -> None:
#     perception_function(input_alphabet=input_alphabet)
