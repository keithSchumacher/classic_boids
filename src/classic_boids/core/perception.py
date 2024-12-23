from dataclasses import dataclass

from .protocols import (
    DriveName,
    InternalStateProtocol,
    InputAlphabetProtocol,
    BoidID,
    NeighborhoodProtocol,
    PerceptionFunctionProtocol,
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
    perception_type: DriveName,
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
    neighborhood_info = {boid_id: (positions[int(boid_id)], velocities[int(boid_id)]) for boid_id in neighborhood_ids}

    return Neighborhood(ids=neighborhood_ids, info=neighborhood_info)


def compute_perceptions(
    perception_functions: dict[DriveName, PerceptionFunctionProtocol],
    input_alphabet: InputAlphabetProtocol,
    internal_state: InternalStateProtocol,
) -> dict[DriveName, NeighborhoodProtocol]:
    """
    Compute the perception of each drive (SEPARATION, ALIGNMENT, COHESION) by
    calling the corresponding function and return them in a dictionary.

    :param perception_functions:
        A mapping of DriveName to a callable that takes (input_alphabet, internal_state, drive_name)
        and returns a NeighborhoodProtocol describing visible neighbors for that drive.
    :param input_alphabet:
        The global input data (positions, velocities, etc.) to check against.
    :param internal_state:
        The internal state of the boid (or entity) for which we are computing perceptions.
    :return:
        A dictionary mapping each DriveName to the resulting NeighborhoodProtocol,
        containing the neighbor IDs and their information.
    """
    p_s = perception_functions[DriveName.SEPARATION](input_alphabet, internal_state, DriveName.SEPARATION)
    p_a = perception_functions[DriveName.ALIGNMENT](input_alphabet, internal_state, DriveName.ALIGNMENT)
    p_c = perception_functions[DriveName.COHESION](input_alphabet, internal_state, DriveName.COHESION)
    return {
        DriveName.SEPARATION: p_s,
        DriveName.ALIGNMENT: p_a,
        DriveName.COHESION: p_c,
    }
