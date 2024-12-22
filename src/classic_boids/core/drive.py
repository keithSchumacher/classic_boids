import numpy as np
from .protocols import DriveFunctionProtocol, DriveName, InternalStateProtocol, NeighborhoodProtocol, VectorType
from .vector import Vector, normalize


def separation_drive(neighborhood: NeighborhoodProtocol, internal_state: InternalStateProtocol) -> VectorType:
    position = internal_state.position
    summation = Vector(np.zeros_like(position.data))

    # Early return if no neighbors
    if not neighborhood.ids:
        return summation

    for _, (neighbor_position, _) in neighborhood.info.items():
        # direction: vector pointing from neighbor to current boid
        direction = position - neighbor_position
        # distance_sq: squared distance between boid and neighbor
        distance_sq = direction.norm() ** 2

        # Add contribution to summation
        summation += direction / distance_sq

    # If after summation the vector is still zero, return zero
    if summation.norm() == 0:
        return Vector(np.zeros_like(position.data))

    # Otherwise, return the normalized summation
    return normalize(summation)


def alignment_drive(neighborhood: NeighborhoodProtocol, internal_state: InternalStateProtocol) -> VectorType:
    """Compute the alignment drive for a boid based on its neighbors' velocities.

    If the boid has no neighbors, the alignment drive is zero.
    Otherwise, the drive points in the direction of the average neighbor velocity
    relative to the boid's current velocity.
    """
    velocity = internal_state.velocity
    summation = Vector(np.zeros_like(velocity.data))

    # Early return if no neighbors
    if not neighborhood.ids:
        return summation

    # Sum all neighbors' velocities
    for _, (_, neighbor_velocity) in neighborhood.info.items():
        summation += neighbor_velocity

    # Compute average of neighbors' velocities
    average_velocity = summation / len(neighborhood.ids)

    # Return the normalized vector pointing from the current velocity toward the average velocity
    return normalize(average_velocity - velocity)


def cohesion_drive(neighborhood: NeighborhoodProtocol, internal_state: InternalStateProtocol) -> VectorType:
    """Compute the cohesion drive for a boid based on its neighbors' positions.

    If the boid has no neighbors, the cohesion drive is zero.
    Otherwise, the drive points in the direction of the average neighbor position
    relative to the boid's current position.
    """
    position = internal_state.position
    summation = Vector(np.zeros_like(position.data))

    # Early return if no neighbors
    if not neighborhood.ids:
        return summation

    # Sum all neighbors' positions
    for _, (neighbor_position, _) in neighborhood.info.items():
        summation += neighbor_position

    # Compute average of neighbors' positions
    average_position = summation / len(neighborhood.ids)

    # Return the normalized vector pointing from the current position toward the average position
    return normalize(average_position - position)


def compute_drives(
    drive_functions: dict[DriveName, DriveFunctionProtocol],
    neighborhood: dict[DriveName, NeighborhoodProtocol],
    internal_state: InternalStateProtocol,
) -> dict[DriveName, VectorType]:
    """Compute the actions by calling each drive function and return them in a dictionary."""
    a_s = drive_functions[DriveName.SEPARATION](neighborhood[DriveName.SEPARATION], internal_state)
    a_a = drive_functions[DriveName.ALIGNMENT](neighborhood[DriveName.COHESION], internal_state)
    a_c = drive_functions[DriveName.COHESION](neighborhood[DriveName.COHESION], internal_state)
    return {
        DriveName.SEPARATION: a_s,
        DriveName.ALIGNMENT: a_a,
        DriveName.COHESION: a_c,
    }
