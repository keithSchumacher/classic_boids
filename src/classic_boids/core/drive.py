import numpy as np
from classic_boids.core.internal_state import InternalState
from .protocols import VectorType
from .vector import Vector, normalize
from .perception import Neighborhood


def separation_drive(
    neighborhood: Neighborhood, internal_state: InternalState
) -> VectorType:
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


def alignment_drive(
    neighborhood: Neighborhood, internal_state: InternalState
) -> VectorType:
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
