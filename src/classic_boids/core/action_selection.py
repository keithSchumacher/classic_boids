from dataclasses import replace
from .protocols import DriveName, InternalStateProtocol, VectorProtocol
from .vector import truncate


def action_selection(
    actions: dict[DriveName, VectorProtocol], internal_state: InternalStateProtocol
) -> InternalStateProtocol:
    """
    Compute the next state of the boid given the actions from each drive and
    update the boid's internal state (position and velocity).

    This function calculates a net force (weighted sum of the individual
    action drives), truncates it based on the maximum achievable force, applies
    it to the boid's velocity (considering the boid's mass), and then updates
    the boid's position based on the new velocity. Both the force-applied velocity
    and the resulting velocity are truncated by their respective maximum
    achievable limits.

    Parameters
    ----------
    actions : dict[DriveName, VectorProtocol]
        Dictionary of action values keyed by their drive type (e.g. separation, alignment, cohesion).
    internal_state : InternalStateProtocol
        The boid's internal state, containing current position, velocity, mass,
        and limits (max velocity, max force).

    Returns
    -------
    InternalStateProtocol
        The updated internal state with modified velocity and position.
    """
    # Compute the net force exerted on the boid by summing up weighted actions.
    # updated position and velocity assume system 'tick' is same time length as time unit of velocity
    net_force = truncate(
        actions[DriveName.SEPARATION] * internal_state.action_weights[DriveName.SEPARATION]
        + actions[DriveName.ALIGNMENT] * internal_state.action_weights[DriveName.ALIGNMENT]
        + actions[DriveName.COHESION] * internal_state.action_weights[DriveName.COHESION],
        internal_state.max_achievable_force,
    )
    updated_velocity = truncate(
        internal_state.velocity + (net_force / internal_state.mass), internal_state.max_achievable_velocity
    )
    # updated_position assumes system 'tick' is same time length as time unit of velocity
    updated_position = internal_state.position + updated_velocity
    return replace(internal_state, velocity=updated_velocity, position=updated_position)
