import numpy as np
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID
from classic_boids.core.drive import separation_drive
from tests.test_utilities import vectors_close


def test_separation_drive_no_neighbors():
    # Boid at origin, no neighbors
    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={"cohesion": 0.0, "alignment": 0.0, "separation": 5.0},
        perception_field_of_view={
            "cohesion": 0.0,
            "alignment": 0.0,
            "separation": np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )

    neighborhood = Neighborhood(ids=[], info={})

    result = separation_drive(neighborhood, internal_state)
    # If normalize raises on zero vectors, this may fail. If it returns zero vector, check that:
    assert vectors_close(
        result, Vector(np.array([0.0, 0.0]))
    ), "With no neighbors, separation drive should be zero."


def test_separation_drive_single_neighbor():
    # One neighbor at (1,1) and boid at (0,0).
    # direction = boid_pos - neighbor_pos = (0,0) - (1,1) = (-1,-1)
    # distance_sq = (sqrt(1^2+1^2))^2 = 2
    # contribution = direction / distance_sq = (-1,-1)/2 = (-0.5,-0.5)
    # normalized result should be direction of (-1,-1) but scaled to unit length
    # unit direction = (-1/-sqrt(2), -1/-sqrt(2)) = (-0.707..., -0.707...)

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={"cohesion": 0.0, "alignment": 0.0, "separation": 5.0},
        perception_field_of_view={
            "cohesion": 0.0,
            "alignment": 0.0,
            "separation": 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1)],
        info={BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 0.0])))},
    )

    result = separation_drive(neighborhood, internal_state)
    expected_direction = np.array([-1.0, -1.0]) / np.sqrt(2)  # normalized direction
    assert vectors_close(
        result, Vector(expected_direction)
    ), f"Expected {expected_direction} but got {result.data}"


def test_separation_drive_symmetric_neighbors():
    # Boid at origin. Neighbors placed symmetrically around it should cancel out.
    # For example:
    # Neighbor A at (1,0), neighbor B at (-1,0)
    # direction A = (0,0)-(1,0)=(-1,0), distance_sq=1
    # direction B = (0,0)-(-1,0)=(1,0), distance_sq=1
    # summation = (-1,0)/1 + (1,0)/1 = (0,0)
    # normalized(0,0) = (0,0) [if allowed, otherwise error]

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={"cohesion": 0.0, "alignment": 0.0, "separation": 5.0},
        perception_field_of_view={
            "cohesion": 0.0,
            "alignment": 0.0,
            "separation": 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(2): (Vector(np.array([-1.0, 0.0])), Vector(np.array([0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    # Expect zero vector again
    assert vectors_close(
        result, Vector(np.array([0.0, 0.0]))
    ), "Symmetric neighbors should cancel out."


def test_separation_drive_asymmetric_neighbors():
    # Boid at origin. Two neighbors not symmetric:
    # Neighbor A at (1,1), direction = (-1,-1)
    # Neighbor B at (2,0), direction = (-2,0)
    # Summation = (-1,-1)/(2) + (-2,0)/(4)
    # For (-1,-1), distance_sq=2; contribution = (-0.5,-0.5)
    # For (-2,0), distance_sq=4; contribution = (-0.5,0)
    # Summation = (-0.5-0.5, -0.5+0) = (-1.0,-0.5)
    # norm = sqrt(1 + 0.25)=sqrt(1.25)=~1.1180
    # normalized = (-1.0/1.1180, -0.5/1.1180) ~ (-0.8944, -0.4472)

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={"cohesion": 0.0, "alignment": 0.0, "separation": 5.0},
        perception_field_of_view={
            "cohesion": 0.0,
            "alignment": 0.0,
            "separation": 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(2): (Vector(np.array([2.0, 0.0])), Vector(np.array([0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    expected = np.array([-1.0, -0.5])
    norm = np.linalg.norm(expected)
    expected_normalized = expected / norm

    assert vectors_close(
        result, Vector(expected_normalized)
    ), f"Expected {expected_normalized} but got {result.data}"
