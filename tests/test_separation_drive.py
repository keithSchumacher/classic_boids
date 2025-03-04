import numpy as np
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, DriveName
from classic_boids.core.drive import separation_drive
from tests.test_utilities import vectors_close


def test_separation_drive_no_neighbors():
    # Boid at origin, no neighbors
    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(ids=[], info={})

    result = separation_drive(neighborhood, internal_state)
    # If normalize raises on zero vectors, this may fail. If it returns zero vector, check that:
    assert vectors_close(result, Vector(np.array([0.0, 0.0]))), "With no neighbors, separation drive should be zero."


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
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1)],
        info={BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 0.0])))},
    )

    result = separation_drive(neighborhood, internal_state)
    expected_direction = np.array([-1.0, -1.0]) / np.sqrt(2)  # normalized direction
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


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
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(2): (Vector(np.array([-1.0, 0.0])), Vector(np.array([0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    # Expect zero vector or very close to it
    assert vectors_close(result, Vector(np.array([0.0, 0.0]))), "Symmetric neighbors should cancel out."


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
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(2): (Vector(np.array([2.0, 0.0])), Vector(np.array([0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    # Compute expected:
    # For neighbor at (1,1), direction = (-1,-1), distance_sq = 2, contribution = (-0.5, -0.5)
    # For neighbor at (2,0), direction = (-2,0), distance_sq = 4, contribution = (-0.5, 0)
    # Sum = (-1, -0.5)
    # Normalized = (-1, -0.5) / sqrt(1.25) = (-0.894, -0.447)
    expected_direction = np.array([-1.0, -0.5]) / np.sqrt(1.25)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


# 3D Tests

def test_3d_separation_drive_no_neighbors():
    # Boid at origin, no neighbors
    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0, 0.0])),
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(ids=[], info={})

    result = separation_drive(neighborhood, internal_state)
    assert vectors_close(result, Vector(np.array([0.0, 0.0, 0.0]))), "With no neighbors, separation drive should be zero."
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_separation_drive_single_neighbor():
    # One neighbor at (1,1,1) and boid at (0,0,0).
    # direction = boid_pos - neighbor_pos = (0,0,0) - (1,1,1) = (-1,-1,-1)
    # distance_sq = (sqrt(1^2+1^2+1^2))^2 = 3
    # contribution = direction / distance_sq = (-1,-1,-1)/3 = (-0.333,-0.333,-0.333)
    # normalized result should be direction of (-1,-1,-1) but scaled to unit length
    # unit direction = (-1/-sqrt(3), -1/-sqrt(3), -1/-sqrt(3)) = (-0.577..., -0.577..., -0.577...)

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0, 0.0])),
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1)],
        info={BoidID(1): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([0.0, 0.0, 0.0])))},
    )

    result = separation_drive(neighborhood, internal_state)
    expected_direction = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3)  # normalized direction
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_separation_drive_symmetric_neighbors():
    # Boid at origin. Neighbors placed symmetrically around it should cancel out.
    # For example:
    # Neighbor A at (1,0,0), neighbor B at (-1,0,0)
    # direction A = (0,0,0)-(1,0,0)=(-1,0,0), distance_sq=1
    # direction B = (0,0,0)-(-1,0,0)=(1,0,0), distance_sq=1
    # summation = (-1,0,0)/1 + (1,0,0)/1 = (0,0,0)
    # normalized(0,0,0) = (0,0,0) [if allowed, otherwise error]

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0, 0.0])),
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0, 0.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(2): (Vector(np.array([-1.0, 0.0, 0.0])), Vector(np.array([0.0, 0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    # Expect zero vector or very close to it
    assert vectors_close(result, Vector(np.array([0.0, 0.0, 0.0]))), "Symmetric neighbors should cancel out."
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_separation_drive_asymmetric_neighbors():
    # Boid at origin. Three neighbors not symmetric:
    # Neighbor A at (1,1,1), direction = (-1,-1,-1)
    # Neighbor B at (2,0,0), direction = (-2,0,0)
    # Neighbor C at (0,0,2), direction = (0,0,-2)
    # For (-1,-1,-1), distance_sq=3; contribution = (-0.333,-0.333,-0.333)
    # For (-2,0,0), distance_sq=4; contribution = (-0.5,0,0)
    # For (0,0,-2), distance_sq=4; contribution = (0,0,-0.5)
    # Summation = (-0.333-0.5, -0.333, -0.333-0.5) = (-0.833, -0.333, -0.833)
    # norm = sqrt(0.833^2 + 0.333^2 + 0.833^2) = sqrt(1.4166) = ~1.19
    # normalized = (-0.833/1.19, -0.333/1.19, -0.833/1.19) ~ (-0.7, -0.28, -0.7)

    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0, 0.0])),
        perception_distance={DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 5.0},
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 2 * np.pi,
        },
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        },
    )

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3)],
        info={
            BoidID(1): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(2): (Vector(np.array([2.0, 0.0, 0.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(3): (Vector(np.array([0.0, 0.0, 2.0])), Vector(np.array([0.0, 0.0, 0.0]))),
        },
    )

    result = separation_drive(neighborhood, internal_state)
    # Compute expected:
    # For neighbor at (1,1,1), direction = (-1,-1,-1), distance_sq = 3, contribution = (-0.333, -0.333, -0.333)
    # For neighbor at (2,0,0), direction = (-2,0,0), distance_sq = 4, contribution = (-0.5, 0, 0)
    # For neighbor at (0,0,2), direction = (0,0,-2), distance_sq = 4, contribution = (0, 0, -0.5)
    # Sum = (-0.833, -0.333, -0.833)
    # Normalized = (-0.833, -0.333, -0.833) / sqrt(1.4166)
    sum_vector = np.array([-0.833, -0.333, -0.833])
    expected_direction = sum_vector / np.sqrt(np.sum(sum_vector**2))
    
    # Use np.isclose with a higher tolerance for floating point comparison
    assert np.allclose(result.data, expected_direction, rtol=1e-3, atol=1e-3), \
        f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"
