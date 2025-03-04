import numpy as np
import pytest
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, DriveName
from classic_boids.core.drive import alignment_drive
from tests.test_utilities import vectors_close


@pytest.fixture
def base_internal_state() -> InternalState:
    return InternalState(
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


@pytest.fixture
def base_internal_state_3d() -> InternalState:
    return InternalState(
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


@pytest.fixture
def no_neighbors() -> Neighborhood:
    return Neighborhood(ids=[], info={})


@pytest.fixture
def single_neighbor() -> Neighborhood:
    # Boid velocity: (1,0)
    # Neighbor velocity: (0,1)
    # Expected: direction ~ (-0.707..., 0.707...)
    return Neighborhood(
        ids=[BoidID(1)],
        info={BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 1.0])))},
    )


@pytest.fixture
def multiple_neighbors_same_velocity() -> Neighborhood:
    # Neighbors both have velocity (0,2)
    return Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 1.0])), Vector(np.array([0.0, 2.0]))),
            BoidID(2): (Vector(np.array([2.0, 2.0])), Vector(np.array([0.0, 2.0]))),
        },
    )


@pytest.fixture
def multiple_neighbors_diverse_velocities() -> Neighborhood:
    # Neighbors: v1=(2,0), v2=(0,2), v3=(2,2)
    # Boid velocity: (1,1)
    return Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0])), Vector(np.array([2.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 1.0])), Vector(np.array([0.0, 2.0]))),
            BoidID(3): (Vector(np.array([1.0, 1.0])), Vector(np.array([2.0, 2.0]))),
        },
    )


# 3D Fixtures
@pytest.fixture
def no_neighbors_3d() -> Neighborhood:
    return Neighborhood(ids=[], info={})


@pytest.fixture
def single_neighbor_3d() -> Neighborhood:
    # Boid velocity: (1,0,0)
    # Neighbor velocity: (0,1,0)
    # Expected: direction ~ (-0.707..., 0.707..., 0)
    return Neighborhood(
        ids=[BoidID(1)],
        info={BoidID(1): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([0.0, 1.0, 0.0])))},
    )


@pytest.fixture
def multiple_neighbors_same_velocity_3d() -> Neighborhood:
    # Neighbors both have velocity (0,2,0)
    return Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([0.0, 2.0, 0.0]))),
            BoidID(2): (Vector(np.array([2.0, 2.0, 2.0])), Vector(np.array([0.0, 2.0, 0.0]))),
        },
    )


@pytest.fixture
def multiple_neighbors_diverse_velocities_3d() -> Neighborhood:
    # Neighbors: v1=(2,0,0), v2=(0,2,0), v3=(0,0,2), v4=(1,1,1)
    # Boid velocity: (1,0,0)
    return Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3), BoidID(4)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0, 0.0])), Vector(np.array([2.0, 0.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 1.0, 0.0])), Vector(np.array([0.0, 2.0, 0.0]))),
            BoidID(3): (Vector(np.array([0.0, 0.0, 1.0])), Vector(np.array([0.0, 0.0, 2.0]))),
            BoidID(4): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([1.0, 1.0, 1.0]))),
        },
    )


def test_alignment_drive_no_neighbors(base_internal_state, no_neighbors):
    result = alignment_drive(no_neighbors, base_internal_state)
    expected = Vector(np.zeros(2))
    assert vectors_close(result, expected), "No neighbors should return a zero vector."


def test_alignment_drive_single_neighbor(base_internal_state, single_neighbor):
    result = alignment_drive(single_neighbor, base_internal_state)
    expected_direction = np.array([-1.0, 1.0]) / np.sqrt(2)  # from earlier calculation
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


def test_alignment_drive_multiple_neighbors_same_velocity(base_internal_state, multiple_neighbors_same_velocity):
    result = alignment_drive(multiple_neighbors_same_velocity, base_internal_state)
    # from earlier calculation: expected = (-1,2)/sqrt(5)
    expected = np.array([-1.0, 2.0]) / np.sqrt(5)
    assert vectors_close(result, Vector(expected)), f"Expected {expected} but got {result.data}"


def test_alignment_drive_multiple_neighbors_diverse_velocities():
    internal_state = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 1.0])),
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

    # Create the neighborhood directly instead of using the fixture
    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0])), Vector(np.array([2.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 1.0])), Vector(np.array([0.0, 2.0]))),
            BoidID(3): (Vector(np.array([1.0, 1.0])), Vector(np.array([2.0, 2.0]))),
        },
    )

    result = alignment_drive(neighborhood, internal_state)
    # Average velocity = (2+0+2, 0+2+2)/3 = (4,4)/3 = (1.33, 1.33)
    # Boid velocity = (1,1)
    # Difference = (1.33-1, 1.33-1) = (0.33, 0.33)
    # Normalized = (0.33, 0.33)/sqrt(0.33^2 + 0.33^2) = (0.33, 0.33)/0.47 = (0.707, 0.707)
    expected = np.array([0.33, 0.33]) / np.sqrt(0.33**2 + 0.33**2)
    assert vectors_close(result, Vector(expected)), f"Expected {expected} but got {result.data}"


# 3D Tests
def test_3d_alignment_drive_no_neighbors(base_internal_state_3d, no_neighbors_3d):
    result = alignment_drive(no_neighbors_3d, base_internal_state_3d)
    expected = Vector(np.zeros(3))
    assert vectors_close(result, expected), "No neighbors should return a zero vector."
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_alignment_drive_single_neighbor(base_internal_state_3d, single_neighbor_3d):
    result = alignment_drive(single_neighbor_3d, base_internal_state_3d)
    # Boid velocity: (1,0,0)
    # Neighbor velocity: (0,1,0)
    # Difference: (-1,1,0)
    # Normalized: (-1,1,0)/sqrt(2) = (-0.707..., 0.707..., 0)
    expected_direction = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_alignment_drive_multiple_neighbors_same_velocity(base_internal_state_3d, multiple_neighbors_same_velocity_3d):
    result = alignment_drive(multiple_neighbors_same_velocity_3d, base_internal_state_3d)
    # Boid velocity: (1,0,0)
    # Neighbors both have velocity (0,2,0)
    # Average velocity: (0,2,0)
    # Difference: (-1,2,0)
    # Normalized: (-1,2,0)/sqrt(5) = (-0.447..., 0.894..., 0)
    expected = np.array([-1.0, 2.0, 0.0]) / np.sqrt(5)
    assert vectors_close(result, Vector(expected)), f"Expected {expected} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_alignment_drive_multiple_neighbors_diverse_velocities():
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

    # Create the neighborhood directly instead of using the fixture
    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3), BoidID(4)],
        info={
            BoidID(1): (Vector(np.array([1.0, 0.0, 0.0])), Vector(np.array([2.0, 0.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 1.0, 0.0])), Vector(np.array([0.0, 2.0, 0.0]))),
            BoidID(3): (Vector(np.array([0.0, 0.0, 1.0])), Vector(np.array([0.0, 0.0, 2.0]))),
            BoidID(4): (Vector(np.array([1.0, 1.0, 1.0])), Vector(np.array([1.0, 1.0, 1.0]))),
        },
    )

    result = alignment_drive(neighborhood, internal_state)
    # Average velocity = (2+0+0+1, 0+2+0+1, 0+0+2+1)/4 = (3,3,3)/4 = (0.75, 0.75, 0.75)
    # Boid velocity = (1,0,0)
    # Difference = (0.75-1, 0.75-0, 0.75-0) = (-0.25, 0.75, 0.75)
    # Normalized = (-0.25, 0.75, 0.75)/sqrt(0.25^2 + 0.75^2 + 0.75^2) = (-0.25, 0.75, 0.75)/1.09 = (-0.229, 0.688, 0.688)
    expected = np.array([-0.25, 0.75, 0.75]) / np.sqrt(0.25**2 + 0.75**2 + 0.75**2)
    assert vectors_close(result, Vector(expected)), f"Expected {expected} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"
