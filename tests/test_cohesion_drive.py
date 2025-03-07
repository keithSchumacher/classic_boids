import numpy as np
import pytest
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, DriveName
from classic_boids.core.drive import cohesion_drive
from tests.test_utilities import vectors_close


@pytest.fixture
def base_internal_state() -> InternalState:
    return InternalState(
        id=BoidID(0),
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),  # velocity isn't used by cohesion directly
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
        velocity=Vector(np.array([1.0, 0.0, 0.0])),  # velocity isn't used by cohesion directly
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


def test_cohesion_drive_no_neighbors(base_internal_state):
    # No neighbors means a zero vector result
    neighborhood = Neighborhood(ids=[], info={})
    result = cohesion_drive(neighborhood, base_internal_state)
    expected = Vector(np.zeros(2))
    assert vectors_close(result, expected), "No neighbors should yield a zero vector."


def test_cohesion_drive_single_neighbor(base_internal_state):
    # Boid at (0,0), single neighbor at (2,2).
    # average_position = (2,2)
    # direction = (2,2)-(0,0) = (2,2)
    # normalized = (2,2)/| (2,2)| = (2,2)/sqrt(8)= (0.707...,0.707...)

    neighbor_pos = Vector(np.array([2.0, 2.0]))
    neighborhood = Neighborhood(ids=[BoidID(1)], info={BoidID(1): (neighbor_pos, Vector(np.array([0.0, 0.0])))})

    result = cohesion_drive(neighborhood, base_internal_state)
    expected_direction = Vector(np.array([2.0, 2.0])) / np.sqrt(8)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


def test_cohesion_drive_multiple_neighbors_same_position(base_internal_state):
    # Boid at (0,0). Two neighbors both at (3,3).
    # average_position = (3,3)
    # direction = (3,3)-(0,0)=(3,3)
    # normalized = (3,3)/sqrt(18)= (3,3)/4.2426 ~ (0.707...,0.707...)

    neighbor_pos = Vector(np.array([3.0, 3.0]))
    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (neighbor_pos, Vector(np.array([0.0, 0.0]))),
            BoidID(2): (neighbor_pos, Vector(np.array([0.0, 0.0]))),
        },
    )

    result = cohesion_drive(neighborhood, base_internal_state)
    expected_direction = Vector(np.array([3.0, 3.0])) / np.sqrt(18)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


def test_cohesion_drive_multiple_neighbors_diverse_positions(base_internal_state):
    # Boid at (0,0).
    # Neighbors at: (2,0), (0,2), and (2,2)
    # sum = (2,0)+(0,2)+(2,2)=(4,4)
    # average=(4,4)/3=(1.333...,1.333...)
    # direction=(1.333...,1.333...)-(0,0)=(1.333...,1.333...)
    # norm = sqrt(1.333^2+1.333^2)=sqrt(2*(1.333^2))=1.333*sqrt(2)=~1.8868
    # normalized= (1.333...,1.333...)/1.8868 ~ (0.707...,0.707...)

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3)],
        info={
            BoidID(1): (Vector(np.array([2.0, 0.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 2.0])), Vector(np.array([0.0, 0.0]))),
            BoidID(3): (Vector(np.array([2.0, 2.0])), Vector(np.array([0.0, 0.0]))),
        },
    )

    result = cohesion_drive(neighborhood, base_internal_state)
    # Compute expected:
    avg = Vector(np.array([4.0, 4.0])) / 3.0  # (1.3333,1.3333)
    norm_val = avg.norm()
    expected_direction = avg / norm_val
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"


# 3D Tests

def test_3d_cohesion_drive_no_neighbors(base_internal_state_3d):
    # No neighbors means a zero vector result
    neighborhood = Neighborhood(ids=[], info={})
    result = cohesion_drive(neighborhood, base_internal_state_3d)
    expected = Vector(np.zeros(3))
    assert vectors_close(result, expected), "No neighbors should yield a zero vector."
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_cohesion_drive_single_neighbor(base_internal_state_3d):
    # Boid at (0,0,0), single neighbor at (2,2,2).
    # average_position = (2,2,2)
    # direction = (2,2,2)-(0,0,0) = (2,2,2)
    # normalized = (2,2,2)/| (2,2,2)| = (2,2,2)/sqrt(12)= (0.577...,0.577...,0.577...)

    neighbor_pos = Vector(np.array([2.0, 2.0, 2.0]))
    neighborhood = Neighborhood(ids=[BoidID(1)], info={BoidID(1): (neighbor_pos, Vector(np.array([0.0, 0.0, 0.0])))})

    result = cohesion_drive(neighborhood, base_internal_state_3d)
    expected_direction = Vector(np.array([2.0, 2.0, 2.0])) / np.sqrt(12)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_cohesion_drive_multiple_neighbors_same_position(base_internal_state_3d):
    # Boid at (0,0,0). Two neighbors both at (3,3,3).
    # average_position = (3,3,3)
    # direction = (3,3,3)-(0,0,0)=(3,3,3)
    # normalized = (3,3,3)/sqrt(27)= (3,3,3)/5.196 ~ (0.577...,0.577...,0.577...)

    neighbor_pos = Vector(np.array([3.0, 3.0, 3.0]))
    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2)],
        info={
            BoidID(1): (neighbor_pos, Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(2): (neighbor_pos, Vector(np.array([0.0, 0.0, 0.0]))),
        },
    )

    result = cohesion_drive(neighborhood, base_internal_state_3d)
    expected_direction = Vector(np.array([3.0, 3.0, 3.0])) / np.sqrt(27)
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"


def test_3d_cohesion_drive_multiple_neighbors_diverse_positions(base_internal_state_3d):
    # Boid at (0,0,0).
    # Neighbors at: (2,0,0), (0,2,0), (0,0,2), and (2,2,2)
    # sum = (2,0,0)+(0,2,0)+(0,0,2)+(2,2,2)=(4,4,4)
    # average=(4,4,4)/4=(1,1,1)
    # direction=(1,1,1)-(0,0,0)=(1,1,1)
    # norm = sqrt(1^2+1^2+1^2)=sqrt(3)=~1.732
    # normalized= (1,1,1)/1.732 ~ (0.577...,0.577...,0.577...)

    neighborhood = Neighborhood(
        ids=[BoidID(1), BoidID(2), BoidID(3), BoidID(4)],
        info={
            BoidID(1): (Vector(np.array([2.0, 0.0, 0.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(2): (Vector(np.array([0.0, 2.0, 0.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(3): (Vector(np.array([0.0, 0.0, 2.0])), Vector(np.array([0.0, 0.0, 0.0]))),
            BoidID(4): (Vector(np.array([2.0, 2.0, 2.0])), Vector(np.array([0.0, 0.0, 0.0]))),
        },
    )

    result = cohesion_drive(neighborhood, base_internal_state_3d)
    # Compute expected:
    avg = Vector(np.array([4.0, 4.0, 4.0])) / 4.0  # (1,1,1)
    norm_val = avg.norm()
    expected_direction = avg / norm_val
    assert vectors_close(result, Vector(expected_direction)), f"Expected {expected_direction} but got {result.data}"
    assert len(result) == 3, "Result should be a 3D vector"
