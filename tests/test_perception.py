import pytest
import numpy as np
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import compute_perceptions, perception, Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, DriveName


@pytest.fixture(scope="session")
def narrow_fov_boid():
    return InternalState(
        id=BoidID(0),
        position=Vector(np.array([0, 0])),
        velocity=Vector(np.array([1, 0])),
        perception_distance={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 5.0,
        },
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 0.0,
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


@pytest.fixture(scope="session")
def full_fov_boid():
    return InternalState(
        id=BoidID(1),
        position=Vector(np.array([1, 1])),
        velocity=Vector(np.array([0, 1])),
        perception_distance={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 15.0,
        },
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


@pytest.fixture(scope="session")
def normal_fov_boid():
    return InternalState(
        id=BoidID(2),
        position=Vector(np.array([4, 4])),
        velocity=Vector(np.array([-1, 0])),
        perception_distance={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 5.0,
        },
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: np.pi / 1.5,
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


@pytest.fixture(scope="session")
def another_boid():
    return InternalState(
        id=BoidID(3),
        position=Vector(np.array([10, 10])),
        velocity=Vector(np.array([0, -1])),
        perception_distance={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: 5.0,
        },
        perception_field_of_view={
            DriveName.COHESION: 0.0,
            DriveName.ALIGNMENT: 0.0,
            DriveName.SEPARATION: np.pi / 3,
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


# Test fixture for the InputAlphabet
@pytest.fixture
def input_alphabet(narrow_fov_boid, full_fov_boid, normal_fov_boid, another_boid):
    positions = {
        BoidID(0): narrow_fov_boid.position,
        BoidID(1): full_fov_boid.position,
        BoidID(2): normal_fov_boid.position,
        BoidID(3): another_boid.position,
    }
    velocities = {
        BoidID(0): narrow_fov_boid.velocity,
        BoidID(1): full_fov_boid.velocity,
        BoidID(2): normal_fov_boid.velocity,
        BoidID(3): another_boid.velocity,
    }
    return InputAlphabet(positions=positions, velocities=velocities)


# Test for boid with very narrow field of view
def test_narrow_fov(input_alphabet, narrow_fov_boid):
    # Expected to see no other boids due to very narrow FOV
    expected_neighborhood = Neighborhood(ids=[], info={})
    assert perception(input_alphabet, narrow_fov_boid, DriveName.SEPARATION) == expected_neighborhood


# Test for boid with 360-degree field of view
def test_full_fov(input_alphabet, full_fov_boid):
    # Expected to see all other boids except itself due to 360-degree FOV
    expected_ids = [BoidID(0), BoidID(2), BoidID(3)]
    expected_info = {
        boid_id: (input_alphabet.positions[boid_id], input_alphabet.velocities[boid_id]) for boid_id in expected_ids
    }

    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)

    assert perception(input_alphabet, full_fov_boid, DriveName.SEPARATION) == expected_neighborhood


# Test for boid with normal field of view
def test_normal_fov(input_alphabet, normal_fov_boid):
    # Initially, normal_fov_boid can only see boid 1 within both distance and FOV.
    expected_ids = [BoidID(1)]
    expected_info = {
        boid_id: (input_alphabet.positions[boid_id], input_alphabet.velocities[boid_id]) for boid_id in expected_ids
    }
    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)
    assert perception(input_alphabet, normal_fov_boid, DriveName.SEPARATION) == expected_neighborhood

    # Now extend the separation distance so it can also see boid 0.
    normal_fov_boid.perception_distance = {DriveName.COHESION: 0.0, DriveName.ALIGNMENT: 0.0, DriveName.SEPARATION: 6.0}
    expected_ids = [BoidID(0), BoidID(1)]
    expected_info = {
        boid_id: (input_alphabet.positions[boid_id], input_alphabet.velocities[boid_id]) for boid_id in expected_ids
    }
    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)
    assert perception(input_alphabet, normal_fov_boid, DriveName.SEPARATION) == expected_neighborhood


@pytest.fixture
def mock_perception_separation():
    """
    Example mock for separation perception function.
    Returns a Neighborhood with no neighbors for demonstration.
    """

    def _mock(input_alphabet, internal_state, drive_name):
        return Neighborhood(ids=[], info={})

    return _mock


@pytest.fixture
def mock_perception_alignment():
    """
    Example mock for alignment perception function.
    Returns a Neighborhood with a single neighbor for demonstration.
    """

    def _mock(input_alphabet, internal_state, drive_name):
        boid_id = BoidID(999)
        return Neighborhood(ids=[boid_id], info={boid_id: ("dummy_pos", "dummy_vel")})

    return _mock


@pytest.fixture
def mock_perception_cohesion():
    """
    Example mock for cohesion perception function.
    Returns a Neighborhood with two neighbors for demonstration.
    """

    def _mock(input_alphabet, internal_state, drive_name):
        boid_id_1 = BoidID(1000)
        boid_id_2 = BoidID(1001)
        return Neighborhood(
            ids=[boid_id_1, boid_id_2],
            info={
                boid_id_1: ("cohesion_pos_1", "cohesion_vel_1"),
                boid_id_2: ("cohesion_pos_2", "cohesion_vel_2"),
            },
        )

    return _mock


@pytest.fixture
def perception_functions(mock_perception_separation, mock_perception_alignment, mock_perception_cohesion):
    """
    Creates a dictionary mapping each DriveName to its mock perception function.
    """
    return {
        DriveName.SEPARATION: mock_perception_separation,
        DriveName.ALIGNMENT: mock_perception_alignment,
        DriveName.COHESION: mock_perception_cohesion,
    }


def test_compute_perceptions_with_mocks(perception_functions, input_alphabet, narrow_fov_boid):
    """
    Test that compute_perceptions correctly calls each drive function and
    returns the expected dictionary of Neighborhood objects.
    """
    result = compute_perceptions(perception_functions, input_alphabet, narrow_fov_boid)

    # Check keys
    assert set(result.keys()) == {DriveName.SEPARATION, DriveName.ALIGNMENT, DriveName.COHESION}

    # Check that the result for SEPARATION is what mock_perception_separation returns
    assert result[DriveName.SEPARATION].ids == []
    assert result[DriveName.SEPARATION].info == {}

    # Check that the result for ALIGNMENT is what mock_perception_alignment returns
    assert len(result[DriveName.ALIGNMENT].ids) == 1
    assert list(result[DriveName.ALIGNMENT].ids)[0] == BoidID(999)
    assert "dummy_pos" in result[DriveName.ALIGNMENT].info[BoidID(999)]

    # Check that the result for COHESION is what mock_perception_cohesion returns
    assert len(result[DriveName.COHESION].ids) == 2
    assert BoidID(1000) in result[DriveName.COHESION].info
    assert BoidID(1001) in result[DriveName.COHESION].info


def test_compute_perceptions_with_real_perception_functions(perception_functions, input_alphabet, full_fov_boid):
    perception_functions = {
        DriveName.SEPARATION: perception,
        DriveName.ALIGNMENT: perception,
        DriveName.COHESION: perception,
    }
    result = compute_perceptions(perception_functions, input_alphabet, full_fov_boid)
    assert DriveName.SEPARATION in result
    assert DriveName.ALIGNMENT in result
    assert DriveName.COHESION in result

    # test actual perception function results
    expected_ids = [BoidID(0), BoidID(2), BoidID(3)]
    expected_info = {
        boid_id: (input_alphabet.positions[boid_id], input_alphabet.velocities[boid_id]) for boid_id in expected_ids
    }
    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)

    assert result[DriveName.SEPARATION] == expected_neighborhood
