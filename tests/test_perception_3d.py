import pytest
import numpy as np
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import compute_perceptions, perception, Neighborhood
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, DriveName


@pytest.fixture(scope="session")
def narrow_fov_boid_3d():
    return InternalState(
        id=BoidID(0),
        position=Vector(np.array([0, 0, 0])),
        velocity=Vector(np.array([1, 0, 0])),
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
def full_fov_boid_3d():
    return InternalState(
        id=BoidID(1),
        position=Vector(np.array([1, 1, 1])),
        velocity=Vector(np.array([0, 1, 0])),
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
def normal_fov_boid_3d():
    return InternalState(
        id=BoidID(2),
        position=Vector(np.array([4, 4, 4])),
        velocity=Vector(np.array([-1, 0, 0])),
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
def another_boid_3d():
    return InternalState(
        id=BoidID(3),
        position=Vector(np.array([10, 10, 10])),
        velocity=Vector(np.array([0, -1, 0])),
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
def input_alphabet_3d(narrow_fov_boid_3d, full_fov_boid_3d, normal_fov_boid_3d, another_boid_3d):
    positions = {
        BoidID(0): narrow_fov_boid_3d.position,
        BoidID(1): full_fov_boid_3d.position,
        BoidID(2): normal_fov_boid_3d.position,
        BoidID(3): another_boid_3d.position,
    }
    velocities = {
        BoidID(0): narrow_fov_boid_3d.velocity,
        BoidID(1): full_fov_boid_3d.velocity,
        BoidID(2): normal_fov_boid_3d.velocity,
        BoidID(3): another_boid_3d.velocity,
    }
    return InputAlphabet(positions=positions, velocities=velocities)


# Test for boid with very narrow field of view in 3D
def test_narrow_fov_3d(input_alphabet_3d, narrow_fov_boid_3d):
    # Expected to see no other boids due to very narrow FOV
    expected_neighborhood = Neighborhood(ids=[], info={})
    assert perception(input_alphabet_3d, narrow_fov_boid_3d, DriveName.SEPARATION) == expected_neighborhood


# Test for boid with 360-degree field of view in 3D
def test_full_fov_3d(input_alphabet_3d, full_fov_boid_3d):
    # Expected to see all other boids except itself due to 360-degree FOV
    expected_ids = [BoidID(0), BoidID(2), BoidID(3)]
    expected_info = {
        boid_id: (input_alphabet_3d.positions[boid_id], input_alphabet_3d.velocities[boid_id])
        for boid_id in expected_ids
    }

    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)

    assert perception(input_alphabet_3d, full_fov_boid_3d, DriveName.SEPARATION) == expected_neighborhood


# Test for boid with normal field of view in 3D
def test_normal_fov_3d(input_alphabet_3d, normal_fov_boid_3d):
    # Initially, normal_fov_boid_3d can only see boid 1 within both distance and FOV.
    expected_ids = [BoidID(1)]
    expected_info = {
        boid_id: (input_alphabet_3d.positions[boid_id], input_alphabet_3d.velocities[boid_id])
        for boid_id in expected_ids
    }
    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)
    assert perception(input_alphabet_3d, normal_fov_boid_3d, DriveName.SEPARATION) == expected_neighborhood

    # Now extend the separation distance so it can also see boid 0.
    normal_fov_boid_3d.perception_distance = {
        DriveName.COHESION: 0.0,
        DriveName.ALIGNMENT: 0.0,
        DriveName.SEPARATION: 7.0,
    }
    expected_ids = [BoidID(0), BoidID(1)]
    expected_info = {
        boid_id: (input_alphabet_3d.positions[boid_id], input_alphabet_3d.velocities[boid_id])
        for boid_id in expected_ids
    }
    expected_neighborhood = Neighborhood(ids=expected_ids, info=expected_info)
    assert perception(input_alphabet_3d, normal_fov_boid_3d, DriveName.SEPARATION) == expected_neighborhood


@pytest.fixture
def mock_perception_separation_3d():
    """
    Example mock for separation perception function in 3D.
    Returns a Neighborhood with no neighbors for demonstration.
    """

    def _mock(input_alphabet, internal_state, drive_name):
        return Neighborhood(ids=[], info={})

    return _mock


@pytest.fixture
def mock_perception_alignment_3d():
    """
    Example mock for alignment perception function in 3D.
    Returns a Neighborhood with a single neighbor for demonstration.
    """

    def _mock(input_alphabet, internal_state, drive_name):
        boid_id = BoidID(999)
        return Neighborhood(ids=[boid_id], info={boid_id: ("dummy_pos", "dummy_vel")})

    return _mock


@pytest.fixture
def mock_perception_cohesion_3d():
    """
    Example mock for cohesion perception function in 3D.
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
def perception_functions_3d(mock_perception_separation_3d, mock_perception_alignment_3d, mock_perception_cohesion_3d):
    """
    Creates a dictionary mapping each DriveName to its mock perception function for 3D.
    """
    return {
        DriveName.SEPARATION: mock_perception_separation_3d,
        DriveName.ALIGNMENT: mock_perception_alignment_3d,
        DriveName.COHESION: mock_perception_cohesion_3d,
    }


def test_compute_perceptions_with_mocks_3d(perception_functions_3d, input_alphabet_3d, narrow_fov_boid_3d):
    """
    Test that compute_perceptions correctly calls each drive function and
    returns the expected dictionary of Neighborhood objects in 3D.
    """
    result = compute_perceptions(perception_functions_3d, input_alphabet_3d, narrow_fov_boid_3d)

    # Check keys
    assert set(result.keys()) == {DriveName.SEPARATION, DriveName.ALIGNMENT, DriveName.COHESION}

    # Check that the result for SEPARATION is what mock_perception_separation_3d returns
    assert result[DriveName.SEPARATION].ids == []
    assert result[DriveName.SEPARATION].info == {}

    # Check that the result for ALIGNMENT is what mock_perception_alignment_3d returns
    assert result[DriveName.ALIGNMENT].ids == [BoidID(999)]
    assert result[DriveName.ALIGNMENT].info == {BoidID(999): ("dummy_pos", "dummy_vel")}

    # Check that the result for COHESION is what mock_perception_cohesion_3d returns
    assert result[DriveName.COHESION].ids == [BoidID(1000), BoidID(1001)]
    assert result[DriveName.COHESION].info == {
        BoidID(1000): ("cohesion_pos_1", "cohesion_vel_1"),
        BoidID(1001): ("cohesion_pos_2", "cohesion_vel_2"),
    }


# Test with real perception functions in 3D
def test_compute_perceptions_with_real_perception_functions_3d(input_alphabet_3d, full_fov_boid_3d):
    """
    Test that compute_perceptions works with the real perception function in 3D.
    """
    # Create a dictionary of real perception functions
    real_perception_functions = {
        DriveName.SEPARATION: perception,
        DriveName.ALIGNMENT: perception,
        DriveName.COHESION: perception,
    }

    # Compute perceptions
    result = compute_perceptions(real_perception_functions, input_alphabet_3d, full_fov_boid_3d)

    # Check keys
    assert set(result.keys()) == {DriveName.SEPARATION, DriveName.ALIGNMENT, DriveName.COHESION}

    # For this test, we expect all perceptions to be empty because we set the perception
    # distances for COHESION and ALIGNMENT to 0.0 in the fixture
    assert len(result[DriveName.COHESION].ids) == 0
    assert len(result[DriveName.ALIGNMENT].ids) == 0

    # For SEPARATION, we expect to see all other boids because we set the perception
    # distance to 15.0 and the FOV to 2*pi in the fixture
    assert set(result[DriveName.SEPARATION].ids) == {BoidID(0), BoidID(2), BoidID(3)}


# Test 3D specific perception scenarios
def test_3d_specific_perception():
    """
    Test perception in 3D-specific scenarios, like boids at different heights.
    """
    # Create boids at different heights
    boid_below = InternalState(
        id=BoidID(0),
        position=Vector(np.array([0, 0, 0])),
        velocity=Vector(np.array([0, 0, 1])),  # Looking up
        perception_distance={DriveName.SEPARATION: 10.0},
        perception_field_of_view={DriveName.SEPARATION: np.pi / 2},  # 90 degrees
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={DriveName.SEPARATION: 1.0},
    )

    boid_above = InternalState(
        id=BoidID(1),
        position=Vector(np.array([0, 0, 5])),  # 5 units above
        velocity=Vector(np.array([1, 0, 0])),  # Looking horizontally
        perception_distance={DriveName.SEPARATION: 10.0},
        perception_field_of_view={DriveName.SEPARATION: np.pi / 2},  # 90 degrees
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
        action_weights={DriveName.SEPARATION: 1.0},
    )

    # Create input alphabet
    positions = {
        BoidID(0): boid_below.position,
        BoidID(1): boid_above.position,
    }
    velocities = {
        BoidID(0): boid_below.velocity,
        BoidID(1): boid_above.velocity,
    }
    input_alphabet = InputAlphabet(positions=positions, velocities=velocities)

    # Test that boid_below can see boid_above (since it's looking up)
    result = perception(input_alphabet, boid_below, DriveName.SEPARATION)
    assert BoidID(1) in result.ids

    # Test that boid_above cannot see boid_below (since it's looking horizontally)
    result = perception(input_alphabet, boid_above, DriveName.SEPARATION)
    assert BoidID(0) not in result.ids

    # Now change boid_above to look down
    boid_above.velocity = Vector(np.array([0, 0, -1]))
    velocities[BoidID(1)] = boid_above.velocity
    input_alphabet = InputAlphabet(positions=positions, velocities=velocities)

    # Test that boid_above can now see boid_below
    result = perception(input_alphabet, boid_above, DriveName.SEPARATION)
    assert BoidID(0) in result.ids
