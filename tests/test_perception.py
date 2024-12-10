import pytest
import numpy as np
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.internal_state import InternalState, PerceptionAttributes
from classic_boids.core.perception import perception
from classic_boids.core.vector import Vector
from classic_boids.core.protocols import BoidID, Neighborhood


@pytest.fixture(scope="session")
def narrow_fov_boid():
    return InternalState(
        id=BoidID(0),
        position=Vector(np.array([0, 0])),
        velocity=Vector(np.array([1, 0])),
        perception_distance=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=5.0
        ),
        perception_field_of_view=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=0.0
        ),
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )


@pytest.fixture(scope="session")
def full_fov_boid():
    return InternalState(
        id=BoidID(1),
        position=Vector(np.array([1, 1])),
        velocity=Vector(np.array([0, 1])),
        perception_distance=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=15.0
        ),
        perception_field_of_view=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=2 * np.pi
        ),
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )


@pytest.fixture(scope="session")
def normal_fov_boid():
    return InternalState(
        id=BoidID(2),
        position=Vector(np.array([4, 4])),
        velocity=Vector(np.array([-1, 0])),
        perception_distance=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=5.0
        ),
        perception_field_of_view=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=np.pi / 1.5
        ),
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )


@pytest.fixture(scope="session")
def another_boid():
    return InternalState(
        id=BoidID(3),
        position=Vector(np.array([10, 10])),
        velocity=Vector(np.array([0, -1])),
        perception_distance=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=5.0
        ),
        perception_field_of_view=PerceptionAttributes(
            cohesion=0.0, alignment=0.0, separation=np.pi / 3
        ),
        mass=1.0,
        max_achievable_velocity=10.0,
        max_achievable_force=5.0,
    )


# Test fixture for the InputAlphabet
@pytest.fixture
def input_alphabet(narrow_fov_boid, full_fov_boid, normal_fov_boid, another_boid):
    positions = {
        BoidID(
            0
        ): narrow_fov_boid.position,  # Position of the boid itself in some tests
        BoidID(1): full_fov_boid.position,
        BoidID(2): normal_fov_boid.position,
        BoidID(3): another_boid.position,
    }
    velocities = {
        BoidID(
            0
        ): narrow_fov_boid.velocity,  # Position of the boid itself in some tests
        BoidID(1): full_fov_boid.velocity,
        BoidID(2): normal_fov_boid.velocity,
        BoidID(3): another_boid.velocity,
    }
    return InputAlphabet(positions=positions, velocities=velocities)


# Test for boid with very narrow field of view
def test_narrow_fov(input_alphabet, narrow_fov_boid):
    # Expected to see no other boids due to very narrow FOV
    assert perception(input_alphabet, narrow_fov_boid, "separation") == []


# Test for boid with 360-degree field of view
def test_full_fov(input_alphabet, full_fov_boid):
    # Expected to see all other boids except itself due to 360-degree FOV
    assert perception(input_alphabet, full_fov_boid, "separation") == Neighborhood(
        [BoidID(0), BoidID(2), BoidID(3)]
    )


# Test for boid with normal field of view
def test_normal_fov(input_alphabet, normal_fov_boid):
    # Expected to see one boid within a normal FOV and separation distance
    # 2 birds are within the fov, but only one is within distance
    assert perception(input_alphabet, normal_fov_boid, "separation") == Neighborhood(
        [BoidID(1)]
    )
    # extend seperation perception distance to reveal both boids
    normal_fov_boid.perception_distance = PerceptionAttributes(
        cohesion=0.0, alignment=0.0, separation=6.0
    )
    assert perception(input_alphabet, normal_fov_boid, "separation") == Neighborhood(
        [BoidID(0), BoidID(1)]
    )
