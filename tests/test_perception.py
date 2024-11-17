import pytest
import numpy as np
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.internal_state import InternalState, PerceptionAttributes
from classic_boids.core.perception import separation_perception
from classic_boids.core.vector import Vector


# @pytest.fixture(scope="session")
# def perception_distance():
#     return PerceptionAttributes(
#             separation=5.0,
#             alignment=10.0,
#             cohesion=15.0
#         )


# @pytest.fixture(scope="session")
# def perception_field_of_view():
#     return PerceptionAttributes(
#             separation=np.pi/2,
#             alignment=2*np.pi/3,
#             cohesion=np.pi
#         )


@pytest.fixture(scope="session")
def narrow_fov_boid():
    return InternalState(
        id=0,
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
        id=1,
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
        id=2,
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
        id=3,
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
        0: narrow_fov_boid.position,  # Position of the boid itself in some tests
        1: full_fov_boid.position,
        2: normal_fov_boid.position,
        3: another_boid.position,
    }
    velocities = {
        0: narrow_fov_boid.velocity,  # Position of the boid itself in some tests
        1: full_fov_boid.velocity,
        2: normal_fov_boid.velocity,
        3: another_boid.velocity,
    }
    return InputAlphabet(positions=positions, velocities=velocities)


# Test for boid with very narrow field of view
def test_narrow_fov(input_alphabet, narrow_fov_boid):
    # Expected to see no other boids due to very narrow FOV
    assert separation_perception(input_alphabet, narrow_fov_boid) == []


# Test for boid with 360-degree field of view
def test_full_fov(input_alphabet, full_fov_boid):
    # Expected to see all other boids except itself due to 360-degree FOV
    assert separation_perception(input_alphabet, full_fov_boid) == [0, 2, 3]


# Test for boid with normal field of view
def test_normal_fov(input_alphabet, normal_fov_boid):
    # Expected to see one boid within a normal FOV and separation distance
    # 2 birds are within the fov, but only one is within distance
    assert separation_perception(input_alphabet, normal_fov_boid) == [1]
    # extend seperation perception distance to reveal both boids
    normal_fov_boid.perception_distance = PerceptionAttributes(
        cohesion=0.0, alignment=0.0, separation=6.0
    )
    assert separation_perception(input_alphabet, normal_fov_boid) == [0, 1]
