import pytest
import numpy as np
from classic_boids.core.vector import Vector
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.protocols import BoidID


# Fixture to setup InputAlphabet instance for tests
@pytest.fixture
def input_alphabet():
    positions = {
        BoidID(1): Vector(np.array([1, 2])),
        BoidID(2): Vector(np.array([3, 4])),
    }
    velocities = {
        BoidID(1): Vector(np.array([5, 6])),
        BoidID(2): Vector(np.array([7, 8])),
    }
    return InputAlphabet(positions=positions, velocities=velocities)


def test_get_position(input_alphabet):
    np.testing.assert_array_equal(
        input_alphabet.get_position(BoidID(1)), Vector(np.array([1, 2]))
    )
    np.testing.assert_array_equal(
        input_alphabet.get_position(BoidID(2)), Vector(np.array([3, 4]))
    )


def test_get_velocity(input_alphabet):
    np.testing.assert_array_equal(
        input_alphabet.get_velocity(BoidID(1)), Vector(np.array([5, 6]))
    )
    np.testing.assert_array_equal(
        input_alphabet.get_velocity(BoidID(2)), Vector(np.array([7, 8]))
    )


def test_get_positions(input_alphabet):
    expected_positions = {
        BoidID(1): Vector(np.array([1, 2])),
        BoidID(2): Vector(np.array([3, 4])),
    }
    actual_positions = input_alphabet.get_positions()
    for key in expected_positions:
        np.testing.assert_array_equal(actual_positions[key], expected_positions[key])


def test_get_velocities(input_alphabet):
    expected_velocities = {
        BoidID(1): Vector(np.array([5, 6])),
        BoidID(2): Vector(np.array([7, 8])),
    }
    actual_velocities = input_alphabet.get_velocities()
    for key in expected_velocities:
        np.testing.assert_array_equal(actual_velocities[key], expected_velocities[key])


def test_get_position_not_found(input_alphabet):
    with pytest.raises(KeyError):
        input_alphabet.get_position(BoidID(3))


def test_get_velocity_not_found(input_alphabet):
    with pytest.raises(KeyError):
        input_alphabet.get_velocity(BoidID(3))
