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


# Fixture to setup 3D InputAlphabet instance for tests
@pytest.fixture
def input_alphabet_3d():
    positions = {
        BoidID(1): Vector(np.array([1, 2, 3])),
        BoidID(2): Vector(np.array([4, 5, 6])),
    }
    velocities = {
        BoidID(1): Vector(np.array([7, 8, 9])),
        BoidID(2): Vector(np.array([10, 11, 12])),
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


# 3D Tests
def test_3d_get_position(input_alphabet_3d):
    np.testing.assert_array_equal(
        input_alphabet_3d.get_position(BoidID(1)), Vector(np.array([1, 2, 3]))
    )
    np.testing.assert_array_equal(
        input_alphabet_3d.get_position(BoidID(2)), Vector(np.array([4, 5, 6]))
    )
    # Verify it's 3D
    assert len(input_alphabet_3d.get_position(BoidID(1))) == 3


def test_3d_get_velocity(input_alphabet_3d):
    np.testing.assert_array_equal(
        input_alphabet_3d.get_velocity(BoidID(1)), Vector(np.array([7, 8, 9]))
    )
    np.testing.assert_array_equal(
        input_alphabet_3d.get_velocity(BoidID(2)), Vector(np.array([10, 11, 12]))
    )
    # Verify it's 3D
    assert len(input_alphabet_3d.get_velocity(BoidID(1))) == 3


def test_3d_get_positions(input_alphabet_3d):
    expected_positions = {
        BoidID(1): Vector(np.array([1, 2, 3])),
        BoidID(2): Vector(np.array([4, 5, 6])),
    }
    actual_positions = input_alphabet_3d.get_positions()
    for key in expected_positions:
        np.testing.assert_array_equal(actual_positions[key], expected_positions[key])
        # Verify it's 3D
        assert len(actual_positions[key]) == 3


def test_3d_get_velocities(input_alphabet_3d):
    expected_velocities = {
        BoidID(1): Vector(np.array([7, 8, 9])),
        BoidID(2): Vector(np.array([10, 11, 12])),
    }
    actual_velocities = input_alphabet_3d.get_velocities()
    for key in expected_velocities:
        np.testing.assert_array_equal(actual_velocities[key], expected_velocities[key])
        # Verify it's 3D
        assert len(actual_velocities[key]) == 3


def test_3d_get_position_not_found(input_alphabet_3d):
    with pytest.raises(KeyError):
        input_alphabet_3d.get_position(BoidID(3))


def test_3d_get_velocity_not_found(input_alphabet_3d):
    with pytest.raises(KeyError):
        input_alphabet_3d.get_velocity(BoidID(3))
