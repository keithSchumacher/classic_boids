import pytest
from classic_boids.core.vector import Vector
from classic_boids.core.perception import distance, angular_offset
import numpy as np


def test_distance():
    # Test cases for the distance function
    pos_i = Vector(np.array([1, 2, 3]))
    pos_j = Vector(np.array([4, 6, 8]))
    assert (
        pytest.approx(distance(pos_i, pos_j)) == 7.071067
    ), "The distance calculation should match the expected result"


def test_distance_zero():
    # Distance to the same point should be zero
    pos_i = Vector(np.array([0, 0, 0]))
    assert distance(pos_i, pos_i) == 0, "Distance from a point to itself should be zero"

    pos_j = Vector(np.array([2, 3, 4]))
    assert (
        distance(pos_j, pos_j) == 0
    ), "Distance from a non-origin point to itself should be zero"


def test_angular_offset():
    # Test cases for the angular offset function
    pos_i = Vector(np.array([1, 0]))
    pos_j = Vector(np.array([0, 0]))
    vel_j = Vector(np.array([1, 0]))
    assert (
        pytest.approx(angular_offset(pos_i, pos_j, vel_j)) == 0
    ), "Angular offset should be zero for parallel vectors"


def test_angular_offset_perpendicular():
    # Angular offset for perpendicular vectors should be pi/2
    pos_i = Vector(np.array([1, 0]))
    pos_j = Vector(np.array([0, 0]))
    vel_j = Vector(np.array([0, 1]))
    assert (
        pytest.approx(angular_offset(pos_i, pos_j, vel_j), abs=1e-3) == np.pi / 2
    ), "Angular offset should be pi/2 for perpendicular vectors"


def test_angular_offset_opposite():
    # Test for opposite direction vectors
    pos_i = Vector(np.array([1, 0]))
    pos_j = Vector(np.array([0, 0]))
    vel_j = Vector(np.array([-1, 0]))
    assert (
        pytest.approx(angular_offset(pos_i, pos_j, vel_j)) == np.pi
    ), "Angular offset should be pi for opposite vectors"


def test_zero_velocity_angular_offset():
    pos_i = Vector(np.array([1, 0]))
    pos_j = Vector(np.array([0, 0]))
    vel_j = Vector(np.array([0, 0]))
    with pytest.raises(
        ValueError, match="Angular offset cannot be calculated if velocity is zero."
    ):
        angular_offset(pos_i, pos_j, vel_j)


def test_zero_distance_angular_offset():
    # Test for opposite direction vectors
    pos_i = Vector(np.array([1, 0]))
    vel_i = Vector(np.array([-1, 0]))
    assert pytest.approx(angular_offset(pos_i, pos_i, vel_i)) == 0
