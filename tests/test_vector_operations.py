import pytest
from classic_boids.core.vector import Vector, distance, angular_offset, normalize
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


def test_normalization_basic():
    vec = Vector(np.array([3.0, 4.0]))  # Length should be 5.0
    norm_vec = normalize(vec)
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should have length 1."
    # Check direction: should be proportional to original
    assert np.allclose(
        norm_vec, vec * (1 / 5.0)
    ), "Normalized vector should be the original divided by its norm."


def test_normalization_already_normalized():
    vec = Vector(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    norm_vec = normalize(vec)
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should remain length 1."
    assert np.allclose(
        norm_vec, vec
    ), "An already normalized vector should remain unchanged."


def test_normalization_zero_vector():
    vec = Vector(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="Cannot normalize the zero vector."):
        normalize(vec)


def test_normalization_negative_components():
    vec = Vector(np.array([-2.0, 0.0, 2.0]))
    norm_vec = normalize(vec)
    length = np.sqrt((-2.0) ** 2 + 2.0**2)  # sqrt(4+4)=sqrt(8)
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should have length 1."
    assert np.allclose(
        norm_vec, vec * (1.0 / length)
    ), "Normalized vector should be the original divided by its norm."


def test_normalization_high_dimensional():
    vec = Vector(np.array([1.0, 2.0, 3.0, 4.0]))
    norm_vec = normalize(vec)
    length = vec.norm()
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should have length 1."
    assert np.allclose(
        norm_vec, vec * (1.0 / length)
    ), "Normalized vector should be the original divided by its norm."
