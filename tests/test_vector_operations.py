import pytest
from classic_boids.core.vector import Vector, distance, angular_offset, normalize, truncate
import numpy as np

from tests.test_utilities import vectors_close


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
    assert distance(pos_j, pos_j) == 0, "Distance from a non-origin point to itself should be zero"


def test_angular_offset():
    # Test cases for the angular offset function
    pos_i = Vector(np.array([1, 0]))
    pos_j = Vector(np.array([0, 0]))
    vel_j = Vector(np.array([1, 0]))
    assert pytest.approx(angular_offset(pos_i, pos_j, vel_j)) == 0, "Angular offset should be zero for parallel vectors"


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
    with pytest.raises(ValueError, match="Angular offset cannot be calculated if velocity is zero."):
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
    assert np.allclose(norm_vec, vec / 5.0), "Normalized vector should be the original divided by its norm."


def test_normalization_already_normalized():
    vec = Vector(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
    norm_vec = normalize(vec)
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should remain length 1."
    assert np.allclose(norm_vec, vec), "An already normalized vector should remain unchanged."


def test_normalization_zero_vector():
    vec = Vector(np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="Cannot normalize the zero vector."):
        normalize(vec)


def test_normalization_negative_components():
    vec = Vector(np.array([-2.0, 0.0, 2.0]))
    norm_vec = normalize(vec)
    length = np.sqrt((-2.0) ** 2 + 2.0**2)  # sqrt(4+4)=sqrt(8)
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should have length 1."
    assert np.allclose(norm_vec, vec / length), "Normalized vector should be the original divided by its norm."


def test_normalization_high_dimensional():
    vec = Vector(np.array([1.0, 2.0, 3.0, 4.0]))
    norm_vec = normalize(vec)
    length = vec.norm()
    assert np.isclose(norm_vec.norm(), 1.0), "Normalized vector should have length 1."
    assert np.allclose(norm_vec, vec / length), "Normalized vector should be the original divided by its norm."


def test_truncate_vector_already_within_limit():
    v = Vector(np.array([3.0, 4.0]))  # norm = 5.0
    # maximal_size = 10.0, so no change expected
    result = truncate(v, 10.0)
    assert vectors_close(result, v), "Vector within limit should remain unchanged."


def test_truncate_vector_above_limit():
    v = Vector(np.array([6.0, 8.0]))  # norm = 10.0
    # maximal_size = 5.0, should scale down by factor 0.5
    result = truncate(v, 5.0)
    expected = Vector(np.array([3.0, 4.0]))  # norm = 5.0
    assert vectors_close(result, expected), "Vector above limit should be scaled down correctly."


def test_truncate_zero_vector():
    v = Vector(np.zeros(2))  # norm = 0.0
    # maximal_size = 5.0, zero vector stays zero
    result = truncate(v, 5.0)
    expected = Vector(np.zeros(2))
    assert vectors_close(result, expected), "Zero vector should remain zero regardless of limit."


def test_truncate_zero_limit_with_nonzero_vector():
    v = Vector(np.array([1.0, 1.0]))  # norm = sqrt(2) ~ 1.414
    # maximal_size = 0.0, should scale vector down to zero
    result = truncate(v, 0.0)
    expected = Vector(np.zeros(2))
    assert vectors_close(result, expected), "Vector should become zero if limit is zero."


def test_truncate_negative_limit():
    v = Vector(np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="must be non-negative"):
        truncate(v, -1.0)


def test_truncate_3d_vector():
    # 3D vector with norm 13 (from (3,4,12))
    v = Vector(np.array([3.0, 4.0, 12.0]))
    maximal_size = 10.0
    result = truncate(v, maximal_size)
    assert np.isclose(result.norm(), 10.0, atol=1e-7), "3D vector not truncated correctly."
    # Direction should be the same: check if proportional
    # original unit vector = v / 13. truncated should be (v/13)*10
    expected = v * (10.0 / 13.0)
    assert np.allclose(result.data, expected.data, atol=1e-7), "3D vector direction mismatch after truncation."


def test_truncate_4d_vector():
    # 4D vector with equal components
    v = Vector(np.array([2.0, -2.0, 2.0, -2.0]))  # norm = 4
    maximal_size = 3.0
    result = truncate(v, maximal_size)
    assert np.isclose(result.norm(), 3.0, atol=1e-7), "4D vector not truncated correctly."
    # Check direction: unit vector = (2,-2,2,-2)/4. truncated should be unit * 3
    expected = v * (3.0 / 4.0)
    assert np.allclose(result.data, expected.data, atol=1e-7), "4D vector direction mismatch after truncation."
