import numpy as np
from classic_boids.core.vector import Vector, angular_offset, distance, normalize, truncate


def test_3d_vector_equality():
    v1 = Vector(np.array([1.0, 2.0, 3.0]))
    v2 = Vector(np.array([1.0, 2.0, 3.0]))
    assert v1 == v2


def test_3d_vector_inequality():
    v1 = Vector(np.array([1.0, 2.0, 3.0]))
    v2 = Vector(np.array([1.0, 2.0, 4.0]))
    assert v1 != v2


def test_3d_vector_addition():
    v1 = Vector(np.array([1.0, 2.0, 3.0]))
    v2 = Vector(np.array([4.0, 5.0, 6.0]))
    result = v1 + v2
    expected = Vector(np.array([5.0, 7.0, 9.0]))
    assert np.array_equal(result.data, expected.data), "3D vector addition failed"


def test_3d_vector_subtraction():
    v1 = Vector(np.array([5.0, 7.0, 9.0]))
    v2 = Vector(np.array([1.0, 2.0, 3.0]))
    result = v1 - v2
    expected = Vector(np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(result.data, expected.data), "3D vector subtraction failed"


def test_3d_vector_dot_product():
    v1 = Vector(np.array([1.0, 2.0, 3.0]))
    v2 = Vector(np.array([4.0, 5.0, 6.0]))
    result = v1.dot(v2)
    expected = 32.0  # 1*4 + 2*5 + 3*6 = 32
    assert result == expected, "3D dot product calculation failed"


def test_3d_scalar_multiplication():
    v = Vector(np.array([1.0, 2.0, 3.0]))
    scalar = 2.0
    result = v * scalar
    expected = Vector(np.array([2.0, 4.0, 6.0]))
    assert np.array_equal(result.data, expected.data), "3D scalar multiplication failed"


def test_3d_vector_division():
    v = Vector(np.array([2.0, 4.0, 6.0]))
    result = v / 2.0
    expected = Vector(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(result.data, expected.data), "3D vector division failed"


def test_3d_vector_item_access():
    v = Vector(np.array([1.0, 2.0, 3.0]))
    assert v[0] == 1.0 and v[1] == 2.0 and v[2] == 3.0, "3D item access failed"


def test_3d_vector_norm():
    v = Vector(np.array([3.0, 4.0, 12.0]))
    result = v.norm()
    expected = 13.0  # sqrt(3^2 + 4^2 + 12^2) = sqrt(169) = 13
    assert np.isclose(result, expected), "3D norm calculation failed"


def test_3d_distance():
    p1 = Vector(np.array([1.0, 2.0, 3.0]))
    p2 = Vector(np.array([4.0, 6.0, 8.0]))
    result = distance(p1, p2)
    expected = np.sqrt(50)  # sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
    assert np.isclose(result, expected), "3D distance calculation failed"


def test_3d_normalize():
    v = Vector(np.array([3.0, 4.0, 12.0]))
    result = normalize(v)
    # The norm of v is 13, so the normalized vector is v/13
    expected = Vector(np.array([3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0]))
    assert np.allclose(result.data, expected.data), "3D normalize failed"


def test_3d_truncate():
    v = Vector(np.array([3.0, 4.0, 12.0]))  # norm = 13

    # Test when truncation is needed
    max_size = 5.0
    result = truncate(v, max_size)
    expected = v * (max_size / v.norm())  # Scale down to max_size
    assert np.allclose(result.data, expected.data), "3D truncate (with truncation) failed"
    assert np.isclose(result.norm(), max_size), "Truncated vector should have norm equal to max_size"

    # Test when no truncation is needed
    max_size = 20.0
    result = truncate(v, max_size)
    assert np.array_equal(result.data, v.data), "3D truncate (without truncation) should return original vector"


def test_3d_angular_offset():
    # Test case 1: vectors at 90 degrees to each other
    position_i = Vector(np.array([10.0, 0.0, 0.0]))
    position_j = Vector(np.array([0.0, 0.0, 0.0]))
    velocity_j = Vector(np.array([0.0, 1.0, 0.0]))

    result = angular_offset(position_i, position_j, velocity_j)
    expected = np.pi / 2  # 90 degrees
    assert np.isclose(result, expected), "3D angular offset for perpendicular vectors failed"

    # Test case 2: vectors in same direction
    position_i = Vector(np.array([10.0, 0.0, 0.0]))
    position_j = Vector(np.array([0.0, 0.0, 0.0]))
    velocity_j = Vector(np.array([1.0, 0.0, 0.0]))

    result = angular_offset(position_i, position_j, velocity_j)
    expected = 0.0  # 0 degrees
    assert np.isclose(result, expected), "3D angular offset for parallel vectors failed"

    # Test case 3: vectors in opposite direction
    position_i = Vector(np.array([10.0, 0.0, 0.0]))
    position_j = Vector(np.array([0.0, 0.0, 0.0]))
    velocity_j = Vector(np.array([-1.0, 0.0, 0.0]))

    result = angular_offset(position_i, position_j, velocity_j)
    expected = np.pi  # 180 degrees
    assert np.isclose(result, expected), "3D angular offset for opposite vectors failed"

    # Test case 4: arbitrary 3D case
    position_i = Vector(np.array([1.0, 2.0, 3.0]))
    position_j = Vector(np.array([0.0, 0.0, 0.0]))
    velocity_j = Vector(np.array([1.0, 1.0, 1.0]))

    result = angular_offset(position_i, position_j, velocity_j)
    # The angle between [1, 2, 3] and [1, 1, 1] is:
    # arccos((1*1 + 2*1 + 3*1) / (sqrt(1^2 + 2^2 + 3^2) * sqrt(1^2 + 1^2 + 1^2)))
    # = arccos(6 / (sqrt(14) * sqrt(3)))
    expected = np.arccos(6 / (np.sqrt(14) * np.sqrt(3)))
    assert np.isclose(result, expected), "3D angular offset for arbitrary vectors failed"
