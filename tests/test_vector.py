import numpy as np
import pytest
from classic_boids.core.vector import Vector


def test_vector_equality():
    v1 = Vector(np.array([1.0, 2.0]))
    v2 = Vector(np.array([1.0, 2.0]))
    assert v1 == v2


def test_vector_inequality():
    v1 = Vector(np.array([1.0, 2.0]))
    v2 = Vector(np.array([2.0, 2.0]))
    assert v1 != v2


def test_vector_addition():
    v1 = Vector(np.array([1.0, 2.0]))
    v2 = Vector(np.array([3.0, 4.0]))
    result = v1 + v2
    expected = Vector(np.array([4.0, 6.0]))
    assert np.array_equal(result.data, expected.data), "Vector addition failed"


def test_vector_subtraction():
    v1 = Vector(np.array([5.0, 7.0]))
    v2 = Vector(np.array([3.0, 4.0]))
    result = v1 - v2
    expected = Vector(np.array([2.0, 3.0]))
    assert np.array_equal(result.data, expected.data), "Vector subtraction failed"


def test_vector_dot_product():
    v1 = Vector(np.array([1.0, 2.0]))
    v2 = Vector(np.array([3.0, 4.0]))
    result = v1.dot(v2)
    expected = 11.0  # 1*3 + 2*4 = 11
    assert result == expected, "Dot product calculation failed"


def test_scalar_multiplication():
    v = Vector(np.array([1.0, 2.0]))
    scalar = 2.5
    result = v * scalar
    expected = Vector(np.array([2.5, 5.0]))
    assert np.array_equal(result.data, expected.data), "Scalar multiplication failed"


def test_vector_division():
    v = Vector(np.array([2.0, 4.0]))

    # Test division by a non-zero scalar
    result = v / 2.0
    expected = Vector(np.array([1.0, 2.0]))
    assert np.allclose(
        result.data, expected.data
    ), f"Expected {expected.data} but got {result.data}"

    # Test division by a positive non-zero scalar
    result = v / 0.5
    expected = Vector(np.array([4.0, 8.0]))
    assert np.allclose(
        result.data, expected.data
    ), f"Expected {expected.data} but got {result.data}"

    # Test division by zero
    with pytest.raises(ZeroDivisionError, match="Cannot divide vector by zero."):
        _ = v / 0.0


def test_vector_item_access():
    v = Vector(np.array([1.0, 2.0, 3.0]))
    assert v[0] == 1.0 and v[1] == 2.0 and v[2] == 3.0, "Item access failed"


def test_vector_norm():
    v = Vector(np.array([3.0, 4.0]))
    result = v.norm()
    expected = 5.0  # sqrt(3^2 + 4^2)
    assert np.isclose(result, expected), "Norm calculation failed"


def test_vector_length():
    v = Vector(np.array([1.0, 2.0, 3.0, 4.0]))
    result = len(v)
    expected = 4
    assert result == expected, "Vector length calculation failed"
