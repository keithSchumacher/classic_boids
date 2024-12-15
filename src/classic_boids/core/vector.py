from dataclasses import dataclass
from typing import Self
import numpy as np
from numpy.typing import NDArray
from .protocols import VectorProtocol, VectorType


@dataclass
class Vector(VectorProtocol):
    data: NDArray[np.float64]

    def __eq__(self, other: Self) -> bool:
        return np.array_equal(self.data, other.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __add__(self, other: Self) -> "Vector":
        return Vector(self.data + other.data)

    def __sub__(self, other: Self) -> "Vector":
        return Vector(self.data - other.data)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.data * scalar)

    def __truediv__(self, scalar: float) -> "Vector":
        if scalar == 0.0:
            raise ZeroDivisionError("Cannot divide vector by zero.")
        return Vector(self.data / scalar)

    def __len__(self) -> int:
        return len(self.data)

    def dot(self, other: Self) -> float:
        return float(np.dot(self.data, other.data))

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))


def distance(position_i: VectorType, position_j: VectorType) -> float:
    """
    Distnace of Boid B_i from observed Boid B_j
    """
    return (position_i - position_j).norm()


def angular_offset(position_i: VectorType, position_j: VectorType, velocity_j: VectorType) -> float:
    """Angular offset of Boid B_j from Boid B_i"""
    difference = position_i - position_j
    numerator = velocity_j.dot(difference)
    difference_magnitude = difference.norm()
    velocity_magnitude = velocity_j.norm()
    denominator = velocity_magnitude * difference_magnitude
    if difference_magnitude == 0.0:
        return 0.0
    if velocity_magnitude == 0.0:
        raise ValueError("Angular offset cannot be calculated if velocity is zero.")
    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be zero.")

    return np.arccos(numerator / denominator)


def normalize(vector: VectorType) -> VectorType:
    """
    Returns a normalized version of the input vector.

    Assumes vector is not the zero vector. If it is zero-length,
    this function may throw a ZeroDivisionError or produce NaNs.
    """
    if (norm := vector.norm()) == 0:
        raise ValueError("Cannot normalize the zero vector.")
    return vector / norm


def truncate(vector: VectorType, maximal_size: float) -> VectorType:
    """
    Truncate a vector so that its norm does not exceed a given maximum size.

    If the vector's norm (magnitude) is greater than `maximal_size`, the vector
    is scaled down proportionally so that its new norm equals `maximal_size`.
    If the vector's norm is less than or equal to `maximal_size`, it is returned
    unchanged. If `maximal_size` is zero or a positive number and the vector is zero-length,
    the vector is returned unchanged.

    Raises:
        ValueError: If `maximal_size` is negative.
    """
    if maximal_size < 0:
        raise ValueError("maximal_size must be non-negative.")
    norm = vector.norm()
    if norm <= maximal_size:
        return vector
    return normalize(vector) * maximal_size
