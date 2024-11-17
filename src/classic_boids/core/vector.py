from dataclasses import dataclass
from typing import Self
import numpy as np
from numpy.typing import NDArray
from .protocols import VectorProtocol


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

    def __len__(self) -> int:
        return len(self.data)

    def dot(self, other: Self) -> float:
        return float(np.dot(self.data, other.data))

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))
