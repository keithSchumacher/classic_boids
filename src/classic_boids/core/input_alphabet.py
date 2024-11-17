from dataclasses import dataclass, field
from typing import Generic
from .protocols import InputAlphabetProtocol, VectorType


@dataclass
class InputAlphabet(InputAlphabetProtocol[VectorType], Generic[VectorType]):
    positions: dict[int, VectorType] = field(default_factory=dict)
    velocities: dict[int, VectorType] = field(default_factory=dict)

    def get_position(self, boid_id: int) -> VectorType:
        return self.positions[boid_id]

    def get_velocity(self, boid_id: int) -> VectorType:
        return self.velocities[boid_id]

    def get_positions(self) -> dict[int, VectorType]:
        return self.positions

    def get_velocities(self) -> dict[int, VectorType]:
        return self.velocities
