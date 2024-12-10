from dataclasses import dataclass, field
from typing import Generic
from .protocols import InputAlphabetProtocol, VectorType
from classic_boids.core.protocols import BoidID


@dataclass
class InputAlphabet(InputAlphabetProtocol[VectorType], Generic[VectorType]):
    positions: dict[BoidID, VectorType] = field(default_factory=dict)
    velocities: dict[BoidID, VectorType] = field(default_factory=dict)

    def get_position(self, boid_id: BoidID) -> VectorType:
        return self.positions[boid_id]

    def get_velocity(self, boid_id: BoidID) -> VectorType:
        return self.velocities[boid_id]

    def get_positions(self) -> dict[BoidID, VectorType]:
        return self.positions

    def get_velocities(self) -> dict[BoidID, VectorType]:
        return self.velocities
