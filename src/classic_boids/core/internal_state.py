from dataclasses import dataclass
from typing import Generic

from classic_boids.core.protocols import BoidID
from .protocols import DriveName, VectorType, InternalStateProtocol


@dataclass
class InternalState(Generic[VectorType], InternalStateProtocol[VectorType, dict[DriveName, float]]):
    id: BoidID
    position: VectorType
    velocity: VectorType
    perception_distance: dict[DriveName, float]
    perception_field_of_view: dict[DriveName, float]
    mass: float
    max_achievable_velocity: float
    max_achievable_force: float
    action_weights: dict[DriveName, VectorType]
