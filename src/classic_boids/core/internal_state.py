from dataclasses import dataclass
from typing import TypedDict, Generic

from classic_boids.core.protocols import BoidID
from .protocols import VectorType, InternalStateProtocol


class PerceptionAttributes(TypedDict):
    separation: float
    alignment: float
    cohesion: float


@dataclass
class InternalState(
    Generic[VectorType], InternalStateProtocol[VectorType, PerceptionAttributes]
):
    id: BoidID
    position: VectorType
    velocity: VectorType
    perception_distance: PerceptionAttributes
    perception_field_of_view: PerceptionAttributes
    mass: float
    max_achievable_velocity: float
    max_achievable_force: float
