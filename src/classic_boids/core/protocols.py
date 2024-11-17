from typing import Protocol, Generic, Self, TypeVar, runtime_checkable


# TODO make sure methods are compatible with OpenUSD Vec3D
# https://docs.omnivrse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/Gf.html#pxr.Gf.Vec3d
class VectorProtocol(Protocol):
    def __getitem__(self, index: int) -> float:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __mul__(self, scalar: float) -> Self:
        ...

    def __len__(self) -> int:
        ...

    def dot(self, other: Self) -> float:
        ...

    def norm(self) -> float:
        ...


VectorType = TypeVar("VectorType", bound=VectorProtocol)
PerceptionAttributeType = TypeVar("PerceptionAttributeType")


# TODO change Dict[int, VectorType] to Dict[BoidID, VectorType]
class InputAlphabetProtocol(Protocol, Generic[VectorType]):
    positions: dict[int, VectorType]
    velocities: dict[int, VectorType]

    def get_position(self, boid_id: int) -> VectorType:
        ...

    def get_velocity(self, boid_id: int) -> VectorType:
        ...

    def get_positions(self) -> dict[int, VectorType]:
        ...

    def get_velocities(self) -> dict[int, VectorType]:
        ...


@runtime_checkable
class InternalStateProtocol(Protocol, Generic[VectorType, PerceptionAttributeType]):
    id: int
    position: VectorType
    velocity: VectorType
    perception_distance: PerceptionAttributeType
    perception_field_of_view: PerceptionAttributeType
    mass: float
    max_achievable_velocity: float
    max_achievable_force: float


# TODO change output type to list of boid_ids or boids
class PerceptionFunctionProtocol(Protocol):
    def __call__(
        self,
        input_alphabet: InputAlphabetProtocol,
        internal_state: InternalStateProtocol,
    ) -> list[int]:
        ...
