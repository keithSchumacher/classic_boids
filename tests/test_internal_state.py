import pytest
from classic_boids.core.protocols import BoidID, DriveName, InternalStateProtocol
from classic_boids.core.internal_state import InternalState
from classic_boids.core.vector import Vector
import numpy as np


class TestInternalState:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.position = Vector(np.array([0.0, 0.0]))
        self.velocity = Vector(np.array([1.0, 1.0]))
        self.perception_distance = (
            {
                DriveName.SEPARATION: 5.0,
                DriveName.ALIGNMENT: 10.0,
                DriveName.COHESION: 15.0,
            },
        )
        self.perception_field_of_view = (
            {
                DriveName.SEPARATION: np.pi / 2,
                DriveName.ALIGNMENT: 2 * np.pi / 3,
                DriveName.COHESION: np.pi,
            },
        )
        self.action_weights = {
            DriveName.COHESION: 1.0 / 3,
            DriveName.ALIGNMENT: 1.0 / 3,
            DriveName.SEPARATION: 1.0 / 3,
        }
        self.mass = 1.0
        self.max_velocity = 10.0
        self.max_force = 5.0
        self.internal_state = InternalState(
            id=BoidID(1),
            position=self.position,
            velocity=self.velocity,
            perception_distance=self.perception_distance,
            perception_field_of_view=self.perception_field_of_view,
            mass=self.mass,
            max_achievable_velocity=self.max_velocity,
            max_achievable_force=self.max_force,
            action_weights=self.action_weights,
        )

    def test_internal_state_initialization(self):
        assert self.internal_state.id == BoidID(1)
        assert self.internal_state.position == self.position
        assert self.internal_state.velocity == self.velocity
        assert self.internal_state.perception_distance == self.perception_distance
        assert self.internal_state.perception_field_of_view == self.perception_field_of_view
        assert self.internal_state.mass == self.mass
        assert self.internal_state.max_achievable_velocity == self.max_velocity
        assert self.internal_state.max_achievable_force == self.max_force
        assert self.internal_state.action_weights == self.action_weights

    def test_internal_state_vector_operations(self):
        new_position = self.internal_state.position + Vector(np.array([2.0, 3.0]))
        assert new_position == Vector(np.array([2.0, 3.0]))
        new_velocity = self.internal_state.velocity - Vector(np.array([0.5, 0.5]))
        assert new_velocity == Vector(np.array([0.5, 0.5]))

    def test_internal_state_update(self):
        self.internal_state.position += self.internal_state.position
        assert self.internal_state.position == self.position * 2

    def test_internal_state_conforms_to_protocol(self):
        def process_internal_state(state: InternalStateProtocol[Vector, dict[DriveName, float]]):
            assert isinstance(state, InternalStateProtocol)
            assert state.id == self.internal_state.id

        process_internal_state(self.internal_state)
