import numpy as np
import pytest

from classic_boids.core.action_selection import action_selection
from classic_boids.core.boid import Boid
from classic_boids.core.drive import alignment_drive, cohesion_drive, compute_drives, separation_drive
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.core.internal_state import InternalState
from classic_boids.core.perception import compute_perceptions, perception
from classic_boids.core.protocols import (
    DriveName,
    BoidID,
)
from classic_boids.core.vector import Vector


class TestBoid:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.id = BoidID(1)
        self.position = Vector(np.array([0.0, 0.0]))
        self.velocity = Vector(np.array([1.0, 1.0]))
        self.perception_distance = {DriveName.SEPARATION: 5.0, DriveName.ALIGNMENT: 10.0, DriveName.COHESION: 15.0}
        self.perception_field_of_view = {
            DriveName.SEPARATION: np.pi / 2,
            DriveName.ALIGNMENT: 2 * np.pi / 3,
            DriveName.COHESION: np.pi,
        }
        self.action_weights = {DriveName.COHESION: 1.0 / 3, DriveName.ALIGNMENT: 1.0 / 3, DriveName.SEPARATION: 1.0 / 3}
        self.mass = 1.0
        self.max_velocity = 10.0
        self.max_force = 5.0
        self.internal_state = InternalState(
            id=self.id,
            position=self.position,
            velocity=self.velocity,
            perception_distance=self.perception_distance,
            perception_field_of_view=self.perception_field_of_view,
            mass=self.mass,
            max_achievable_velocity=self.max_velocity,
            max_achievable_force=self.max_force,
            action_weights=self.action_weights,
        )
        self.perception_functions = {
            DriveName.SEPARATION: perception,
            DriveName.ALIGNMENT: perception,
            DriveName.COHESION: perception,
        }
        self.drive_functions = {
            DriveName.SEPARATION: separation_drive,
            DriveName.ALIGNMENT: alignment_drive,
            DriveName.COHESION: cohesion_drive,
        }
        self.boid = Boid(
            internal_state=self.internal_state,
            perception_functions=self.perception_functions,
            drive_functions=self.drive_functions,
        )
        self.input_alphabet_positions = {self.id: self.position, BoidID(2): Vector(np.array([3, 4]))}
        self.input_alphabet_velocities = {
            self.id: self.velocity,
            BoidID(2): Vector(np.array([7, 8])),
        }
        self.input_alphabet = InputAlphabet(
            positions=self.input_alphabet_positions, velocities=self.input_alphabet_velocities
        )

    def test_boid_initialization(self):
        assert self.boid.internal_state == self.internal_state
        assert self.boid.perception_functions == self.perception_functions
        assert self.boid.drive_functions == self.drive_functions

    def test_step(self):
        expected_neighborhoods = compute_perceptions(
            self.perception_functions, self.input_alphabet, self.internal_state
        )
        expected_actions = compute_drives(self.drive_functions, expected_neighborhoods, self.internal_state)
        expected_internal_state = action_selection(expected_actions, self.internal_state)
        expected_output_alphabet = expected_internal_state.get_output_alphabet()
        results = self.boid.step(self.input_alphabet)
        # Check if boid's state has updated
        assert expected_internal_state == self.boid.internal_state
        # Check it output alphabet is as expected
        assert expected_output_alphabet == results
        assert (
            expected_internal_state.id,
            expected_internal_state.position,
            expected_internal_state.velocity,
        ) == results
