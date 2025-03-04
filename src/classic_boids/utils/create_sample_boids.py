from typing import List

import numpy as np
from classic_boids.core.boid import Boid
from classic_boids.core.internal_state import InternalState
from classic_boids.core.drive import alignment_drive, cohesion_drive, separation_drive
from classic_boids.core.perception import perception
from classic_boids.core.protocols import BoidID, DriveName
from classic_boids.core.vector import Vector


def create_sample_boids(num_boids: int) -> List[Boid]:
    """
    Utility function to create some sample Boid objects.
    You can randomize positions/velocities or define them however you'd like.
    """

    # Create the drives and perceptions you need
    perception_functions = {
        DriveName.SEPARATION: perception,
        DriveName.ALIGNMENT: perception,
        DriveName.COHESION: perception,
    }
    drive_functions = {
        DriveName.SEPARATION: separation_drive,
        DriveName.ALIGNMENT: alignment_drive,
        DriveName.COHESION: cohesion_drive,
    }

    boids = []
    for i in range(num_boids):
        # Example random initialization for demonstration
        boid_id = BoidID(i)
        position = Vector(np.random.uniform(-10.0, 10.0, size=2))
        velocity = Vector(np.random.uniform(-1.0, 1.0, size=2))

        internal_state = InternalState(
            id=boid_id,
            position=position,
            velocity=velocity,
            perception_distance={
                DriveName.SEPARATION: 5.0,
                DriveName.ALIGNMENT: 10.0,
                DriveName.COHESION: 15.0,
            },
            perception_field_of_view={
                DriveName.SEPARATION: np.pi / 2,
                DriveName.ALIGNMENT: 2 * np.pi / 3,
                DriveName.COHESION: np.pi,
            },
            mass=1.0,
            max_achievable_velocity=10.0,
            max_achievable_force=5.0,
            action_weights={
                DriveName.COHESION: 1.0 / 3,
                DriveName.ALIGNMENT: 1.0 / 3,
                DriveName.SEPARATION: 1.0 / 3,
            },
        )

        boid = Boid(
            internal_state=internal_state,
            perception_functions=perception_functions,
            drive_functions=drive_functions,
        )
        boids.append(boid)

    return boids


def create_sample_boids_3d(num_boids: int) -> List[Boid]:
    """
    Utility function to create some sample 3D Boid objects.
    Creates boids with 3D position and velocity vectors.
    """

    # Create the drives and perceptions you need
    perception_functions = {
        DriveName.SEPARATION: perception,
        DriveName.ALIGNMENT: perception,
        DriveName.COHESION: perception,
    }
    drive_functions = {
        DriveName.SEPARATION: separation_drive,
        DriveName.ALIGNMENT: alignment_drive,
        DriveName.COHESION: cohesion_drive,
    }

    boids = []
    for i in range(num_boids):
        # Example random initialization for demonstration
        boid_id = BoidID(i)
        position = Vector(np.random.uniform(-10.0, 10.0, size=3))
        velocity = Vector(np.random.uniform(-1.0, 1.0, size=3))

        internal_state = InternalState(
            id=boid_id,
            position=position,
            velocity=velocity,
            perception_distance={
                DriveName.SEPARATION: 5.0,
                DriveName.ALIGNMENT: 10.0,
                DriveName.COHESION: 15.0,
            },
            perception_field_of_view={
                DriveName.SEPARATION: np.pi / 2,
                DriveName.ALIGNMENT: 2 * np.pi / 3,
                DriveName.COHESION: np.pi,
            },
            mass=1.0,
            max_achievable_velocity=10.0,
            max_achievable_force=5.0,
            action_weights={
                DriveName.COHESION: 1.0 / 3,
                DriveName.ALIGNMENT: 1.0 / 3,
                DriveName.SEPARATION: 1.0 / 3,
            },
        )

        boid = Boid(
            internal_state=internal_state,
            perception_functions=perception_functions,
            drive_functions=drive_functions,
        )
        boids.append(boid)

    return boids