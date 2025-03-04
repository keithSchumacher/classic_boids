import pytest
import numpy as np
from classic_boids.utils.create_sample_boids import create_sample_boids, create_sample_boids_3d
from classic_boids.core.protocols import DriveName


def test_create_sample_boids():
    """Test that the create_sample_boids function creates the correct number of 2D boids."""
    num_boids = 5
    boids = create_sample_boids(num_boids)
    
    # Check that we have the correct number of boids
    assert len(boids) == num_boids
    
    # Check that each boid has 2D position and velocity vectors
    for boid in boids:
        assert len(boid.internal_state.position.data) == 2
        assert len(boid.internal_state.velocity.data) == 2


def test_create_sample_boids_3d():
    """Test that the create_sample_boids_3d function creates the correct number of 3D boids."""
    num_boids = 5
    boids = create_sample_boids_3d(num_boids)
    
    # Check that we have the correct number of boids
    assert len(boids) == num_boids
    
    # Check that each boid has 3D position and velocity vectors
    for boid in boids:
        assert len(boid.internal_state.position.data) == 3
        assert len(boid.internal_state.velocity.data) == 3


def test_boid_properties():
    """Test that the created boids have the expected properties."""
    boids_2d = create_sample_boids(1)
    boids_3d = create_sample_boids_3d(1)
    
    for boid_list in [boids_2d, boids_3d]:
        boid = boid_list[0]
        
        # Check that the boid has the expected perception distances
        assert boid.internal_state.perception_distance[DriveName.SEPARATION] == 5.0
        assert boid.internal_state.perception_distance[DriveName.ALIGNMENT] == 10.0
        assert boid.internal_state.perception_distance[DriveName.COHESION] == 15.0
        
        # Check that the boid has the expected field of view angles
        assert boid.internal_state.perception_field_of_view[DriveName.SEPARATION] == np.pi / 2
        assert boid.internal_state.perception_field_of_view[DriveName.ALIGNMENT] == 2 * np.pi / 3
        assert boid.internal_state.perception_field_of_view[DriveName.COHESION] == np.pi
        
        # Check other properties
        assert boid.internal_state.mass == 1.0
        assert boid.internal_state.max_achievable_velocity == 10.0
        assert boid.internal_state.max_achievable_force == 5.0
        
        # Check action weights
        assert boid.internal_state.action_weights[DriveName.COHESION] == 1.0 / 3
        assert boid.internal_state.action_weights[DriveName.ALIGNMENT] == 1.0 / 3
        assert boid.internal_state.action_weights[DriveName.SEPARATION] == 1.0 / 3 