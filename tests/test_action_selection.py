import pytest
import numpy as np
from classic_boids.core.action_selection import action_selection
from classic_boids.core.internal_state import InternalState
from classic_boids.core.protocols import DriveName
from classic_boids.core.vector import Vector


@pytest.fixture
def initial_state():
    return InternalState(
        id=1,
        position=Vector(np.array([0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0])),
        perception_distance={DriveName.SEPARATION: 10.0, DriveName.ALIGNMENT: 15.0, DriveName.COHESION: 20.0},
        perception_field_of_view={DriveName.SEPARATION: 180.0, DriveName.ALIGNMENT: 120.0, DriveName.COHESION: 360.0},
        mass=1.0,
        max_achievable_velocity=5.0,
        max_achievable_force=10.0,
        action_weights={DriveName.SEPARATION: 1.0, DriveName.ALIGNMENT: 1.0, DriveName.COHESION: 1.0},
    )


@pytest.fixture
def initial_state_3d():
    return InternalState(
        id=1,
        position=Vector(np.array([0.0, 0.0, 0.0])),
        velocity=Vector(np.array([1.0, 0.0, 0.0])),
        perception_distance={DriveName.SEPARATION: 10.0, DriveName.ALIGNMENT: 15.0, DriveName.COHESION: 20.0},
        perception_field_of_view={DriveName.SEPARATION: 180.0, DriveName.ALIGNMENT: 120.0, DriveName.COHESION: 360.0},
        mass=1.0,
        max_achievable_velocity=5.0,
        max_achievable_force=10.0,
        action_weights={DriveName.SEPARATION: 1.0, DriveName.ALIGNMENT: 1.0, DriveName.COHESION: 1.0},
    )


def test_action_selection_no_mutation(initial_state):
    # Actions chosen so the net force is along x-axis only, mirroring the scalar case:
    actions = {
        DriveName.SEPARATION: Vector(np.array([2.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([3.0, 0.0])),
        DriveName.COHESION: Vector(np.array([4.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state)

    # Original state unchanged
    assert initial_state.position == Vector(np.array([0.0, 0.0]))
    assert initial_state.velocity == Vector(np.array([1.0, 0.0]))

    # Computation:
    # net_force = [2+3+4, 0] = [9,0] magnitude=9 <=10 no truncation needed
    # new_velocity=(1,0)+(9/1)=(10,0), magnitude=10 truncated to 5 => (5,0)
    # new_position=(0,0)+(5,0)=(5,0)
    assert new_state.velocity == Vector(np.array([5.0, 0.0]))
    assert new_state.position == Vector(np.array([5.0, 0.0]))


def test_action_selection_zero_actions(initial_state):
    actions = {
        DriveName.SEPARATION: Vector(np.array([0.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([0.0, 0.0])),
        DriveName.COHESION: Vector(np.array([0.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state)
    # net_force=[0,0]
    # velocity=(1,0)+(0,0)=(1,0), magnitude=1 <=5 no truncation
    # position=(0,0)+(1,0)=(1,0)
    assert new_state.position == Vector(np.array([1.0, 0.0]))
    assert new_state.velocity == Vector(np.array([1.0, 0.0]))
    # original unchanged
    assert initial_state.position == Vector(np.array([0.0, 0.0]))
    assert initial_state.velocity == Vector(np.array([1.0, 0.0]))


def test_action_selection_exceeds_force(initial_state):
    actions = {
        DriveName.SEPARATION: Vector(np.array([20.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([0.0, 0.0])),
        DriveName.COHESION: Vector(np.array([0.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state)
    # net_force=[20,0], magnitude=20 truncated to 10 => scale=10/20=0.5 => [10,0]
    # velocity=(1,0)+(10,0)=(11,0), magnitude=11 truncated to 5 => scale=5/11 ~0.4545 => (5,0)
    # position=(0,0)+(5,0)=(5,0)
    assert new_state.velocity == Vector(np.array([5.0, 0.0]))
    assert new_state.position == Vector(np.array([5.0, 0.0]))
    # original unchanged
    assert initial_state.velocity == Vector(np.array([1.0, 0.0]))
    assert initial_state.position == Vector(np.array([0.0, 0.0]))


def test_action_selection_mixed_actions(initial_state):
    actions = {
        DriveName.SEPARATION: Vector(np.array([3.0, 3.0])),
        DriveName.ALIGNMENT: Vector(np.array([4.0, 4.0])),
        DriveName.COHESION: Vector(np.array([1.0, 1.0])),
    }
    # calculations step-by-step:
    # initial_state:
    #   velocity = (1,0)
    #   mass = 1.0
    #   max_achievable_force = 10.0
    #   max_achievable_velocity = 5.0

    # net_force = SEPARATION + ALIGNMENT + COHESION
    #           = (3,3) + (4,4) + (1,1)
    #           = (8,8)
    #
    # magnitude((8,8)) = sqrt(8^2 + 8^2) = sqrt(64+64)= sqrt(128) ≈ 11.3149
    # Since 11.3149 > max_achievable_force(10), we scale down:
    # scale = 10 / 11.3149 ≈ 0.8839
    # truncated net_force = (8 * 0.8839, 8 * 0.8839) ≈ (7.071, 7.071)
    # magnitude now = 10 (as intended)

    # new_velocity = old_velocity + (net_force / mass)
    #              = (1,0) + (7.071,7.071)/1
    #              = (1 + 7.071, 0 + 7.071)
    #              = (8.071, 7.071)
    #
    # magnitude((8.071,7.071)) = sqrt(8.071^2 + 7.071^2)
    #                          ≈ sqrt(65.14 + 49.0)
    #                          ≈ sqrt(114.14)
    #                          ≈ 10.68
    #
    # max_achievable_velocity = 5.0, so we must truncate again:
    # scale_velocity = 5 / 10.68 ≈ 0.4688
    #
    # truncated velocity = (8.071 * 0.4688, 7.071 * 0.4688)
    #                    ≈ (3.78, 3.31)
    #
    # new_position = old_position + new_velocity
    #              = (0,0) + (3.78,3.31)
    #              = (3.78,3.31) (approximate due to rounding)
    new_state = action_selection(actions, initial_state)
    expected_velocity = Vector(np.array([3.78, 3.31]))
    expected_position = Vector(np.array([3.78, 3.31]))

    # We used approximate arithmetic above; if your truncate function and vector math are exact,
    # verify the actual returned values and adjust these expected values accordingly.
    assert np.allclose(new_state.velocity.data, expected_velocity.data, atol=1e-1)
    assert np.allclose(new_state.position.data, expected_position.data, atol=1e-1)

    # Check original state unchanged
    assert initial_state.velocity == Vector(np.array([1.0, 0.0]))
    assert initial_state.position == Vector(np.array([0.0, 0.0]))


def test_3d_action_selection_no_mutation(initial_state_3d):
    # Actions chosen so the net force is along x-axis only, mirroring the scalar case:
    actions = {
        DriveName.SEPARATION: Vector(np.array([2.0, 0.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([3.0, 0.0, 0.0])),
        DriveName.COHESION: Vector(np.array([4.0, 0.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state_3d)

    # Original state unchanged
    assert initial_state_3d.position == Vector(np.array([0.0, 0.0, 0.0]))
    assert initial_state_3d.velocity == Vector(np.array([1.0, 0.0, 0.0]))

    # Computation:
    # net_force = [2+3+4, 0, 0] = [9,0,0] magnitude=9 <=10 no truncation needed
    # new_velocity=(1,0,0)+(9/1)=(10,0,0), magnitude=10 truncated to 5 => (5,0,0)
    # new_position=(0,0,0)+(5,0,0)=(5,0,0)
    assert new_state.velocity == Vector(np.array([5.0, 0.0, 0.0]))
    assert new_state.position == Vector(np.array([5.0, 0.0, 0.0]))


def test_3d_action_selection_zero_actions(initial_state_3d):
    actions = {
        DriveName.SEPARATION: Vector(np.array([0.0, 0.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([0.0, 0.0, 0.0])),
        DriveName.COHESION: Vector(np.array([0.0, 0.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state_3d)
    # net_force=[0,0,0]
    # velocity=(1,0,0)+(0,0,0)=(1,0,0), magnitude=1 <=5 no truncation
    # position=(0,0,0)+(1,0,0)=(1,0,0)
    assert new_state.position == Vector(np.array([1.0, 0.0, 0.0]))
    assert new_state.velocity == Vector(np.array([1.0, 0.0, 0.0]))
    # original unchanged
    assert initial_state_3d.position == Vector(np.array([0.0, 0.0, 0.0]))
    assert initial_state_3d.velocity == Vector(np.array([1.0, 0.0, 0.0]))


def test_3d_action_selection_exceeds_force(initial_state_3d):
    actions = {
        DriveName.SEPARATION: Vector(np.array([20.0, 0.0, 0.0])),
        DriveName.ALIGNMENT: Vector(np.array([0.0, 0.0, 0.0])),
        DriveName.COHESION: Vector(np.array([0.0, 0.0, 0.0])),
    }

    new_state = action_selection(actions, initial_state_3d)
    # net_force=[20,0,0], magnitude=20 truncated to 10 => scale=10/20=0.5 => [10,0,0]
    # velocity=(1,0,0)+(10,0,0)=(11,0,0), magnitude=11 truncated to 5 => scale=5/11 ~0.4545 => (5,0,0)
    # position=(0,0,0)+(5,0,0)=(5,0,0)
    assert new_state.velocity == Vector(np.array([5.0, 0.0, 0.0]))
    assert new_state.position == Vector(np.array([5.0, 0.0, 0.0]))
    # original unchanged
    assert initial_state_3d.velocity == Vector(np.array([1.0, 0.0, 0.0]))
    assert initial_state_3d.position == Vector(np.array([0.0, 0.0, 0.0]))


def test_3d_action_selection_mixed_actions(initial_state_3d):
    actions = {
        DriveName.SEPARATION: Vector(np.array([3.0, 3.0, 3.0])),
        DriveName.ALIGNMENT: Vector(np.array([4.0, 4.0, 4.0])),
        DriveName.COHESION: Vector(np.array([1.0, 1.0, 1.0])),
    }
    # calculations step-by-step:
    # initial_state:
    #   velocity = (1,0,0)
    #   mass = 1.0
    #   max_achievable_force = 10.0
    #   max_achievable_velocity = 5.0

    # net_force = SEPARATION + ALIGNMENT + COHESION
    #           = (3,3,3) + (4,4,4) + (1,1,1)
    #           = (8,8,8)
    #
    # magnitude((8,8,8)) = sqrt(8^2 + 8^2 + 8^2) = sqrt(64+64+64)= sqrt(192) ≈ 13.856
    # Since 13.856 > max_achievable_force(10), we scale down:
    # scale = 10 / 13.856 ≈ 0.7217
    # truncated net_force = (8 * 0.7217, 8 * 0.7217, 8 * 0.7217) ≈ (5.774, 5.774, 5.774)
    # magnitude now = 10 (as intended)

    # new_velocity = old_velocity + (net_force / mass)
    #              = (1,0,0) + (5.774,5.774,5.774)/1
    #              = (1 + 5.774, 0 + 5.774, 0 + 5.774)
    #              = (6.774, 5.774, 5.774)
    #
    # magnitude((6.774,5.774,5.774)) = sqrt(6.774^2 + 5.774^2 + 5.774^2)
    #                                ≈ sqrt(45.89 + 33.34 + 33.34)
    #                                ≈ sqrt(112.57)
    #                                ≈ 10.61
    #
    # max_achievable_velocity = 5.0, so we must truncate again:
    # scale_velocity = 5 / 10.61 ≈ 0.4713
    #
    # truncated velocity = (6.774 * 0.4713, 5.774 * 0.4713, 5.774 * 0.4713)
    #                    ≈ (3.19, 2.72, 2.72)
    #
    # new_position = old_position + new_velocity
    #              = (0,0,0) + (3.19,2.72,2.72)
    #              = (3.19,2.72,2.72) (approximate due to rounding)
    new_state = action_selection(actions, initial_state_3d)
    expected_velocity = Vector(np.array([3.19, 2.72, 2.72]))
    expected_position = Vector(np.array([3.19, 2.72, 2.72]))

    # We used approximate arithmetic above; if your truncate function and vector math are exact,
    # verify the actual returned values and adjust these expected values accordingly.
    assert np.allclose(new_state.velocity.data, expected_velocity.data, atol=1e-1)
    assert np.allclose(new_state.position.data, expected_position.data, atol=1e-1)

    # Check original state unchanged
    assert initial_state_3d.velocity == Vector(np.array([1.0, 0.0, 0.0]))
    assert initial_state_3d.position == Vector(np.array([0.0, 0.0, 0.0]))
