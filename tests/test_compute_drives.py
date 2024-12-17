from unittest.mock import Mock
import numpy as np
from classic_boids.core.drive import compute_drives
from classic_boids.core.protocols import DriveName, InternalStateProtocol, NeighborhoodProtocol
from classic_boids.core.vector import Vector


def test_compute_drives():
    # Arrange: Create mocks for each drive function
    separation_mock = Mock(return_value=Vector(np.array([1.0, 1.0])))
    alignment_mock = Mock(return_value=Vector(np.array([2.0, 2.0])))
    cohesion_mock = Mock(return_value=Vector(np.array([3.0, 3.0])))

    drive_functions = {
        DriveName.SEPARATION: separation_mock,
        DriveName.ALIGNMENT: alignment_mock,
        DriveName.COHESION: cohesion_mock,
    }

    # Mock neighborhood and internal_state
    neighborhood = Mock(spec=NeighborhoodProtocol)
    internal_state = Mock(spec=InternalStateProtocol)

    # Act: Call the function under test
    result = compute_drives(drive_functions, neighborhood, internal_state)

    # Assert: Check that the results are as expected
    assert result[DriveName.SEPARATION] == Vector(np.array([1.0, 1.0]))
    assert result[DriveName.ALIGNMENT] == Vector(np.array([2.0, 2.0]))
    assert result[DriveName.COHESION] == Vector(np.array([3.0, 3.0]))

    # Ensure each drive function was called with the expected arguments
    separation_mock.assert_called_once_with(neighborhood, internal_state)
    alignment_mock.assert_called_once_with(neighborhood, internal_state)
    cohesion_mock.assert_called_once_with(neighborhood, internal_state)
