from classic_boids.core.action_selection import action_selection
from classic_boids.core.drive import compute_drives
from classic_boids.core.perception import compute_perceptions
from classic_boids.core.protocols import (
    DriveFunctionProtocol,
    DriveName,
    InputAlphabetProtocol,
    InternalStateProtocol,
    PerceptionFunctionProtocol,
    VectorType,
    BoidID,
)


class Boid:
    """
    A Boid is an autonomous agent that can:
      1. Perceive its environment using a set of perception functions.
      2. Compute a set of drive actions based on those perceptions.
      3. Select and apply the resulting action(s) to update its internal state.

    Example Usage:
        # 1) Instantiate the Boid with an initial internal_state, plus
        #    dictionaries of perception and drive functions:
        boid = Boid(
            internal_state=my_internal_state,
            perception_functions={
                DriveName.SEPARATION: perception,
                DriveName.ALIGNMENT: perception,
                DriveName.COHESION: perception,
            },
            drive_functions={
                DriveName.SEPARATION: separation_drive,
                DriveName.ALIGNMENT: alignment_drive,
                DriveName.COHESION: cohesion_drive,
            },
        )

        # 2) Create or update an InputAlphabet for the current simulation step.
        #    For instance:
        #    input_alphabet = InputAlphabet(positions=..., velocities=...)

        # 3) Step the Boid through one iteration:
        boid.step(input_alphabet)
    """

    def __init__(
        self,
        internal_state: InternalStateProtocol,
        perception_functions: dict[DriveName, PerceptionFunctionProtocol],
        drive_functions: dict[DriveName, DriveFunctionProtocol],
    ):
        self.internal_state = internal_state
        self.perception_functions = perception_functions
        self.drive_functions = drive_functions

    def step(self, input_alphabet: InputAlphabetProtocol) -> tuple[BoidID, VectorType, VectorType]:
        """
        Perform one simulation step for this boid. This includes:
            1. Perceiving the environment (compute_perceptions).
            2. Computing drive vectors (compute_drives).
            3. Selecting and applying the resulting action (action_selection).
            4. Updating the boid's internal state accordingly.
        """
        # 1. Compute perceptions for each drive
        neighborhoods = compute_perceptions(self.perception_functions, input_alphabet, self.internal_state)
        # 2. Compute drives
        actions = compute_drives(self.drive_functions, neighborhoods, self.internal_state)
        # 3. Action selection => updated internal state
        self.internal_state = action_selection(actions, self.internal_state)
        # 4. Return the output alphabet
        return self.internal_state.get_output_alphabet()
