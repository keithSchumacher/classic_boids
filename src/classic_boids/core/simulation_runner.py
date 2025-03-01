import csv
import numpy as np
from typing import List

from classic_boids.core.boid import Boid
from classic_boids.core.protocols import BoidID
from classic_boids.core.vector import Vector
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.utils.create_sample_boids import create_sample_boids


class SimulationRunner:
    """
    A harness for running a multi-boid simulation and storing results.
    """

    def __init__(self, boids: List[Boid], num_steps: int):
        """
        Parameters
        ----------
        boids : List[Boid]
            A list of initialized Boid objects.
        num_steps : int
            How many timesteps to run in the simulation.
        """
        self.boids = boids
        self.num_steps = num_steps

    def run(self, output_csv_path: str) -> None:
        """
        Run the simulation for the specified number of steps
        and write all boid positions/velocities to a CSV file.

        Parameters
        ----------
        output_csv_path : str
            File path for the CSV file to write results.
        """
        # Open CSV file and write header
        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time", "boid_id", "pos_x", "pos_y", "vel_x", "vel_y"])

            # Main simulation loop
            for t in range(self.num_steps):
                # 1. Gather positions and velocities for input alphabet
                positions = {}
                velocities = {}
                for boid in self.boids:
                    id, position, velocity = boid.internal_state.get_output_alphabet()
                    positions[id] = position
                    velocities[id] = velocity

                # 2. Create input alphabet for this timestep
                input_alphabet = InputAlphabet(positions=positions, velocities=velocities)

                # 3. Step each boid
                for boid in self.boids:
                    id, position, velocity = boid.step(input_alphabet)
                    # 4. After all boids update, write their new states to CSV
                    writer.writerow([t, id, position[0], position[1], velocity[0], velocity[1]])




def main():
    # Example usage:
    num_boids = 20
    num_steps = 200

    # Create some boids
    boids = create_sample_boids(num_boids)

    # Create the simulation harness
    sim_runner = SimulationRunner(boids=boids, num_steps=num_steps)

    # Run simulation and save results
    sim_runner.run(output_csv_path="boid_simulation_results.csv")


if __name__ == "__main__":
    main()
