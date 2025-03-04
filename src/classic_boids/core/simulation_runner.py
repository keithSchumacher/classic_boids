import csv
import os
import numpy as np
from typing import List, Optional

from classic_boids.core.boid import Boid
from classic_boids.core.protocols import BoidID
from classic_boids.core.vector import Vector
from classic_boids.core.input_alphabet import InputAlphabet
from classic_boids.utils.create_sample_boids import create_sample_boids, create_sample_boids_3d


class SimulationRunner:
    """
    A harness for running a multi-boid simulation and storing results.
    """

    def __init__(self, boids: List[Boid], num_steps: int, is_3d: bool = False):
        """
        Parameters
        ----------
        boids : List[Boid]
            A list of initialized Boid objects.
        num_steps : int
            How many timesteps to run in the simulation.
        is_3d : bool, optional
            Whether the simulation is 3D (True) or 2D (False). Default is False.
        """
        self.boids = boids
        self.num_steps = num_steps
        self.is_3d = is_3d

    def run(self, output_csv_path: Optional[str] = None) -> str:
        """
        Run the simulation for the specified number of steps
        and write all boid positions/velocities to a CSV file.

        Parameters
        ----------
        output_csv_path : str, optional
            File path for the CSV file to write results.
            If None, the file will be saved to the artifacts folder with a default name.

        Returns
        -------
        str
            The path to the CSV file where results were saved.
        """
        # If no output path is provided, use the artifacts folder
        if output_csv_path is None:
            # Create the artifacts directory if it doesn't exist
            artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Set default filename based on dimension
            if self.is_3d:
                output_csv_path = os.path.join(artifacts_dir, "boid_simulation_results_3d.csv")
            else:
                output_csv_path = os.path.join(artifacts_dir, "boid_simulation_results_2d.csv")
        
        # Open CSV file and write header
        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header based on dimension
            if self.is_3d:
                writer.writerow(["time", "boid_id", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"])
            else:
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
                    if self.is_3d:
                        writer.writerow([t, int(id), position[0], position[1], position[2], 
                                         velocity[0], velocity[1], velocity[2]])
                    else:
                        writer.writerow([t, int(id), position[0], position[1], 
                                         velocity[0], velocity[1]])
        
        print(f"Simulation results saved to {output_csv_path}")
        return output_csv_path


def run_2d_simulation(num_boids: int = 20, num_steps: int = 200, output_csv_path: Optional[str] = None) -> str:
    """
    Run a 2D boid simulation and save the results to a CSV file.
    
    Parameters
    ----------
    num_boids : int, optional
        Number of boids to simulate. Default is 20.
    num_steps : int, optional
        Number of time steps to simulate. Default is 200.
    output_csv_path : str, optional
        File path for the CSV file to write results.
        If None, the file will be saved to the artifacts folder with a default name.
    
    Returns
    -------
    str
        The path to the CSV file where results were saved.
    """
    # Create some 2D boids
    boids = create_sample_boids(num_boids)

    # Create the simulation harness
    sim_runner = SimulationRunner(boids=boids, num_steps=num_steps, is_3d=False)

    # Run simulation and save results
    return sim_runner.run(output_csv_path=output_csv_path)


def run_3d_simulation(num_boids: int = 20, num_steps: int = 200, output_csv_path: Optional[str] = None) -> str:
    """
    Run a 3D boid simulation and save the results to a CSV file.
    
    Parameters
    ----------
    num_boids : int, optional
        Number of boids to simulate. Default is 20.
    num_steps : int, optional
        Number of time steps to simulate. Default is 200.
    output_csv_path : str, optional
        File path for the CSV file to write results.
        If None, the file will be saved to the artifacts folder with a default name.
    
    Returns
    -------
    str
        The path to the CSV file where results were saved.
    """
    # Create some 3D boids
    boids = create_sample_boids_3d(num_boids)

    # Create the simulation harness
    sim_runner = SimulationRunner(boids=boids, num_steps=num_steps, is_3d=True)

    # Run simulation and save results
    return sim_runner.run(output_csv_path=output_csv_path)


def main():
    """
    Example usage of the SimulationRunner class.
    """
    # Run a 2D simulation
    csv_2d = run_2d_simulation(num_boids=20, num_steps=200)
    
    # Run a 3D simulation
    csv_3d = run_3d_simulation(num_boids=20, num_steps=200)
    
    # Optionally animate the results
    print("\nTo animate the results, run:")
    print(f"  python -m src.classic_boids.utils.animate_boids {csv_2d}")
    print(f"  python -m src.classic_boids.utils.animate_boids_3d {csv_3d}")


if __name__ == "__main__":
    main()
