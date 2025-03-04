import os
from typing import Optional

from classic_boids.core.simulation_runner import run_2d_simulation
from classic_boids.utils.animate_boids import animate_boids


def generate_sample_2d_data(num_boids: int = 20, num_steps: int = 200, output_file: Optional[str] = None) -> str:
    """
    Generate sample 2D boid data and save it to a CSV file.
    
    Parameters
    ----------
    num_boids : int, optional
        Number of boids to simulate. Default is 20.
    num_steps : int, optional
        Number of time steps to simulate. Default is 200.
    output_file : str, optional
        Name of the output file. If None, defaults to "boid_simulation_results_2d.csv".
        The file will be saved in the artifacts directory.
        
    Returns
    -------
    str
        Path to the generated CSV file.
    """
    # Create the artifacts directory if it doesn't exist
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Set the output path
    if output_file is not None:
        output_path = os.path.join(artifacts_dir, output_file)
    else:
        output_path = None  # Let SimulationRunner use the default
    
    # Run the simulation using SimulationRunner
    csv_path = run_2d_simulation(num_boids=num_boids, num_steps=num_steps, output_csv_path=output_path)
    
    print(f"Sample 2D data saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Generate sample data
    csv_path = generate_sample_2d_data()
    
    # Animate the generated data
    output_file = os.path.join(os.path.dirname(csv_path), "boids_animation.mp4")
    animate_boids(csv_path, interval=100, output_file=output_file) 