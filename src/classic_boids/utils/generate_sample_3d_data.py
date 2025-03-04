import os
import numpy as np
import pandas as pd
from classic_boids.utils.create_sample_boids import create_sample_boids_3d
from classic_boids.core.input_alphabet import InputAlphabet

def generate_sample_3d_data(num_boids=5, num_steps=100, output_file=None):
    """
    Generate sample 3D boid data for testing the animation function.
    
    Parameters
    ----------
    num_boids : int
        Number of boids to simulate.
    num_steps : int
        Number of time steps to simulate.
    output_file : str, optional
        Name of the output CSV file. If None, a default name will be used.
    
    Returns
    -------
    str
        Path to the generated CSV file.
    """
    # Create sample 3D boids
    boids = create_sample_boids_3d(num_boids)
    
    # Initialize data storage
    data = []
    
    # Initial positions and velocities
    positions = {boid.internal_state.id: boid.internal_state.position for boid in boids}
    velocities = {boid.internal_state.id: boid.internal_state.velocity for boid in boids}
    
    # Create input alphabet
    input_alphabet = InputAlphabet(positions=positions, velocities=velocities)
    
    # Simulate boid movement for num_steps
    for t in range(num_steps):
        # Record current state
        for boid in boids:
            boid_id = boid.internal_state.id
            position = positions[boid_id]
            velocity = velocities[boid_id]
            
            data.append({
                'time': t,
                'boid_id': int(boid_id),  # Convert BoidID to int
                'pos_x': position.data[0],
                'pos_y': position.data[1],
                'pos_z': position.data[2],
                'vel_x': velocity.data[0],
                'vel_y': velocity.data[1],
                'vel_z': velocity.data[2]
            })
        
        # Update boids
        new_states = {}
        for boid in boids:
            output = boid.step(input_alphabet)
            new_states[boid.internal_state.id] = output
        
        # Update positions and velocities for next step
        for boid_id, (_, position, velocity) in new_states.items():
            positions[boid_id] = position
            velocities[boid_id] = velocity
        
        # Update input alphabet for next step
        input_alphabet = InputAlphabet(positions=positions, velocities=velocities)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create the artifacts directory if it doesn't exist
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Set the output file path
    if output_file is None:
        output_file = "boid_simulation_results_3d.csv"
    
    output_path = os.path.join(artifacts_dir, output_file)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample 3D data saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Generate sample data and animate it
    csv_file = generate_sample_3d_data(num_boids=5, num_steps=100)
    
    # Import here to avoid circular imports
    from classic_boids.utils.animate_boids_3d import animate_boids_3d
    
    # Animate the generated data
    animate_boids_3d(csv_file, interval=100) 