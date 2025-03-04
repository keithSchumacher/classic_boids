import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def animate_boids_3d(csv_file: str, interval: int = 200, output_file: str = None):
    """
    Animate 3D boid trajectories from a CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV with columns [time, boid_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z].
    interval : int
        Delay between frames in milliseconds (controls animation speed).
    output_file : str, optional
        Name of the output file. If None, a default name will be used.
    """
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Identify unique boids and sorted time steps
    boid_ids = df['boid_id'].unique()
    time_steps = sorted(df['time'].unique())

    # 3. Create a figure and axis for 3D plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We'll store one line object per boid, so we can update each boid's trajectory
    lines = {}
    for boid_id in boid_ids:
        # Plot an empty line initially; we'll update data in `update` function
        (line,) = ax.plot([], [], [], label=f"Boid {boid_id}")
        lines[boid_id] = line

    # Optionally set axis bounds or let matplotlib auto-scale
    # Find the min and max values for each dimension to set appropriate bounds
    x_min, x_max = df['pos_x'].min(), df['pos_x'].max()
    y_min, y_max = df['pos_y'].min(), df['pos_y'].max()
    z_min, z_max = df['pos_z'].min(), df['pos_z'].max()
    
    # Add some padding to the bounds
    padding = 2
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_zlim(z_min - padding, z_max + padding)
    
    ax.set_title("3D Boid Trajectories Animation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    # ax.legend()
    
    # Add a grid for better depth perception
    ax.grid(True)

    # 5. Define update function
    def update(frame_idx):
        """
        frame_idx is an integer indexing into `time_steps`.
        We'll show data up to that time (cumulative line).
        """
        current_time = time_steps[frame_idx]

        # For each boid, filter data up to current_time
        for boid_id in boid_ids:
            # Subset boid data up to `current_time`
            boid_subset = df[(df['boid_id'] == boid_id) & (df['time'] <= current_time)]
            # Extract x, y, z arrays
            x_vals = boid_subset['pos_x'].values
            y_vals = boid_subset['pos_y'].values
            z_vals = boid_subset['pos_z'].values
            # Update the line data
            lines[boid_id].set_data(x_vals, y_vals)
            lines[boid_id].set_3d_properties(z_vals)
        
        # Optionally, update the title with current time
        ax.set_title(f"3D Boid Trajectories (t={current_time})")
        
        # Rotate the view slightly for each frame to create a more dynamic 3D effect
        ax.view_init(elev=30, azim=frame_idx % 360)
        
        # Return the line objects so FuncAnimation knows to redraw them
        return list(lines.values())

    def init():
        # This initializes the animation (empty lines, etc.)
        for boid_id in boid_ids:
            lines[boid_id].set_data([], [])
            lines[boid_id].set_3d_properties([])
        return list(lines.values())

    # 6. Create the animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        init_func=init,
        interval=interval,    # in ms
        blit=False            # blit=True doesn't work well with 3D plots
    )

    # Create the artifacts directory if it doesn't exist
    artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Set the output file path
    if output_file is None:
        output_file = "boids_3d_animation.mp4"
    
    output_path = os.path.join(artifacts_dir, output_file)
    
    # Save the animation to the artifacts folder
    anim.save(output_path, writer="ffmpeg")
    print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    animate_boids_3d("boid_simulation_results.csv", interval=100) 