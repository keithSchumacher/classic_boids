import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_boids(csv_file: str, interval: int = 200):
    """
    Animate boid trajectories from a CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV with columns [time, boid_id, pos_x, pos_y, vel_x, vel_y].
    interval : int
        Delay between frames in milliseconds (controls animation speed).
    """
    # 1. Load data
    df = pd.read_csv(csv_file)

    # 2. Identify unique boids and sorted time steps
    boid_ids = df['boid_id'].unique()
    time_steps = sorted(df['time'].unique())

    # 3. Create a figure and axis
    fig, ax = plt.subplots()
    
    # We’ll store one line object per boid, so we can update each boid’s trajectory
    lines = {}
    for boid_id in boid_ids:
        # Plot an empty line initially; we'll update data in `update` function
        (line,) = ax.plot([], [], label=f"Boid {boid_id}")
        lines[boid_id] = line

    # Optionally set axis bounds or let matplotlib auto-scale
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_title("Boid Trajectories Animation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    # ax.legend()
    ax.grid(True)

    # 4. Prepare data for faster access:
    # We'll store positions in a dict keyed by (boid_id -> times -> (x, y))
    # so that we can accumulate or pick each time step quickly.
    # However, for line animation, we might just filter at each frame too.
    # For demonstration, let's do a direct approach each frame.

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
            # Extract x, y arrays
            x_vals = boid_subset['pos_x'].values
            y_vals = boid_subset['pos_y'].values
            # Update the line data
            # print(boid_id, x_vals, y_vals)
            lines[boid_id].set_data(x_vals, y_vals)
        
        # Optionally, update the title with current time
        ax.set_title(f"Boid Trajectories (t={current_time})")
        
        # Return the line objects so FuncAnimation knows to redraw them
        return list(lines.values())

    def init():
        # This initializes the animation (empty lines, etc.)
        for boid_id in boid_ids:
            lines[boid_id].set_data([], [])
        return list(lines.values())

    # 6. Create the animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        init_func=init,
        interval=interval,    # in ms
        blit=True             # can be True or False, sometimes True is more performant
    )

    # plt.show()
    # or if you want to save to an MP4 (requires ffmpeg or similar):
    anim.save("boids_animation.mp4", writer="ffmpeg")

if __name__ == "__main__":
    # Example usage
    animate_boids("boid_simulation_results.csv", interval=100)
