import pandas as pd
import matplotlib.pyplot as plt

def plot_boid_trajectories(csv_file: str):
    """
    Reads boid simulation data from a CSV file and plots each boid's trajectory
    (pos_x, pos_y) over time on a 2D plane.
    """
    # 1. Load CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # 2. We group by boid_id so each boid gets its own trajectory line
    boid_ids = df["boid_id"].unique()
    
    # 3. Plot each boid’s trajectory with a different color
    for boid_id in boid_ids:
        # Filter out only the rows for this boid
        boid_data = df[df["boid_id"] == boid_id]
        
        # Plot pos_x vs. pos_y as a line
        plt.plot(
            boid_data["pos_x"], 
            boid_data["pos_y"], 
            label=f"Boid {boid_id}"
        )
    
    # 4. Final touches
    plt.title("Boid Trajectories")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example usage: pass the path to your CSV file
    plot_boid_trajectories("boid_simulation_results.csv")
