# Generating Artifacts in Classic Boids

This document provides instructions on the various ways to generate artifacts (visualizations, data files, etc.) in the Classic Boids project.

## Table of Contents

1. [2D Boid Animations](#2d-boid-animations)
2. [3D Boid Animations](#3d-boid-animations)
3. [Generating Sample Data](#generating-sample-data)
4. [Custom Artifact Generation](#custom-artifact-generation)

## 2D Boid Animations

The project includes utilities to create 2D animations of boid movements.

### Using Existing Data

If you already have a CSV file with boid trajectory data, you can generate a 2D animation:

```python
from classic_boids.utils.animate_boids import animate_boids

# Basic usage with default parameters
animate_boids("path/to/your/data.csv")

# Customized usage
animate_boids(
    csv_file="path/to/your/data.csv",
    interval=100,  # Animation speed (milliseconds between frames)
    output_file="custom_animation_name.mp4"  # Custom output filename
)
```

The CSV file should have the following columns:
- `time`: The time step
- `boid_id`: Unique identifier for each boid
- `pos_x`, `pos_y`: Position coordinates
- `vel_x`, `vel_y`: Velocity components

The animation will be saved to the `artifacts` folder in the project root.

## 3D Boid Animations

For 3D simulations, the project provides utilities to create 3D animations.

### Using Existing Data

If you have a CSV file with 3D boid trajectory data:

```python
from classic_boids.utils.animate_boids_3d import animate_boids_3d

# Basic usage
animate_boids_3d("path/to/your/3d_data.csv")

# Customized usage
animate_boids_3d(
    csv_file="path/to/your/3d_data.csv",
    interval=100,  # Animation speed
    output_file="custom_3d_animation.mp4"  # Custom output filename
)
```

The CSV file should have the following columns:
- `time`: The time step
- `boid_id`: Unique identifier for each boid
- `pos_x`, `pos_y`, `pos_z`: 3D position coordinates
- `vel_x`, `vel_y`, `vel_z`: 3D velocity components

The 3D animation includes a rotating view to better visualize the movement in three dimensions.

## Generating Sample Data

The project includes utilities to generate sample data for both 2D and 3D simulations.

### Generating 2D Sample Data

To generate 2D sample data, you can use the built-in boid creation utilities:

```python
from classic_boids.utils.create_sample_boids import create_sample_boids
from classic_boids.core.input_alphabet import InputAlphabet
import pandas as pd
import os

# Create sample boids
num_boids = 5
boids = create_sample_boids(num_boids)

# Simulate and record data
# (Similar to the implementation in generate_sample_3d_data.py but for 2D)
# ...

# Save to CSV in artifacts folder
artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
df.to_csv(os.path.join(artifacts_dir, "boid_simulation_results_2d.csv"), index=False)
```

### Generating 3D Sample Data

To generate 3D sample data and create an animation in one step:

```python
from classic_boids.utils.generate_sample_3d_data import generate_sample_3d_data
from classic_boids.utils.animate_boids_3d import animate_boids_3d

# Generate sample data
csv_file = generate_sample_3d_data(
    num_boids=5,      # Number of boids to simulate
    num_steps=100,    # Number of time steps
    output_file="my_3d_simulation.csv"  # Optional custom filename
)

# Animate the generated data
animate_boids_3d(csv_file, interval=100)
```

This will:
1. Create 5 boids with random initial positions and velocities in 3D space
2. Simulate their movement for 100 time steps
3. Save the trajectory data to a CSV file in the artifacts folder
4. Generate a 3D animation of the trajectories

## Custom Artifact Generation

You can also create custom artifacts by extending the existing utilities.

### Creating Custom Visualizations

To create custom visualizations, you can modify the animation functions:

```python
import matplotlib.pyplot as plt
from classic_boids.utils.animate_boids import animate_boids

# Customize the animation function
def my_custom_animation(csv_file, interval=200):
    # Load data and set up figure (similar to animate_boids)
    # ...
    
    # Add custom visualization elements
    # ...
    
    # Save to artifacts folder
    # ...

# Use your custom function
my_custom_animation("path/to/data.csv")
```

### Batch Processing

For batch processing of multiple simulations:

```python
import os
from classic_boids.utils.generate_sample_3d_data import generate_sample_3d_data
from classic_boids.utils.animate_boids_3d import animate_boids_3d

# Generate multiple datasets with different parameters
for i in range(5):
    # Vary parameters for each run
    num_boids = 5 + i * 2
    
    # Generate data
    csv_file = generate_sample_3d_data(
        num_boids=num_boids,
        num_steps=100,
        output_file=f"simulation_run_{i}.csv"
    )
    
    # Create animation
    animate_boids_3d(
        csv_file=csv_file,
        interval=100,
        output_file=f"animation_run_{i}.mp4"
    )
```

## Command Line Usage

You can also run the utilities from the command line:

```bash
# Generate 3D sample data and animation
python -m src.classic_boids.utils.generate_sample_3d_data

# Animate existing data
python -m src.classic_boids.utils.animate_boids_3d path/to/your/data.csv
```

All generated artifacts will be saved to the `artifacts` folder in the project root directory. 