# Classic Boids

A faithful implementation of the boid flocking algorithm as described in [*The computational beauty of flocking: Boids revisited* (Bajec, Zimic, and Mraz, 2007)](https://www.researchgate.net/publication/243041154_The_computational_beauty_of_flocking_Boids_revisited).

## 3D Boid Simulation Video

▶️ **Click below to watch the 3D simulation in action:**

[![Boids Animation](https://img.youtube.com/vi/RdKcSScaeV4/0.jpg)](https://www.youtube.com/watch?v=RdKcSScaeV4)

<div align="center"><i>👆 Click the image to watch the full video on YouTube 👆</i></div>

## Overview

This project implements the mathematical model of boid behavior with:

- Support for both 2D and 3D simulations
- Precise implementation of separation, alignment, and cohesion drives
- Configurable perception parameters (distance, field of view)
- Visualization tools for generating animations

The implementation follows the formal model from the paper, with each component (perception, drives, action selection) implemented as described in the mathematical formalism.

For details on the mathematical model, see [notes/mathematical_model.md](notes/mathematical_model.md).

## Installation

```bash
# Clone the repository
git clone https://github.com/keithSchumacher/classic_boids.git
cd classic_boids

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Running a Simulation

```python
from classic_boids.core.simulation_runner import run_2d_simulation, run_3d_simulation

# Run a 2D simulation
csv_path = run_2d_simulation(num_boids=20, num_steps=200)

# Run a 3D simulation
csv_path = run_3d_simulation(num_boids=20, num_steps=200)
```

### Generating Animations

```python
from classic_boids.utils.animate_boids import animate_boids
from classic_boids.utils.animate_boids_3d import animate_boids_3d

# Generate 2D animation
animate_boids(csv_file="path/to/simulation_results.csv")

# Generate 3D animation
animate_boids_3d(csv_file="path/to/simulation_results_3d.csv")
```

### Quick Start

Generate sample data and animations with:

```bash
# Generate 2D sample data and animation
python -m src.classic_boids.utils.generate_sample_2d_data

# Generate 3D sample data and animation
python -m src.classic_boids.utils.generate_sample_3d_data
```

## Project Structure

```
classic_boids/
├── src/classic_boids/
│   ├── core/               # Core implementation of the boid model
│   │   ├── vector.py       # Vector operations
│   │   ├── protocols.py    # Type definitions and protocols
│   │   ├── boid.py         # Boid class implementation
│   │   ├── perception.py   # Perception functions
│   │   ├── drive.py        # Drive functions
│   │   └── ...
│   ├── utils/              # Utility functions
│   │   ├── animate_boids.py       # 2D animation
│   │   ├── animate_boids_3d.py    # 3D animation
│   │   └── ...
├── tests/                  # Unit tests
├── artifacts/              # Generated data and animations
└── notes/                  # Documentation
```

## Testing

Run the test suite with:

```bash
python -m pytest tests/
```

## Documentation

- [Mathematical Model](notes/mathematical_model.md) - Detailed explanation of the mathematical formalism
- [Generating Artifacts](notes/generating_artifacts.md) - Guide to generating simulation data and animations
- [Angular Offset in 3D](notes/angular_offset_in_3d.md) - Explanation of angular calculations in 3D space
- [Camera Parameters and Perception](notes/camera_parameters_and_perception.md) - Details on perception model

## License

MIT
