# Angular Offset in 3D Space

This document explains the concept of angular offset in 3D space and how it relates to the field of view (FOV) in boid simulations.

## Angular Offset in 2D vs 3D

The `angular_offset` function calculates the angle between the velocity vector of a boid and the vector pointing from the boid to another entity. This calculation works universally in both 2D and 3D spaces, but there are important conceptual differences:

### In 2D:
- In 2D, there's only one possible angle between two vectors (ranging from 0 to π or 0° to 180°)
- The field of view (FOV) is simply specified by a single angle, representing the maximum allowable angular deviation from the boid's heading

### In 3D:
- In the current implementation, the angular offset is still returning a single scalar value - the angle between two 3D vectors
- This single angle represents the "cone of vision" from the boid's perspective
- The calculation uses the dot product formula: `cos(θ) = (v1·v2)/(|v1|·|v2|)`
- The result is the *minimum* angle between the vectors, regardless of direction in 3D space

## Physical Meaning in 3D

The physical meaning is that:
1. A boid with velocity vector `v` can perceive another boid if the angle between `v` and the vector pointing to the other boid is less than its FOV angle
2. This creates a "vision cone" around the boid's velocity vector

## Limitations of Single-Angle FOV in 3D

While the current implementation works mathematically in 3D space, it doesn't capture what you might expect in a true 3D field of view model:

1. **Single FOV Parameter vs. Full 3D FOV**: 
   - The current implementation uses a single angular parameter for FOV
   - A complete 3D FOV model would typically use two angles (horizontal and vertical field of view)
   - Alternatively, it might use a solid angle (measured in steradians) to define a full 3D cone

2. **Cone vs. Pyramid**: 
   - The current implementation creates a circular cone of vision
   - Real animal vision is often more like a rectangular pyramid (different horizontal and vertical FOV)

3. **Asymmetric Vision**: 
   - In many animals, vision is not symmetrical around the heading vector
   - For example, humans see more below their eye level than above, and more to the sides than up/down

## Advantages of the Current Model

However, the current implementation is a valid simplification that:
- Is computationally efficient
- Still captures the core behavior of limited perception
- Works fine for symmetric conical vision models

## Implementation Details

The current implementation calculates the angular offset as follows:

```python
def angular_offset(position_i: VectorType, position_j: VectorType, velocity_j: VectorType) -> float:
    """Angular offset of Boid B_j from Boid B_i"""
    difference = position_i - position_j
    numerator = velocity_j.dot(difference)
    difference_magnitude = difference.norm()
    velocity_magnitude = velocity_j.norm()
    denominator = velocity_magnitude * difference_magnitude
    if difference_magnitude == 0.0:
        return 0.0
    if velocity_magnitude == 0.0:
        raise ValueError("Angular offset cannot be calculated if velocity is zero.")
    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be zero.")

    return np.arccos(numerator / denominator)
```

This function calculates the angle between the velocity vector of boid j and the vector pointing from boid j to boid i. If this angle is less than the FOV angle, then boid j can perceive boid i. 