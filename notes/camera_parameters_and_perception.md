# Camera Parameters and Perception Models

This document explains how camera lens parameters relate to the perception model used in boid simulations, particularly the concepts of "cone of vision" and "perception distance".

## Field of View and Lens Parameters

### 1. Focal Length
- **Current model equivalent**: FOV angle
- **Relationship**: Shorter focal lengths = wider FOV; longer focal lengths = narrower FOV
- **Formula**: FOV = 2 × arctan(sensor size / (2 × focal length))
- Focal length would directly determine how wide or narrow a boid's vision is

### 2. Sensor Size
- **Current model equivalent**: No direct analog in the current model
- **Importance**: Works with focal length to determine FOV
- Larger sensors with the same focal length = wider FOV
- In practical terms, this is like deciding if your boid has the vision system of a smartphone camera or a full-frame DSLR

### 3. Aspect Ratio
- **Current model equivalent**: None (current FOV is a circular cone)
- Creates rectangular rather than circular FOV (horizontal FOV ≠ vertical FOV)
- Common ratios: 4:3, 16:9, etc.
- This would allow for more realistic vision models where boids might see wider than they see tall

## Perception Distance Parameters

### 4. Depth of Field
- **Current model equivalent**: The perception_distance parameter
- Controls the range at which objects appear in focus
- Determined by aperture (f-stop), focal length, and distance to subject
- Could simulate more realistic perception where boids see clearly only within a certain distance range

### 5. Near and Far Clipping Planes
- **Current model equivalent**: Minimum and maximum perception_distance
- Define the closest and farthest distances the camera can "see"
- These would define the absolute limits of vision

## Other Relevant Parameters

### 6. Resolution
- **Current model equivalent**: None
- Determines the level of detail a boid can perceive
- Higher resolution = ability to detect smaller objects or finer details at greater distances
- Could be used to simulate different "visual acuity" for different boid species

### 7. Exposure/Sensitivity
- **Current model equivalent**: None
- Determines ability to see in different lighting conditions
- Could simulate day/night vision differences among boids

## Practical Implementation

When implementing a more realistic vision model:

1. **Camera Components**: Each boid would have a simulated camera or vision system
2. **Horizontal and Vertical FOV**: Replace the single FOV angle with horizontal and vertical FOV angles
3. **Vision Processing**: Process what the camera "sees" to determine what other entities are visible
4. **Performance Considerations**: Rendering multiple camera views can be computationally expensive

For a basic implementation, the most important parameters to focus on would be:
- Focal length or direct FOV angle settings (horizontal and vertical)
- Near and far clipping planes (minimum and maximum perception distance)
- Camera resolution (for detection detail level)

This approach would provide a much more realistic and physically-based vision model for boids, while still being manageable to implement. 