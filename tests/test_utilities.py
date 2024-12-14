import numpy as np
from classic_boids.core.vector import Vector


# A helper function to compare two vectors for equality.
def vectors_close(v1: Vector, v2: Vector, atol=1e-7) -> bool:
    return np.allclose(v1.data, v2.data, atol=atol)
