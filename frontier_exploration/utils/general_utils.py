import numpy as np
from numba import njit


@njit
def wrap_heading(heading):
    """Wrap a heading in radians to the interval [-pi, pi)."""
    return (heading + np.pi) % (2 * np.pi) - np.pi
