import numpy as np
from numba import jit

@jit(nopython=True)
def _generate_polyad_sign_patterns(D: int) -> tuple:
    """
    Generate grid and sign patterns for polyads of dimension D.

    Parameters
    ----------
    D : int
        Polyad dimension (number of nodes in the polyad).

    Returns
    -------
    tuple
        (grid, signs):
            grid: np.ndarray of shape (2**D, D), all binary patterns for D nodes
            signs: np.ndarray of shape (2**D,), sign pattern for each configuration
    """
    grid = np.empty( (2**D, D), dtype=np.int32 )
    for j in range(2**D):
        for d in range(D):
            grid[j][D-d-1] = (j // 2**d) % 2
    signs = (-1)**grid.sum(axis=1)
    
    return grid, signs