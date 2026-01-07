import numpy as np
from numba import jit

@jit(nopython=True)
def _binary_search_edge_value(keys, values, query):
    """
    Generalized binary search for edge value in a sorted list of edge keys.

    Parameters
    ----------
    keys : np.ndarray
        Array of edge keys (shape: [n_edges, D]).
    values : np.ndarray
        Array of edge values (shape: [n_edges]).
    query : np.ndarray
        Query key (shape: [D]).

    Returns
    -------
    int
        Value associated with query key, or 0 if not found.
    """
    lo = 0
    hi, D = keys.shape
    hi -= 1

    while lo <= hi:
        mid = (lo + hi) // 2
        kmid = keys[mid]

        # --- lex_less(kmid, query) ---
        less = False
        for i in range(D):
            if kmid[i] < query[i]:
                less = True
                break
            elif kmid[i] > query[i]:
                less = False
                break

        if less:
            lo = mid + 1
            continue

        # --- lex_equal(kmid, query) ---
        equal = True
        for i in range(D):
            if kmid[i] != query[i]:
                equal = False
                break
        if equal:
            return values[mid]

        hi = mid - 1

    return 0
