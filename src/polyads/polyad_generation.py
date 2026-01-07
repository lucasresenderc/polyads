import numpy as np
from numba import jit
from .binary_search import _binary_search_edge_value
from .polyad_utils import _generate_polyad_sign_patterns


@jit(nopython=True)
def _get_minima(
        D, power_D, grid, signs, key, Y_xi, # precomputed or prealocated
        keys_p, # i = [i_1, keys_p]
        keys_q, # i' = [i_1p, keys_q]
        value_p, # Y[i_1, keys_p]
        value_q, # Y[i_1, keys_q] if D is odd else Y[i_1p, keys_q]
        edge_indices_i_1, # edge_indices[i_1]
        edge_indices_i_1p, # edge_indices[i_1p]
        edge_values_i_1, # edge_values[i_1]
        edge_values_i_1p # edge_values[i_1p]
    ) -> int:

    m_xi = min(value_p, value_q) # init m_xi
    # go on the edges that start with 1 and are positive:
    for i in range(power_D//2, power_D):
        if signs[i] == 1:
            # fill the 1->1..1 in the even case
            if (D%2 == 0 and i == power_D-1):
                Y_xi[i] = value_q
            else:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1p, edge_values_i_1p, key)
                if Y_xi[i] == 0:
                    return 0, 0
                else:
                    m_xi = min(m_xi, Y_xi[i])

    # go on the edges that start with 0 and are positive (ignore the first one that is always known):
    for i in range(1, power_D//2):
        if signs[i] == 1:
            # fill the 0->1..1 in the odd case
            if (D%2 == 1 and i == power_D//2-1):
                Y_xi[i] = value_q
            else:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1, edge_values_i_1, key)
                if Y_xi[i] == 0:
                    return 0, 0
                else:
                    m_xi = min(m_xi, Y_xi[i])

    Y_xi[0] = value_p # always known

    M_xi = -1
    # go on the edges that start with 0 and are negative (the first is always postive, we skip it):
    for i in range(1, power_D//2):
        if signs[i] == -1:
            for d in range(D-1):
                key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
            Y_xi[i] = _binary_search_edge_value(edge_indices_i_1, edge_values_i_1, key)
            M_xi = min(M_xi, Y_xi[i]) if M_xi != -1 else Y_xi[i]

    # go on the edges that start with 1 and are negative:
    for i in range(power_D//2, power_D):
        if signs[i] == -1:
            for d in range(D-1):
                key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
            Y_xi[i] = _binary_search_edge_value(edge_indices_i_1p, edge_values_i_1p, key)
            M_xi = min(M_xi, Y_xi[i])

    return m_xi, M_xi


@jit(nopython=True)
def _find_active_polyads(
    D: int,
    primary_indices: np.ndarray,
    edge_indices: np.ndarray,
    edge_values: np.ndarray,
    max_n_polyads: int
) -> tuple:
    """
    Generate all active polyads for the model.

    Parameters
    ----------
    D : int
        Index dimensions.
    primary_indices : np.ndarray
        Array of primary node indices.
    edge_indices : np.ndarray
        List of arrays of edge keys for each node, list of n x (D-1) indices
    edge_values : np.ndarray
        List of arrays of edge values for each node.
    max_n_polyads : int
        Maximum number of polyads to generate.

    Returns
    -------
    tuple
        (xis, Y_xis, m_xis, M_xis)
        where each is a np.ndarray with shape depending on the number of polyads found.
    """
    power_D = 2**D
    n_1 = primary_indices.size
    grid, signs = _generate_polyad_sign_patterns(D)
    key = np.empty(D-1, dtype=np.int32)
    Y_xi = np.empty(signs.size,  dtype=np.int32)  # TODO: document
    xis = np.empty((int(max_n_polyads), D, 2), dtype=np.int32)  # TODO: document
    Y_xis = np.empty((int(max_n_polyads), signs.size), dtype=np.int32)  # TODO: document
    m_xis = np.empty(int(max_n_polyads), dtype=np.int32)  # TODO: document
    M_xis = np.empty(int(max_n_polyads), dtype=np.int32)  # TODO: document

    i_current_polyad = 0
    for i_1 in range(n_1):
        for i_1p in range(n_1):
            if i_1p != i_1:
                p_max = edge_values[i_1].size
                q_max = p_max if D % 2 == 1 else edge_values[i_1p].size
                for p in range(p_max):
                    for q in range(q_max):
                        keys_p = edge_indices[i_1][p]
                        value_p = edge_values[i_1][p]
                        if D % 2 == 1:
                            keys_q = edge_indices[i_1][q]
                            value_q = edge_values[i_1][q]
                        else:
                            keys_q = edge_indices[i_1p][q]
                            value_q = edge_values[i_1p][q]

                        # avoid duplicate polyads by ordering i_2,...,i_D
                        # i.e., only keep polyads where keys_p < keys_q lexicographically
                        ordered_tetrad = np.all(keys_p < keys_q)
                        if ordered_tetrad:                           
                            # check and gather the values of the edges
                            # notice that we already know the values of 0->0..0 and 0->1..1 if D is odd
                            # and the values of 0->0..0 and 1->1..1 if D is even
                            m_xi, M_xi = _get_minima(
                                D, power_D, grid, signs, key, Y_xi,
                                keys_p, # i = [i_1, keys_p]
                                keys_q, # i' = [i_1p, keys_q]
                                value_p, # Y[i_1, keys_p]
                                value_q, # Y[i_1, keys_q] if D is odd else Y[i_1p, keys_q]
                                edge_indices[i_1], # edge_indices[i_1]
                                edge_indices[i_1p], # edge_indices[i_1p]
                                edge_values[i_1], # edge_values[i_1]
                                edge_values[i_1p] # edge_values[i_1p]
                            )

                            if (m_xi > 0) and (M_xi == 0 or (M_xi > 0 and i_1 < i_1p)):
                                xis[i_current_polyad] = np.array([[primary_indices[i_1], primary_indices[i_1p]]] + [[keys_p[l], keys_q[l]] for l in range(D-1)]) # D  x 2
                                Y_xis[i_current_polyad] = Y_xi
                                m_xis[i_current_polyad] = m_xi
                                M_xis[i_current_polyad] = M_xi
                                i_current_polyad += 1
                                if i_current_polyad == int(max_n_polyads):
                                    return xis, Y_xis, m_xis, M_xis
    return xis[:i_current_polyad].copy(), Y_xis[:i_current_polyad].copy(), m_xis[:i_current_polyad].copy(), M_xis[:i_current_polyad].copy()


@jit(nopython=True)
def _compute_polyads_permutations(
    D: int,
    xis: np.ndarray
) -> np.ndarray:
    """
    Obtain all permutations (links) for polyads of dimension D.

    Parameters
    ----------
    D : int
        Polyad dimension.
    xis : np.ndarray
        Array of polyad indices (shape: [n_polyads, D, 2]).

    Returns
    -------
    np.ndarray
        Array of all polyad permutations (shape: [2**D * n_polyads, 2*D + 1]).
    """
    grid, _ = _generate_polyad_sign_patterns(D)
    num_polyads = xis.shape[0]
    polyad_permutations = np.empty( (2**D*num_polyads, 2*D + 1), dtype=np.int32 )
    i_perm = 0
    for i, polyad_index in enumerate(xis):
        for j in range(2**D):
            polyad_permutations[i_perm, 2*D] = i
            for d in range(D):
                polyad_permutations[i_perm, d] = polyad_index[d][1] + grid[j][d] * (polyad_index[d][0] - polyad_index[d][1])
                polyad_permutations[i_perm, D+d] = polyad_index[d][0] + grid[j][d] * (polyad_index[d][1] - polyad_index[d][0])
            i_perm += 1
    return polyad_permutations