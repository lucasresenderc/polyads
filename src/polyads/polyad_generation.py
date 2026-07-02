import numpy as np
from numba import jit
from .binary_search import _binary_search_edge_value
from .polyad_utils import _generate_polyad_sign_patterns


@jit(nopython=True)
def _get_minima(
        D, power_D, grid, signs, key, Y_xi,
        keys_p,
        keys_q,
        value_p,
        value_q,
        edge_indices_i_1,
        edge_indices_i_1p,
        edge_values_i_1,
        edge_values_i_1p,
        extensive_margin,
    ) -> tuple:

    m_xi = min(value_p, value_q)
    for i in range(power_D//2, power_D):
        if signs[i] == 1:
            if (D%2 == 0 and i == power_D-1):
                Y_xi[i] = value_q
            else:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1p, edge_values_i_1p, key)
                if Y_xi[i] == 0:
                    return 0.0, 0.0
                m_xi = min(m_xi, Y_xi[i])

    for i in range(1, power_D//2):
        if signs[i] == 1:
            if (D%2 == 1 and i == power_D//2-1):
                Y_xi[i] = value_q
            else:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1, edge_values_i_1, key)
                if Y_xi[i] == 0:
                    return 0.0, 0.0
                m_xi = min(m_xi, Y_xi[i])

    Y_xi[0] = value_p

    if extensive_margin:
        M_xi = 0.0
        for i in range(1, power_D//2):
            if signs[i] == -1:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1, edge_values_i_1, key)
                M_xi = max(M_xi, Y_xi[i])
                if M_xi > 0:
                    return m_xi, M_xi

        for i in range(power_D//2, power_D):
            if signs[i] == -1:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1p, edge_values_i_1p, key)
                M_xi = max(M_xi, Y_xi[i])
                if M_xi > 0:
                    return m_xi, M_xi
    else:
        M_xi = -1.0
        for i in range(1, power_D//2):
            if signs[i] == -1:
                for d in range(D-1):
                    key[d] = keys_p[d] if grid[i][d+1] == 0 else keys_q[d]
                Y_xi[i] = _binary_search_edge_value(edge_indices_i_1, edge_values_i_1, key)
                M_xi = min(M_xi, Y_xi[i]) if M_xi != -1.0 else Y_xi[i]

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
    max_n_polyads: int,
    extensive_margin: bool = False,
) -> tuple:
    """
    Generate all active polyads for the model.
    """
    power_D = 2**D
    n_1 = primary_indices.size
    grid, signs = _generate_polyad_sign_patterns(D)
    key = np.empty(D-1, dtype=np.int32)
    Y_xi = np.empty(signs.size, dtype=np.float64)
    xis = np.empty((int(max_n_polyads), D, 2), dtype=np.int32)
    Y_xis = np.empty((int(max_n_polyads), signs.size), dtype=np.float64)
    m_xis = np.empty(int(max_n_polyads), dtype=np.float64)
    M_xis = np.empty(int(max_n_polyads), dtype=np.float64)

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

                        ordered_tetrad = np.all(keys_p < keys_q)
                        if ordered_tetrad:
                            m_xi, M_xi = _get_minima(
                                D, power_D, grid, signs, key, Y_xi,
                                keys_p,
                                keys_q,
                                value_p,
                                value_q,
                                edge_indices[i_1],
                                edge_indices[i_1p],
                                edge_values[i_1],
                                edge_values[i_1p],
                                extensive_margin,
                            )

                            if extensive_margin:
                                active = (m_xi > 0) and (M_xi == 0)
                            else:
                                active = (m_xi > 0) and (M_xi == 0 or (M_xi > 0 and i_1 < i_1p))
                            if active:
                                xis[i_current_polyad] = np.array([[primary_indices[i_1], primary_indices[i_1p]]] + [[keys_p[l], keys_q[l]] for l in range(D-1)])
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


def _compute_polyad_permutation_groups(D: int, xis: np.ndarray) -> tuple:
    """
    Sort polyad permutations into link groups and count within-group pairs.
    """
    xi_permutations = _compute_polyads_permutations(D, xis)
    xi_permutations = xi_permutations[np.lexsort([xi_permutations[:, i] for i in range(D)])]
    permutation_group_ends = np.append(
        1 + np.where(
            np.sum(np.abs(xi_permutations[1:, :D] - xi_permutations[:-1, :D]), axis=1) > 0
        )[0],
        xi_permutations.shape[0],
    )
    group_sizes = permutation_group_ends[1:] - permutation_group_ends[:-1]
    n_pairs = (group_sizes * (group_sizes + 1) // 2).sum()
    n_pairs += permutation_group_ends[0] * (permutation_group_ends[0] + 1) // 2
    return xi_permutations, permutation_group_ends, n_pairs
