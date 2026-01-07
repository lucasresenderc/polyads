import numpy as np
from numba import jit

@jit(nopython=True)
def _compute_polyad_loss(
    Y_xis: np.ndarray,
    m_xis: int,
    M_xis: int,
    signs: np.ndarray,
    c: float,
    loss: str
) -> tuple:
    """
    Evaluate the distribution for a given polyad.

    Parameters
    ----------
    Y_xi : np.ndarray
        Array of observed counts for each configuration of the polyad.
    min_positive_count : int
        Minimum count among positive-sign configurations.
    min_negative_count : int
        Minimum count among negative-sign configurations.
    signs : np.ndarray
        Array of +1/-1 signs for each configuration.
    c : float
        Linear predictor (dot product of features and parameters).
    loss : str
        Loss function name. Supported: 'poisson_binary', 'poisson_multiclass', 'poisson_binary_multiclass'.

    Returns
    -------
    tuple
        (loss, expectation, variance) for the polyad.
    """
    Y_0 = Y_xis - signs * m_xis
    nl = np.arange(m_xis + M_xis + 1)
    W = np.empty(m_xis + M_xis + 1)
    W[0] = 0
    for i in range(m_xis + M_xis):
        W[i+1] = W[i] - np.sum( signs * np.log(Y_0 + i * signs + (signs+1)/2) ) + c
    W -= np.max(W)
    ps = np.exp(W)

    if loss == "poisson_multiclass":
        err = np.log(ps.sum()) - W[m_xis]
        ps /= ps.sum()
        exp = (nl * ps).sum()
        var = ((nl**2) * ps).sum() - exp**2
        return err, exp - m_xis, var
    
    else:
        err_t = np.log(ps.sum())
        ps_t = ps/ps.sum()
        exp_t = (nl * ps_t).sum()
        var_t = ((nl**2) * ps_t).sum() - exp_t**2

        if loss == "poisson_binary_balanced":
            cut_m = 1
            ps_under_cut = ps_t[0]
            while ps_under_cut < .5 and cut_m < m_xis + M_xis:
                ps_under_cut += ps_t[cut_m]
                cut_m += 1
        else:
            cut_m = (m_xis + M_xis + 1)//2

        if m_xis >= cut_m:
            err_s = np.log(ps[cut_m:].sum())
            ps_s = ps[cut_m:]/ps[cut_m:].sum()
            exp_s = (nl[cut_m:] * ps_s).sum()
            var_s = ((nl[cut_m:]**2) * ps_s).sum() - exp_s**2
        else:
            err_s = np.log(ps[:cut_m].sum())
            ps_s = ps[:cut_m]/ps[:cut_m].sum()
            exp_s = (nl[:cut_m] * ps_s).sum()
            var_s = ((nl[:cut_m]**2) * ps_s).sum() - exp_s**2
        return err_t - err_s, exp_t - exp_s, var_t - var_s