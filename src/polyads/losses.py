import numpy as np
from numba import jit

LOSS_REGISTRY = {
    "poisson_multiclass": {
        "family": "qmle",
        "kernel": "multiclass",
        "extensive_margin": False,
        "dtype": np.int32,
    },
    "poisson_binary": {
        "family": "qmle",
        "kernel": "binary",
        "extensive_margin": False,
        "dtype": np.int32,
    },
    "poisson_binary_balanced": {
        "family": "qmle",
        "kernel": "binary_balanced",
        "extensive_margin": False,
        "dtype": np.int32,
    },
    "bernoulli": {
        "family": "qmle",
        "kernel": "multiclass",
        "extensive_margin": True,
        "dtype": np.int32,
    },
    "gmm_unleveled": {
        "family": "gmm",
        "leveled": False,
        "extensive_margin": False,
        "dtype": np.float64,
    },
    "gmm_leveled": {
        "family": "gmm",
        "leveled": True,
        "extensive_margin": False,
        "dtype": np.float64,
    },
}
SUPPORTED_LOSSES = tuple(LOSS_REGISTRY.keys())


def resolve_loss(loss: str) -> dict:
    if loss not in LOSS_REGISTRY:
        raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}, got '{loss}'")
    return LOSS_REGISTRY[loss]


def is_gmm(loss: str) -> bool:
    return resolve_loss(loss)["family"] == "gmm"


def qmle_kernel(loss: str) -> str:
    return resolve_loss(loss)["kernel"]


def extensive_margin(loss: str) -> bool:
    return resolve_loss(loss)["extensive_margin"]


def value_dtype(loss: str):
    return resolve_loss(loss)["dtype"]


@jit(nopython=True)
def _compute_polyad_gmm_moment(
    logA: float,
    logB: float,
    has_B: bool,
    S_plus_i: np.ndarray,
    S_minus_i: np.ndarray,
    beta: np.ndarray,
    leveled: bool,
) -> tuple:
    """
    Evaluate the scalar GMM moment of a polyad and its gradient in beta.

    The polyad moment bracket is, with A = prod_{+} Y_i, B = prod_{-} Y_i,
    S_plus = sum_{+} X_i and S_minus = sum_{-} X_i:

        unleveled: g  = A exp(-S_plus.beta)  - B exp(-S_minus.beta)
        leveled:   g  = A exp( S_minus.beta) - B exp( S_plus.beta)

    The leveled form is (3.1) of Jochmans (2015): it multiplies the unleveled
    bracket by exp(sum_all X_i.beta) and stays well behaved when the regressors
    are non-negative. A and B are passed in log space (logA, logB); when the
    negative-sign product is absent (has_B is False) B = 0.

    Returns
    -------
    tuple
        (g, dg) where g is the scalar moment and dg its gradient (shape [p]).
    """
    sp = S_plus_i @ beta
    sm = S_minus_i @ beta

    if leveled:
        term_A = np.exp(logA + sm)
        g = term_A
        dg = S_minus_i * term_A
        if has_B:
            term_B = np.exp(logB + sp)
            g -= term_B
            dg -= S_plus_i * term_B
    else:
        term_A = np.exp(logA - sp)
        g = term_A
        dg = -S_plus_i * term_A
        if has_B:
            term_B = np.exp(logB - sm)
            g -= term_B
            dg += S_minus_i * term_B

    return g, dg


@jit(nopython=True)
def _compute_polyad_loss(
    Y_xis: np.ndarray,
    m_xis: float,
    M_xis: float,
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
        QMLE kernel name. Supported: 'multiclass', 'binary', 'binary_balanced'.

    Returns
    -------
    tuple
        (loss, expectation, variance) for the polyad.
    """
    Y_0 = Y_xis - signs * m_xis
    n_levels = int(m_xis + M_xis + 1)
    nl = np.arange(n_levels)
    W = np.empty(n_levels)
    W[0] = 0
    for i in range(int(m_xis + M_xis)):
        W[i+1] = W[i] - np.sum( signs * np.log(Y_0 + i * signs + (signs+1)/2) ) + c
    W -= np.max(W)
    ps = np.exp(W)

    if loss == "multiclass":
        err = np.log(ps.sum()) - W[int(m_xis)]
        ps /= ps.sum()
        exp = (nl * ps).sum()
        var = ((nl**2) * ps).sum() - exp**2
        return err, exp - m_xis, var
    
    else:
        err_t = np.log(ps.sum())
        ps_t = ps/ps.sum()
        exp_t = (nl * ps_t).sum()
        var_t = ((nl**2) * ps_t).sum() - exp_t**2

        if loss == "binary_balanced":
            cut_m = 1
            ps_under_cut = ps_t[0]
            while ps_under_cut < .5 and cut_m < int(m_xis + M_xis):
                ps_under_cut += ps_t[cut_m]
                cut_m += 1
        else:
            cut_m = int(m_xis + M_xis + 1) // 2

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