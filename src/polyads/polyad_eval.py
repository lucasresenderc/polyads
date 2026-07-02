import numpy as np
from numba import jit
from .losses import _compute_polyad_loss, _compute_polyad_gmm_moment
from .polyad_utils import _generate_polyad_sign_patterns


def _compute_polyad_features(
    D: int,
    p: int,
    xis: np.ndarray,
    X,
    args: tuple = None,
    kwargs: dict = None
) -> np.ndarray:
    """
    Build the antisymmetric feature matrix X_xis = S_plus - S_minus.
    """
    S_plus, S_minus = _compute_polyad_features_gmm(D, p, xis, X, args=args, kwargs=kwargs)
    return S_plus - S_minus


def _compute_polyad_features_gmm(
    D: int,
    p: int,
    xis: np.ndarray,
    X,
    args: tuple = None,
    kwargs: dict = None
) -> tuple:
    """
    Build the sign-summed feature matrices for the GMM moment.

    S_plus collects the features of the positive-sign configurations and
    S_minus those of the negative-sign configurations. The antisymmetric
    instrument is X_xis = S_plus - S_minus.

    Returns
    -------
    tuple
        (S_plus, S_minus), each of shape [n_polyads, p].
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    num_polyads = xis.shape[0]
    S_plus = np.zeros((num_polyads, p), dtype=np.float32)
    S_minus = np.zeros((num_polyads, p), dtype=np.float32)
    grid, signs = _generate_polyad_sign_patterns(D)
    key = np.empty(D, dtype=np.int32)
    power_D = 2 ** D
    for i_polyad, polyad_index in enumerate(xis):
        for i in range(power_D):
            for d in range(D):
                key[d] = polyad_index[d][grid[i][d]]
            X_config = np.asarray(X(key, *args, **kwargs), dtype=np.float32)
            if signs[i] > 0:
                S_plus[i_polyad] += X_config
            else:
                S_minus[i_polyad] += X_config
    return S_plus, S_minus


@jit(nopython=True)
def _evaluate_polyad_losses(
    D: int,
    Y_xis: np.ndarray,
    X_xis: np.ndarray,
    m_xis: np.ndarray,
    M_xis: np.ndarray,
    beta: np.ndarray,
    loss: str
) -> tuple:
    """
    Evaluate all polyads at a given parameter vector beta.

    Parameters
    ----------
    D : int
        Polyad dimension.
    Y_xis : np.ndarray
        Matrix of observed counts for each polyad (shape: [n_polyads, 2**D]).
    X_xis : np.ndarray
        Feature matrix for all polyads (shape: [n_polyads, p]).
    m_xis : np.ndarray
        Minimum positive counts for each polyad.
    M_xis : np.ndarray
        Minimum negative counts for each polyad.
    beta : np.ndarray
        Parameter vector.
    loss : str
        Loss function name. Supported: 'multiclass', 'binary', 'binary_balanced'

    Returns
    -------
    tuple
        (log_likelihoods, expectations, variances) for all polyads.
    """
    num_polyads = m_xis.size
    log_likelihoods = np.empty(num_polyads)
    expectations = np.empty(num_polyads)
    variances = np.empty(num_polyads)
    _, signs = _generate_polyad_sign_patterns(D)
    for i in range(num_polyads):
        log_likelihood, expectation, variance = _compute_polyad_loss(
            Y_xis[i], m_xis[i], M_xis[i], signs, X_xis[i] @ beta, loss
        )
        log_likelihoods[i] = log_likelihood
        expectations[i] = expectation
        variances[i] = variance
    return log_likelihoods, expectations, variances


@jit(nopython=True)
def _evaluate_polyad_gmm(
    D: int,
    Y_xis: np.ndarray,
    S_plus: np.ndarray,
    S_minus: np.ndarray,
    X_xis: np.ndarray,
    M_xis: np.ndarray,
    beta: np.ndarray,
    leveled: bool,
) -> tuple:
    """
    Evaluate the aggregated GMM moment vector and its Jacobian at beta.

    s(beta) = mean_i X_xis[i] * g_i(beta)            (p-vector)
    G(beta) = mean_i outer(X_xis[i], dg_i(beta))     (p x p Jacobian)

    Returns
    -------
    tuple
        (s, G, g_is) where g_is holds the per-polyad scalar moment (used by
        the variance computation).
    """
    num_polyads = M_xis.size
    p = beta.size
    _, signs = _generate_polyad_sign_patterns(D)
    power_D = 2 ** D

    s = np.zeros(p)
    G = np.zeros((p, p))
    g_is = np.empty(num_polyads)

    for i in range(num_polyads):
        has_B = M_xis[i] != 0
        logA = 0.0
        logB = 0.0
        for c in range(power_D):
            if signs[c] > 0:
                logA += np.log(Y_xis[i, c])
            elif has_B:
                logB += np.log(Y_xis[i, c])

        g, dg = _compute_polyad_gmm_moment(
            logA, logB, has_B, S_plus[i], S_minus[i], beta, leveled
        )
        g_is[i] = g

        pw = i / (i + 1)
        cw = 1 / (i + 1)
        s = s * pw + X_xis[i] * g * cw
        G = G * pw + (X_xis[i][:, None] * dg[None, :]) * cw

    return s, G, g_is


@jit(nopython=True)
def _compute_polyad_pairwise_covariance(
    D: int,
    p: int,
    num_polyads: int,
    polyad_permutations: np.ndarray,
    permutation_group_ends: np.ndarray,
    expectations: np.ndarray,
    X_xis: np.ndarray,
    variance_threshold: float,
) -> np.ndarray:
    """
    Compute covariance matrix for polyad parameter estimates.

    Parameters
    ----------
    D : int
        Polyad dimension.
    p : int
        Features dimension.
    polyad_permutations : np.ndarray
        Array of all polyad permutations (shape: [n_permutations, 2*D + 1]).
    permutation_group_ends : np.ndarray
        Indices marking the end of each permutation group.
    expectations : np.ndarray
        Expected values for each polyad.
    X_xis : np.ndarray
        Feature matrix for all polyads (shape: [n_polyads, p]).
    variance_threshold : float
        Threshold for variance approximation.

    Returns
    -------
    np.ndarray
        Covariance matrix (shape: [p, p]).
    """
    covariance_matrix = np.zeros((p, p))
    start = 0
    for i, end in enumerate(permutation_group_ends):
        sum_probs = 0
        approximate = True
        if variance_threshold == 0:
            approximate = False
        if variance_threshold > 0 and variance_threshold < 1:
            for d in range(D):
                probs = np.sort(polyad_permutations[start:end, D+d])
                # get the sum of squared counts of each unique value without numpy functions
                sum_probs = 0
                current_val = probs[0]
                count = 1
                for val in probs[1:]:
                    if val == current_val:
                        count += 1
                    else:
                        sum_probs += count**2
                        current_val = val
                        count = 1
                sum_probs += count**2
                if sum_probs / (end - start)**2 > variance_threshold * D:
                    approximate = False
                    break
        
        if approximate:
            grad_term = np.zeros(p)
            for i_p in range(start, end):
                grad_term += X_xis[polyad_permutations[i_p, 2*D]] * expectations[polyad_permutations[i_p, 2*D]]
            covariance_matrix = i/(i+1) * covariance_matrix + 1/(i+1) * grad_term[None, :] * grad_term[:, None]
        else:
            cov_term = np.zeros((p,p))
            for i_p in range(start, end):
                # add it with itself
                cov_term += (X_xis[polyad_permutations[i_p, 2*D]][None, :] * X_xis[polyad_permutations[i_p, 2*D]][:, None]) * (expectations[polyad_permutations[i_p, 2*D]] * expectations[polyad_permutations[i_p, 2*D]]) / 2**D
                for i_pp in range(i_p + 1, end):
                    # count twice the distance because we are only looping over half the matrix
                    w = expectations[polyad_permutations[i_p, 2*D]] * expectations[polyad_permutations[i_pp, 2*D]] / 2 ** np.sum(polyad_permutations[i_p, D:2*D] == polyad_permutations[i_pp, D:2*D])
                    cov_term += w * (X_xis[polyad_permutations[i_p, 2*D]][None, :] * X_xis[polyad_permutations[i_pp, 2*D]][:, None])
                    cov_term += w * (X_xis[polyad_permutations[i_pp, 2*D]][None, :] * X_xis[polyad_permutations[i_p, 2*D]][:, None])
            covariance_matrix = i/(i+1) * covariance_matrix + 1/(i+1) * cov_term

        start = end
    
    return (len(permutation_group_ends)/num_polyads**2) * covariance_matrix


@jit(nopython=True)
def _evaluate_loss(
    p : int,
    X_xis: np.ndarray,
    log_likelihoods: np.ndarray,
    expectations: np.ndarray,
    variances: np.ndarray
) -> tuple:
    """
    Evaluate quadratic loss, gradient, and Hessian for the polyad model.

    Parameters
    ----------
    p : int
        Features dimension.
    X_xis : np.ndarray
        Feature matrix for all polyads (shape: [n_polyads, p]).
    log_likelihoods : np.ndarray
        Log-likelihoods for each polyad.
    expectations : np.ndarray
        Expected values for each polyad.
    variances : np.ndarray
        Variances for each polyad.

    Returns
    -------
    tuple
        (loss, gradient, hessian) for the model.
    """
    loss = 0
    gradient = np.zeros(p)
    hessian = np.zeros((p,p))
    for i, features in enumerate(X_xis):
        pw = i/(i + 1)
        cw = 1/(i + 1)
        loss = loss * pw + log_likelihoods[i] * cw
        gradient = gradient * pw + expectations[i] * features * cw
        hessian = hessian * pw + variances[i] * (features[None, :] * features[:, None]) * cw
    return loss, gradient, hessian
