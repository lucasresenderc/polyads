
import warnings
try:
    from numba.core.errors import NumbaPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import time
from .polyad_eval import (
    _compute_polyad_features_gmm,
    _evaluate_polyad_losses, _evaluate_polyad_gmm,
    _evaluate_loss, _compute_polyad_pairwise_covariance
)
from .polyad_generation import _find_active_polyads, _compute_polyad_permutation_groups
from .utils import get_bar_description, _get_keys_values
from .losses import resolve_loss


def _newton_step(
    beta: np.ndarray,
    jacobian: np.ndarray,
    vector: np.ndarray,
    max_step: float,
    use_abs_det: bool = False,
) -> tuple:
    """
    Take one damped Newton step: beta <- beta - clip(G^{-1} vector, max_step).
    """
    det = np.linalg.det(jacobian)
    invertible = abs(det) > 1e-8 if use_abs_det else det > 1e-8
    if invertible:
        update_direction = np.linalg.solve(jacobian, vector)
    else:
        update_direction = vector
    update_norm = np.linalg.norm(update_direction)
    if update_norm > max_step:
        update_direction = update_direction / update_norm * max_step
        update_norm = max_step
    return beta - update_direction.astype(beta.dtype, copy=False), update_norm, det, invertible


def _sandwich_variance(
    D: int,
    p: int,
    num_polyads: int,
    xis: np.ndarray,
    plug_in: np.ndarray,
    X_xis: np.ndarray,
    jacobian: np.ndarray,
    variance_threshold: float,
) -> tuple:
    """
    Hájek-projection covariance sandwiched by the inverse Jacobian/Hessian.
    """
    xi_permutations, permutation_group_ends, n_pairs = _compute_polyad_permutation_groups(D, xis)
    covariance_matrix = _compute_polyad_pairwise_covariance(
        D, p, num_polyads,
        xi_permutations, permutation_group_ends,
        plug_in, X_xis,
        variance_threshold=variance_threshold,
    )
    inv_jacobian = np.linalg.inv(jacobian)
    var = inv_jacobian @ covariance_matrix @ inv_jacobian.T
    return var, n_pairs


# Main fitting routine for PolyadEstimator
def _fit_polyad_estimator(
    beta: np.ndarray,
    df: np.ndarray,
    eval_X: np.ndarray,
    eval_kwargs: dict = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    max_step: float = 1.0,
    use_tqdm: bool = False,
    loss: str = "poisson_multiclass",
    max_n_polyads: int = 1e7,
    variance_threshold: float = 0.0,
) -> dict:
    """
    Fit the polyad model using iterative optimization.

    Parameters
    ----------
    beta : np.ndarray
        Initial parameter vector for the model.
    df : np.ndarray
        Data array containing the observed data (primary and edge indices/values).
    eval_X : np.ndarray
        Feature matrix for evaluation.
    eval_kwargs : dict, optional
        Additional keyword arguments for feature evaluation (default: None).
    max_iter : int, optional
        Maximum number of optimization iterations (default: 100).
    tol : float, optional
        Tolerance for convergence (default: 1e-4).
    max_step : float, optional
        Maximum allowed step size for parameter updates (default: 1.0).
    use_tqdm : bool, optional
        Whether to display a progress bar using tqdm (default: False).
    loss : str, optional
        Loss function name. Supported: 'poisson_binary', 'poisson_multiclass',
        'poisson_binary_balanced', 'bernoulli', 'gmm_unleveled', 'gmm_leveled'.
        (default: 'poisson_multiclass')
    max_n_polyads : int, optional
        Maximum number of polyads to consider (default: 1e7).
    variance_threshold : float, optional
        Threshold for variance approximation (default: 0.0). Ranges from 0 (never) to 1 (always).

    Returns
    -------
    dict
        Dictionary containing the results of the optimization, including:
        - 'beta': Final parameter vector
        - 'converged': Whether convergence was achieved
        - 'loss': Final loss value
        - 'score': Gradient at solution
        - 'hessian': Hessian matrix at solution
        - 'det': Determinant of Hessian
        - 'iterations': Number of iterations performed
        - 'var': Estimated parameter covariance matrix
        - 'n_polyads': Number of polyads used
        - 'time': Total runtime in seconds
    """

    num_pos_edges = len(df)
    if use_tqdm:
        from tqdm import tqdm
        progress_bar = tqdm(total=max_iter, desc=f"Gathering polyads over {num_pos_edges}² pairs of edges")

    ts = time.time()
    D: int = len(df.columns) - 1  # Polyad dimension
    p: int = beta.size

    cfg = resolve_loss(loss)
    is_gmm = cfg["family"] == "gmm"

    primary_indices, edge_indices, edge_values = _get_keys_values(df.values)
    edge_values = [v.astype(np.float64) for v in edge_values]
    xis, Y_xis, m_xis, M_xis = _find_active_polyads(
        D, primary_indices, edge_indices, edge_values, int(max_n_polyads),
        extensive_margin=cfg["extensive_margin"])

    S_plus, S_minus = _compute_polyad_features_gmm(D, p, xis, eval_X, kwargs=eval_kwargs)
    X_xis = S_plus - S_minus

    num_polyads: int = m_xis.size

    if num_polyads == 0:
        if use_tqdm:
            progress_bar.close()
        return {
            "beta": beta,
            "converged": False,
            "loss": np.inf,
            "score": np.full(p, np.inf),
            "hessian": np.full((p, p), np.inf),
            "det": np.inf,
            "iterations": 0,
            "var": np.full((p, p), np.inf),
            "n_edges": num_pos_edges,
            "n_polyads": 0,
            "n_pairs": 0,
            "time": time.time() - ts
        }

    if is_gmm:
        leveled = cfg["leveled"]
        s = np.full(p, np.inf)
        G = np.full((p, p), np.inf)
        g_is = np.empty(num_polyads)
        G_det = np.inf
        s_norm = np.inf
        update_norm = np.inf

        iteration = 0
        while iteration < max_iter:
            iteration += 1

            s, G, g_is = _evaluate_polyad_gmm(
                D, Y_xis, S_plus, S_minus, X_xis, M_xis, beta, leveled)
            s_norm = np.linalg.norm(s)

            beta, update_norm, G_det, invertible = _newton_step(
                beta, G, s, max_step, use_abs_det=True)

            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_description(get_bar_description(beta, num_polyads, grad_norm=s_norm, det=G_det))

            if s_norm < tol or update_norm < tol or iteration == max_iter or not invertible:
                var = np.inf * np.ones((p, p))
                n_pairs = 0

                if invertible:
                    if use_tqdm:
                        progress_bar.set_description(
                            f"Finding all permutations of the {num_polyads} active polyads")
                    if use_tqdm:
                        progress_bar.set_description(
                            f"Looping over {num_polyads} polyads to evaluate the variance")

                    var, n_pairs = _sandwich_variance(
                        D, p, num_polyads, xis, g_is, X_xis, G, variance_threshold)

                if use_tqdm:
                    progress_bar.close()

                return {
                    "beta": beta,
                    "converged": bool(s_norm < tol or update_norm < tol),
                    "loss": s_norm**2,
                    "score": s,
                    "hessian": G,
                    "det": abs(G_det),
                    "iterations": iteration,
                    "var": var,
                    "n_edges": num_pos_edges,
                    "n_polyads": num_polyads,
                    "n_pairs": n_pairs,
                    "time": time.time() - ts
                }

    if use_tqdm:
        progress_bar.set_description(get_bar_description(beta, num_polyads))

    iteration: int = 0
    while iteration < max_iter:
        iteration += 1

        log_likelihoods, expectations, variances = _evaluate_polyad_losses(
            D, Y_xis, X_xis, m_xis, M_xis, beta, cfg["kernel"])
        loss_val, gradient, hessian = _evaluate_loss(p, X_xis, log_likelihoods, expectations, variances)
        gradient_norm = np.linalg.norm(gradient)

        beta, update_norm, hessian_det, invertible = _newton_step(
            beta, hessian, gradient, max_step, use_abs_det=False)

        if use_tqdm:
            progress_bar.update(1)
            progress_bar.set_description(
                get_bar_description(beta, num_polyads, grad_norm=gradient_norm, det=hessian_det))

        if update_norm < tol or iteration == max_iter or not invertible:
            var = np.inf * np.ones((p, p))
            n_pairs = 0

            if invertible:
                if use_tqdm:
                    progress_bar.set_description(
                        f"Finding all permutations of the {num_polyads} active polyads")
                if use_tqdm:
                    progress_bar.set_description(
                        f"Looping over {num_polyads} polyads to evaluate the variance")

                var, n_pairs = _sandwich_variance(
                    D, p, num_polyads, xis, expectations, X_xis, hessian, variance_threshold)

            if use_tqdm:
                progress_bar.close()

            return {
                "beta": beta,
                "converged": bool(update_norm < tol),
                "loss": loss_val,
                "score": gradient,
                "hessian": hessian,
                "det": hessian_det,
                "iterations": iteration,
                "var": var,
                "n_edges": num_pos_edges,
                "n_polyads": num_polyads,
                "n_pairs": n_pairs,
                "time": time.time() - ts
            }
