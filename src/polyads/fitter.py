
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
    _compute_polyad_features, _evaluate_polyad_losses,
    _evaluate_loss, _compute_polyad_pairwise_covariance
)
from .polyad_generation import _find_active_polyads, _compute_polyads_permutations
from .utils import get_bar_description, _get_keys_values


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
    variance_threshold: float = 1.0,
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
        Maximum allowed step size for parameter updates (default: 0.5).
    use_tqdm : bool, optional
        Whether to display a progress bar using tqdm (default: False).
    loss : str, optional
        Loss function name. Supported: 'poisson_binary', 'poisson_multiclass', 'poisson_binary_multiclass'.
        (default: 'poisson_multiclass')
    max_n_polyads : int, optional
        Maximum number of polyads to consider (default: 1e7).
    variance_threshold : float, optional
        Threshold for variance approximation (default: 1.0). Ranges from 0 (never) to 1 (always).

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

    primary_indices, edge_indices, edge_values = _get_keys_values(df.values)
    # Find all active polyads in the data
    xis, Y_xis, m_xis, M_xis = _find_active_polyads(
        D, primary_indices, edge_indices, edge_values, int(max_n_polyads))
    # Compute feature matrix for all polyads
    X_xis = _compute_polyad_features(
        D, p, xis, eval_X, kwargs=eval_kwargs)

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

    if use_tqdm:
        progress_bar.set_description(get_bar_description(beta, num_polyads))

    iteration: int = 0
    while iteration < max_iter:
        iteration += 1

        # Evaluate loss, gradient, and Hessian for current parameters
        log_likelihoods, expectations, variances = _evaluate_polyad_losses(
            D, Y_xis, X_xis, m_xis, M_xis, beta, loss)
        loss_val, gradient, hessian = _evaluate_loss(p, X_xis, log_likelihoods, expectations, variances)        
        hessian_det = np.linalg.det(hessian)
        gradient_norm = np.linalg.norm(gradient)

        # Compute update direction (Newton or gradient step)
        if hessian_det > 1e-8:
            update_direction = np.linalg.solve(hessian, gradient)
        else:
            update_direction = gradient
        update_norm = np.linalg.norm(update_direction)

        # Limit step size if necessary
        update_direction = update_direction if update_norm <= max_step else update_direction/update_norm * max_step
        beta -= update_direction
        if use_tqdm:
            progress_bar.update(1)
            progress_bar.set_description(get_bar_description(beta, num_polyads, grad_norm=gradient_norm, det=hessian_det))
        
        # Check for convergence or stopping criteria
        if update_norm < tol or iteration == max_iter or hessian_det <= 1e-8:
            var = np.inf * np.ones((p, p))
            n_pairs = 0

            if hessian_det > 1e-8:
                if use_tqdm:
                    progress_bar.set_description(f"Finding all permutations of the {num_polyads} active polyads")

                # Compute covariance matrix for parameter estimates
                xi_permutations = _compute_polyads_permutations(D, xis)
                xi_permutations = xi_permutations[np.lexsort([xi_permutations[:,i] for i in range(D)])]
                permutation_group_ends = np.append(
                    1 + np.where(np.sum(np.abs(xi_permutations[1:,:D] - xi_permutations[:-1,:D]), axis=1) > 0)[0],
                    xi_permutations.shape[0])
                n_pairs = permutation_group_ends[1:] - permutation_group_ends[:-1]
                n_pairs = (n_pairs * (n_pairs + 1) // 2).sum()
                n_pairs += permutation_group_ends[0] * (permutation_group_ends[0] + 1) // 2

                if use_tqdm:
                    progress_bar.set_description(f"Looping over {num_polyads} polyads to evaluate the variance")

                covariance_matrix = _compute_polyad_pairwise_covariance(
                    D, p, num_polyads,
                    xi_permutations, permutation_group_ends,
                    expectations, X_xis,
                    variance_threshold = variance_threshold
                )
                inv_hessian = np.linalg.inv(hessian)
                var = inv_hessian @ covariance_matrix @ inv_hessian.T
                
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
