import numpy as np
from scipy.stats import norm
import pandas as pd
from .validations import _validate_fit_inputs
from .fitter import _fit_polyad_estimator

class PolyadEstimator:
    """
    Scikit-learn-like estimator for polyad-based statistical modeling.
    """
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        max_step: float = 1.0,
        loss: str = "poisson_multiclass",
        max_n_polyads: int = int(1e8),
        variance_threshold: float = 0.0,
        use_tqdm: bool = False
    ) -> None:
        """
        Initialize a PolyadEstimator instance.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of optimization iterations (default: 100).
        tol : float, optional
            Convergence tolerance for the optimization (default: 1e-4).
        max_step : float, optional
            Maximum step size for parameter updates (default: 0.5).
        loss : str, optional
            Loss function name. Supported: 'poisson_binary', 'poisson_multiclass'.
            (default: 'poisson_multiclass')
        max_n_polyads : int, optional
            Maximum number of polyads to generate/use (default: 1e7).
        variance_threshold : float, optional
            Threshold for variance approximation (default: 0.0). Ranges from 0 (never) to 1 (always).
        use_tqdm : bool, optional
            Whether to display a progress bar during fitting (default: False).

        Returns
        -------
        None
        """
        self.max_iter = max_iter
        self.tol = tol
        self.max_step = max_step
        self.loss = loss
        self.max_n_polyads = int(max_n_polyads)
        self.variance_threshold = variance_threshold
        self.use_tqdm = use_tqdm
        self.beta_ = None
        self.n_edges_ = None
        self.n_polyads_ = None
        self.n_pairs_ = None
        self.loss_ = None
        self.score_ = None
        self.hessian_ = None
        self.det_ = None
        self.var_ = None
        self.converged_ = None
        self.iterations_ = None
        self.time_ = None

    def _validate_input(self, df, beta_init, eval_X, X, loss, indices, values):
        supported_losses = ["poisson_binary", "poisson_multiclass", "poisson_binary_balanced"]
        return _validate_fit_inputs(df, beta_init, eval_X, X, loss, supported_losses, indices, values)

    @staticmethod
    def default_eval_X(matrix: np.ndarray) -> 'callable':
        """
        Returns an eval_X function for a given 3D matrix.
        """
        def eval_X(key):
            return matrix[tuple(key)]
        return eval_X

    def fit(
        self,
        df: pd.DataFrame,
        indices: list,
        values: str,
        beta_init: np.ndarray,
        eval_X: 'callable | None' = None,
        eval_kwargs: 'dict | None' = None,
        X: 'np.ndarray | None' = None,
        loss: str = None,
    ) -> 'PolyadEstimator':
        """
        Fit the polyad model to data.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns representing the observed data (typically columns: t, i, j, v).
        beta_init : np.ndarray or list or float
            Initial guess for the parameter vector (shape: (n_features,) or scalar).
        eval_X : callable or None, optional
            Feature extraction function. If None and X is provided, uses default_eval_X(X).
            Should accept a key (tuple or array) and return a feature vector.
        eval_kwargs : dict or None, optional
            Additional keyword arguments to pass to eval_X.
        X : np.ndarray or None, optional
            3D or 4D numpy array of features. If provided and eval_X is None, uses default_eval_X(X).

        Returns
        -------
        self : PolyadEstimator
            The fitted estimator instance (self).
        """

        if eval_X is None:
            if X is not None:
                eval_X = self.default_eval_X(X)
                if eval_kwargs is None:
                    eval_kwargs = {}
            else:
                raise ValueError("You must provide either eval_X or X (matrix)!")

        # Allow override of loss for this fit, else use self.loss
        loss_name = loss if loss is not None else self.loss

        # Validate all inputs before fitting
        df, beta_init = self._validate_input(df, beta_init, eval_X, X, loss_name, indices, values)

        result = _fit_polyad_estimator(
            beta_init,
            df,
            eval_X,
            eval_kwargs=eval_kwargs,
            max_iter=self.max_iter,
            tol=self.tol,
            max_step=self.max_step,
            use_tqdm=self.use_tqdm,
            loss=loss_name,
            max_n_polyads=self.max_n_polyads,
            variance_threshold=self.variance_threshold
        )
        self.beta_ = result['beta']
        self.n_edges_ = result['n_edges']
        self.n_polyads_ = result['n_polyads']
        self.n_pairs_ = result['n_pairs']
        self.loss_ = result['loss']
        self.score_ = result['score']
        self.hessian_ = result['hessian']
        self.det_ = result['det']
        self.var_ = result['var']
        self.converged_ = result['converged']
        self.iterations_ = result['iterations']
        self.time_ = result['time']
        return self

    def summary(self, alpha: float = 0.05) -> None:
        """
        Display a regression summary for a fitted PolyadEstimator, similar to statsmodels/pystats.
        Shows coefficient, std err, z, p-value, and confidence intervals.
        """
        if self.beta_ is None or self.var_ is None:
            print("Model is not fitted or variance is not available.")
            return
        beta = np.asarray(self.beta_)
        var = np.asarray(self.var_)
        se = np.sqrt(np.diag(var))
        z = beta / se
        p = 2 * (1 - norm.cdf(np.abs(z)))
        ci_low = beta + norm.ppf(alpha/2) * se
        ci_upp = beta + norm.ppf(1 - alpha/2) * se
        df = pd.DataFrame({
            'coef': beta,
            'std err': se,
            'z': z,
            'P>|z|': p,
            f'[{100*alpha/2:.1f}%': ci_low,
            f'{100*(1-alpha/2):.1f}%]': ci_upp
        })
        print("="*65)
        print("\t"*3 + "Polyad Fit Results")
        print("="*65)

        if self.n_polyads_ == 0:
            print(f"No polyads found. No optimization step was taken.")
        else:
            print(f"Converged: {self.converged_}")
            print(f"Iterations: {self.iterations_}")
            print(f"Time: {self.time_:.2f} seconds")
            print(f"Number of Positive Edges: {self.n_edges_}")
            print(f"Number of Active Polyads: {self.n_polyads_}")
            if self.n_polyads_ == self.max_n_polyads:
                print("(Maximum number of polyads reached. Results may be unreliable.)")

            # Provide statistics when the model reaches a singular Hessian
            if self.det_ <= 1e-8 and self.n_polyads_ > 0:
                print("="*65)
                print(df)
                print("="*65)
                print("The optimization procedure encountered a singular Hessian,")
                print("its eigenvalues and eigenvectors are:")
                eigvals, eigvecs = np.linalg.eigh(self.hessian_)
                eigvals *= (np.abs(eigvals) > 1e-8)
                eigvecs *= (np.abs(eigvecs) > 1e-8)
                df = pd.DataFrame({
                    "eigenvalue": eigvals,
                    "eigenvector": [eigvecs[i] for i in range(eigvecs.shape[0])],
                })
                df.sort_values(by="eigenvalue", inplace=True)
                print(df)
                print("A singular Hessian may be associated with collinear variables.")
                print("It can also be associated with ill-defined models, e.g.,")
                print("one feature do not vary and is absorbed by the fixed effects.")
                print("The eigenvectors of zero may explain both cases.")
            else:
                print(f"Number of Pairs of Active Polyads Sharing Edges: {self.n_pairs_}")
                print("="*65)
                print(df)


