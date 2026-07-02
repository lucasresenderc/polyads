import numpy as np
import pandas as pd
from .losses import SUPPORTED_LOSSES, resolve_loss


def _validate_fit_inputs(df, beta_init, eval_X, X, loss, index_columns, value_column):
    """
    Validate inputs for PolyadEstimator.fit.
    Raises ValueError or TypeError with clear messages if invalid.
    """
    cfg = resolve_loss(loss)

    # DataFrame check
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(index_columns, list) or not index_columns:
        raise ValueError("index_columns must be a non-empty list or tuple of column names.")
    if len(index_columns) < 2:
        raise ValueError("index_columns must have at least 2 columns.")
    all_columns = set(df.columns)
    if not set(index_columns).issubset(all_columns):
        raise ValueError(f"Some index_columns {index_columns} are not in DataFrame columns {list(df.columns)}.")
    if value_column not in all_columns:
        raise ValueError(f"value_column '{value_column}' is not in DataFrame columns {list(df.columns)}.")
    if value_column in index_columns:
        raise ValueError("value_column cannot be one of the index_columns.")

    # Keep only relevant columns
    df = df[list(index_columns) + [value_column]].copy()

    # Check for NaN or infinite values and negative values
    if df.isnull().values.any():
        raise ValueError("df contains NaN values.")
    if not np.isfinite(df.values).all():
        raise ValueError("df contains infinite values.")
    if (df[value_column] < 0).any():
        raise ValueError("df contains negative values.")

    values = df[value_column].values

    if cfg["family"] == "qmle" and cfg["extensive_margin"]:
        if not np.all(np.equal(np.mod(values, 1), 0)):
            raise ValueError("bernoulli loss requires integer-valued outcomes.")
        if (values > 1).any():
            raise ValueError("bernoulli loss requires outcomes in {0, 1}.")
    elif cfg["dtype"] == np.int32:
        if not np.all(np.equal(np.mod(values, 1), 0)):
            raise ValueError("df contains non-integer values.")
    elif cfg["family"] == "gmm":
        pass

    # Only keep rows with nonzero values in the value column
    df = df[df[value_column] != 0].copy()
    df[value_column] = df[value_column].astype(cfg["dtype"])

    # beta_init check
    if np.isscalar(beta_init):
        beta_init = np.array([beta_init], dtype=np.float32)
    elif isinstance(beta_init, list):
        beta_init = np.array(beta_init, dtype=np.float32)
    elif isinstance(beta_init, np.ndarray):
        beta_init = np.array(beta_init, dtype=np.float32)
    else:
        raise TypeError("beta_init must be a numpy array, list, or scalar.")
    if beta_init.ndim != 1:
        raise ValueError("beta_init must be a 1D vector.")

    # X check
    if X is not None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray if provided.")
        if np.any(np.isnan(X)) or np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")

    # eval_X check
    if eval_X is not None and not callable(eval_X):
        raise TypeError("eval_X must be callable if provided.")

    if loss not in SUPPORTED_LOSSES:
        raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}, got '{loss}'")

    return df, beta_init
