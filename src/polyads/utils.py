import numpy as np


def get_bar_description(beta, num_polyads, grad_norm=None, det=None):
    """
    Return a string description for the progress bar.
    Shows beta[0], gradient norm, and hessian determinant if provided.
    """
    desc = f"{num_polyads} polyads | beta[0]: {beta[0]:.3f}"
    if grad_norm is not None:
        desc += f" | grad_norm: {grad_norm:.2e}"
    if det is not None:
        desc += f" | det: {det:.2e}"
    return desc


def _get_keys_values(Y):
    """Sort and extract keys and values from Y array."""
    D = Y.shape[1] - 1
    Y_sorted = Y[np.lexsort([Y[:,d] for d in range(D-1, -1, -1)])]

    i_1s = np.unique(Y_sorted[:,0])
    keys = [Y_sorted[ Y_sorted[:,0] == i_1 , 1:D] for i_1 in i_1s]
    values = [Y_sorted[ Y_sorted[:,0] == i_1 , D] for i_1 in i_1s]
    return i_1s, keys, values
