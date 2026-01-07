import numpy as np
import pandas as pd

def generate_data(seed, n_ds, c, shape, beta, groups=None, return_full = False):
    """
    Generate synthetic data for polyad model.
    Allows for p-dimensional features (last axis of X is p).
    beta should be a vector of length p or a scalar.
    """
    rng = np.random.default_rng(seed)
    D = len(n_ds)

    # beta: shape (p,) or scalar
    beta = np.asarray(beta)
    if beta.ndim == 0:
        beta = beta[None]
    p = beta.size

    X = rng.normal(size=(*n_ds, p))
    # for i in range(1, n_ds[-1]):
    #     X[:,:,i,-1] = X[:,:,0,-1] # to force a singular Hessian
    
    # Compute linear predictor
    linpred = c + np.tensordot(X, beta, axes=([-1],[0]))

    # Fill the null groups
    if groups is None:
        groups = []
        for d in range(D):
            groups.append( [d_p for d_p in range(D) if d_p != d] )
    
    # Add the fixed effects
    thetas = []
    for g in groups:
        theta_g = rng.normal(scale = .25, size = tuple( n_ds[d] for d in g ))
        thetas.append(theta_g)
        linpred += theta_g[tuple(slice(None) if d in g else np.newaxis for d in range(D))]

    if shape == np.inf:
        lam = np.exp(linpred)
    else:
        lam = rng.gamma(shape, np.exp(linpred)/shape)
    Y = rng.poisson(lam)

    keys = np.where(Y > 0)
    values = Y[Y>0]
    
    df = pd.DataFrame({f"i_{i}": keys[i] for i in range(len(n_ds))})
    df["Y"] = values

    if return_full:
        return Y, X, linpred, thetas
    else:
        return df, X
