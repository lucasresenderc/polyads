import numpy as np
import pandas as pd


def generate_jochmans_panel(
    seed,
    design,
    n=50,
    m=50,
    psi0=1.0,
    prob_non_zero=0.01,
    return_full=False,
):
    """
    Jochmans (2015) Monte Carlo DGP for a two-way exponential model, eq. (2.7).

    y_ij = exp(x_ij * psi0) * alpha_i * gamma_j * epsilon_ij

    with x_ij ~ N(0, 1), alpha_i ~ LogN(0, 1), gamma_j ~ LogN(0, 1), and
    epsilon_ij drawn so that E[epsilon_ij] = 1 and Var[epsilon_ij] = sigma2_ij
    for one of Designs 1--5 in Section II.C.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n, m))
    alpha = np.exp(rng.normal(size=n))
    gamma = np.exp(rng.normal(size=m))
    mu = np.exp(x * psi0) * alpha[:, None] * gamma[None, :]

    if design == 1:
        sigma2 = np.ones((n, m))
    elif design == 2:
        sigma2 = 1.0 / mu
    elif design == 3:
        sigma2 = mu
    elif design == 4:
        sigma2 = 1.0 / mu**2
    elif design == 5:
        sigma2 = mu**2
    else:
        raise ValueError(f"design must be in 1..5, got {design}")

    log_var = np.log(1.0 + sigma2)
    log_mean = -0.5 * log_var
    eps = np.exp(rng.normal(log_mean, np.sqrt(log_var))) * (rng.uniform(0, 1, size=(n, m)) < prob_non_zero) / prob_non_zero
    y = mu * eps

    X = x[..., None].astype(np.float32)
    if return_full:
        return y, X, mu, alpha, gamma

    keys = np.where(y > 0)
    df = pd.DataFrame({"i_0": keys[0], "i_1": keys[1], "Y": y[keys].astype(np.int32)})
    return df, X


def generate_data(seed, n_ds, c, shape, beta, groups=None, return_full=False, y_mode="poisson"):
    """
    Generate synthetic data for polyad model.
    Allows for p-dimensional features (last axis of X is p).
    beta should be a vector of length p or a scalar.

    When y_mode='poisson' (default), Y is Poisson with intensity
    exp(linpred) (or Gamma-Poisson if shape is finite).

    When y_mode='bernoulli', Y is Bernoulli with
    P(Y=1) = sigmoid(linpred).
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

    if y_mode == "bernoulli":
        prob = 1 / (1 + np.exp(-linpred))
        Y = rng.binomial(1, prob)
    elif y_mode == "poisson":
        if shape == np.inf:
            lam = np.exp(linpred)
            Y = rng.poisson(lam)
        else:
            lam = rng.gamma(shape, np.exp(linpred)/shape)
            Y = rng.poisson(lam)
    else:
        raise ValueError("y_mode must be 'poisson' or 'bernoulli'")

    keys = np.where(Y > 0)
    values = Y[Y>0]
    
    df = pd.DataFrame({f"i_{i}": keys[i] for i in range(len(n_ds))})
    df["Y"] = values

    if return_full:
        return Y, X, linpred, thetas
    else:
        return df, X
