"""
Monte Carlo comparison of PPML (binary / multiclass) and GMM (unleveled / leveled)
on the three-way Poisson DGP from tests/basic_usage.py.
"""
import numpy as np
from src.polyads.data import generate_data
from src.polyads.model import PolyadEstimator

BETA_TRUE = np.array([1.0, -0.5], dtype=np.float32)
GROUPS = [[0, 1], [0, 2], [1, 2]]
ESTIMATORS = (
    "poisson_multiclass",
    "poisson_binary",
    "gmm_unleveled",
    "gmm_leveled",
)
N_REP = 100


def dgp_params(n_side=40, c=-5, shape=np.inf):
    return dict(
        n_ds=(n_side, n_side, n_side),
        c=c,
        shape=shape,
        beta=BETA_TRUE,
        groups=GROUPS,
    )


def generate_panel(seed, n_side=40, c=-5, shape=np.inf):
    return generate_data(seed=seed, **dgp_params(n_side=n_side, c=c, shape=shape))


def fit_once(df, X, beta_init, loss, warmup=False):
    cols = df.columns.tolist()
    est = PolyadEstimator(
        max_n_polyads=10 if warmup else int(1e8),
        use_tqdm=False,
        loss=loss,
    )
    est.fit(df, cols[:-1], cols[-1], beta_init, X=X)
    return est


def fit_estimator(df, X, beta_init, loss):
    fit_once(df, X, beta_init, loss, warmup=True)
    return fit_once(df, X, beta_init, loss, warmup=False)


def safe_se(var):
    se = np.sqrt(np.diag(var))
    se[~np.isfinite(se)] = np.nan
    return se


def monte_carlo(n_rep=N_REP, n_side=40, c=-5, beta_init=None, estimators=ESTIMATORS):
    if beta_init is None:
        beta_init = np.zeros(BETA_TRUE.size, dtype=np.float32)

    estimates = {name: [] for name in estimators}
    ses = {name: [] for name in estimators}
    converged = {name: [] for name in estimators}
    n_edges = []
    n_polyads = {name: [] for name in estimators}

    for rep in range(n_rep):
        df, X = generate_panel(seed=rep + 1, n_side=n_side, c=c)
        n_edges.append(len(df))

        for name in estimators:
            est = fit_estimator(df, X, beta_init, name)
            estimates[name].append(est.beta_.copy())
            ses[name].append(safe_se(est.var_))
            converged[name].append(est.converged_)
            n_polyads[name].append(est.n_polyads_)

        if (rep + 1) % max(1, n_rep // 10) == 0:
            print(f"  {rep + 1}/{n_rep}", flush=True)

    out = {}
    z = 1.96
    for name in estimators:
        ests = np.array(estimates[name])
        se_est = np.array(ses[name])
        ci_lo = ests - z * se_est
        ci_hi = ests + z * se_est
        out[name] = {
            "mean": ests.mean(axis=0),
            "sd": ests.std(axis=0, ddof=1),
            "bias": ests.mean(axis=0) - BETA_TRUE,
            "rmse": np.sqrt(((ests - BETA_TRUE) ** 2).mean(axis=0)),
            "ci": np.mean((ci_lo <= BETA_TRUE) & (BETA_TRUE <= ci_hi), axis=0),
            "n_conv": int(np.sum(converged[name])),
            "mean_polyads": float(np.mean(n_polyads[name])),
        }
    out["_panel"] = {"mean_edges": float(np.mean(n_edges))}
    return out


def print_results(results, estimators=ESTIMATORS):
    panel = results.pop("_panel", {})
    if panel:
        print(f"Mean positive edges per panel: {panel['mean_edges']:.0f}")
    print(f"\nTrue beta = {BETA_TRUE}")
    print(
        f"{'Estimator':<22} | "
        f"{'b0 mean':>7} {'sd':>6} {'ci':>5} | "
        f"{'b1 mean':>7} {'sd':>6} {'ci':>5} | "
        f"{'conv':>4} {'polyads':>8}"
    )
    print("-" * 88)
    for name in estimators:
        r = results[name]
        print(
            f"{name:<22} | "
            f"{r['mean'][0]:7.3f} {r['sd'][0]:6.3f} {r['ci'][0]:5.3f} | "
            f"{r['mean'][1]:7.3f} {r['sd'][1]:6.3f} {r['ci'][1]:5.3f} | "
            f"{r['n_conv']:4d} {r['mean_polyads']:8.0f}"
        )

    print(
        f"\n{'Estimator':<22} | "
        f"{'b0 bias':>7} {'rmse':>6} | "
        f"{'b1 bias':>7} {'rmse':>6}"
    )
    print("-" * 52)
    for name in estimators:
        r = results[name]
        print(
            f"{name:<22} | "
            f"{r['bias'][0]:7.3f} {r['rmse'][0]:6.3f} | "
            f"{r['bias'][1]:7.3f} {r['rmse'][1]:6.3f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rep", type=int, default=N_REP)
    parser.add_argument("--n-side", type=int, default=40, help="Cube side length (basic_usage uses 100)")
    parser.add_argument("--c", type=float, default=-5)
    parser.add_argument("--pilot", action="store_true", help="Single replication, print all fits")
    args = parser.parse_args()

    beta_init = np.zeros(BETA_TRUE.size, dtype=np.float32)

    if args.pilot:
        df, X = generate_panel(seed=0, n_side=args.n_side, c=args.c)
        cols = df.columns.tolist()
        print(
            f"panel {args.n_side}^3, positive edges: {len(df)} / {args.n_side ** 3}, "
            f"c={args.c}, beta_true={BETA_TRUE}"
        )
        for loss in ESTIMATORS:
            est = fit_estimator(df, X, beta_init, loss)
            print(f"\n--- {loss} ---")
            print(f"beta = {est.beta_}, converged = {est.converged_}, polyads = {est.n_polyads_}")
    else:
        print(
            f"Monte Carlo: {args.n_rep} reps, n_side={args.n_side}, c={args.c}",
            flush=True,
        )
        results = monte_carlo(
            n_rep=args.n_rep,
            n_side=args.n_side,
            c=args.c,
            beta_init=beta_init,
        )
        print_results(results)
