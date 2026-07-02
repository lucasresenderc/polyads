"""
Replicate Table 1 of Jochmans (2015), Section II.C.

Model: y_ij = exp(x_ij * psi0) * alpha_i * gamma_j * epsilon_ij
Panel size n = m = 50, psi0 = 1, Monte Carlo replications over Designs 1--5
and sparse edge probabilities prob_non_zero in {0.01, ..., 0.05}.
"""
import numpy as np
from src.polyads.data import generate_jochmans_panel
from src.polyads.model import PolyadEstimator

PSI0 = 1.0
N_REP = 1000
DESIGNS = (1, 2, 3, 4, 5)
PROB_NON_ZERO = (0.025, 0.05, .1, .15)


def fit_gmm(df, X, beta_init):
    cols = df.columns.tolist()
    warmup = PolyadEstimator(max_n_polyads=10, use_tqdm=False, loss="gmm_unleveled")
    warmup.fit(df, cols[:-1], cols[-1], beta_init, X=X)

    est = PolyadEstimator(use_tqdm=False, loss="gmm_unleveled")
    est.fit(df, cols[:-1], cols[-1], beta_init, X=X)
    return est


def monte_carlo(
    design,
    prob_non_zero,
    n_rep=N_REP,
    n=50,
    m=50,
    beta_init=None,
    estimators=("gmm",),
):
    if beta_init is None:
        beta_init = np.array([0.0], dtype=np.float32)

    estimates = {name: [] for name in estimators}
    ses = {name: [] for name in estimators}
    converged = {name: [] for name in estimators}

    for rep in range(n_rep):
        df, X = generate_jochmans_panel(
            seed=rep + 1000 * design + int(prob_non_zero * 1e6),
            design=design,
            n=n,
            m=m,
            prob_non_zero=prob_non_zero,
        )

        if "gmm" in estimators:
            est = fit_gmm(df, X, beta_init)
            estimates["gmm"].append(est.beta_[0])
            ses["gmm"].append(np.sqrt(est.var_[0, 0]))
            converged["gmm"].append(est.converged_)

        if (rep + 1) % max(1, n_rep // 10) == 0:
            print(
                f"  design {design}, prob={prob_non_zero:.2f}: {rep + 1}/{n_rep}",
                flush=True,
            )

    out = {}
    for name in estimators:
        ests = np.array(estimates[name])
        se_est = np.array(ses[name])
        z = 1.96
        ci_lo = ests - z * se_est
        ci_hi = ests + z * se_est
        out[name] = {
            "mean": ests.mean(),
            "sd": ests.std(ddof=1),
            "ci": np.mean((ci_lo <= PSI0) & (PSI0 <= ci_hi)),
            "n_conv": int(np.sum(converged[name])),
        }
    return out


def print_table(results_by_design, estimators=("gmm",)):
    header = f"{'Design':>6}"
    for name in estimators:
        header += f" | {name:>5} mean  {name:>5} sd  {name:>5} ci"
    print(header)
    print("-" * len(header))
    for design in DESIGNS:
        if design not in results_by_design:
            continue
        row = f"{design:>6}"
        for name in estimators:
            r = results_by_design[design][name]
            row += f" | {r['mean']:5.3f} {r['sd']:5.3f} {r['ci']:5.3f}"
        print(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rep", type=int, default=N_REP)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--design", type=int, default=None, choices=DESIGNS)
    parser.add_argument(
        "--prob-non-zero",
        type=float,
        default=None,
        help="Single edge probability; default runs all values in PROB_NON_ZERO",
    )
    parser.add_argument("--pilot", action="store_true", help="Run a single replication and print the fit")
    args = parser.parse_args()

    m = args.n if args.m is None else args.m
    prob_values = [args.prob_non_zero] if args.prob_non_zero is not None else list(PROB_NON_ZERO)

    if args.pilot:
        prob = prob_values[0]
        df, X = generate_jochmans_panel(
            seed=0, design=1, n=args.n, m=m, prob_non_zero=prob)
        print(
            f"panel {args.n}x{m}, positive edges: {len(df)} / {args.n * m}, "
            f"prob_non_zero={prob:.2f}"
        )
        est = fit_gmm(df, X, np.array([0.0], dtype=np.float32))
        est.summary()
    else:
        designs = [args.design] if args.design else DESIGNS
        for prob in prob_values:
            print(f"\n=== prob_non_zero = {prob:.2f} ===", flush=True)
            results = {}
            for design in designs:
                print(
                    f"Design {design}: {args.n_rep} reps, n=m={args.n}, prob_non_zero={prob:.2f}",
                    flush=True,
                )
                results[design] = monte_carlo(
                    design,
                    prob_non_zero=prob,
                    n_rep=args.n_rep,
                    n=args.n,
                    m=m,
                    estimators=tuple(args.estimators),
                )

            print_table(results, estimators=tuple(args.estimators))
