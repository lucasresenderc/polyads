# Polyads: Statistical Inference in Large Multi-way Networks

A Python package for estimating multi-way gravity models with high-dimensional fixed effects. The polyad estimator addresses the incidental parameter problem by conditioning on sufficient statistics for fixed effects. The same computational core also supports an extensive-margin estimator for binary edges and GMM estimators for multiplicative-error models with non-integer outcomes.

## Overview

Traditional PPML estimators may suffer from the incidental parameter problem when the number of fixed effects grows with sample size. This package implements estimators that difference out fixed effects through polyad-level contrasts, selected with a single `loss` argument.

**Key features:**
- Conditional likelihood (PPML-style) for count data
- Extensive-margin conditional logit for binary outcomes (`loss="bernoulli"`)
- GMM for multiplicative-error models with any non-negative outcome (`gmm_unleveled`, `gmm_leveled`)
- Arbitrary fixed-effect structures (two-way, three-way, four-way, …)
- Computationally efficient for sparse network data
- Asymptotically valid inference via Hájek-projection sandwich variances

## Estimator families

All estimators share polyad enumeration, Newton optimization, and variance computation. The `loss` argument selects the objective and the data rules:

| `loss` | Data | Description |
|--------|------|-------------|
| `poisson_multiclass` | non-negative integers | Full conditional logit over the polyad orbit (default) |
| `poisson_binary` | non-negative integers | Binary split of the orbit at its midpoint |
| `poisson_binary_balanced` | non-negative integers | Binary split balancing probability mass |
| `bernoulli` | binary `{0, 1}` | Extensive-margin polyad rules + conditional logit |
| `gmm_unleveled` | non-negative (float OK) | GMM moment, unleveled bracket |
| `gmm_leveled` | non-negative (float OK) | GMM moment, leveled bracket (better with non-negative covariates) |

Supported values are exported as `SUPPORTED_LOSSES` from `polyads`.

## Quick start (Poisson counts)

```python
import numpy as np
from src.polyads.data import generate_data
from src.polyads.model import PolyadEstimator

beta_true = np.array([1.0, -0.5], dtype=np.float32)
df, X = generate_data(
    seed=1,
    n_ds=(100, 100, 100),
    c=-5,
    shape=np.inf,
    beta=beta_true,
    groups=[[0, 1], [0, 2], [1, 2]],
)

columns = df.columns.tolist()
estimator = PolyadEstimator(use_tqdm=True)  # loss="poisson_multiclass" by default
estimator.fit(df, columns[:-1], columns[-1], np.zeros(2), X=X)
estimator.summary()
```

## Bernoulli (extensive margin)

For binary outcomes, use `y_mode="bernoulli"` in the data generator and `loss="bernoulli"` in the estimator. Polyad enumeration uses extensive-margin activity rules; the conditional logit objective is unchanged.

```python
df, X = generate_data(
    seed=1,
    n_ds=(40, 40, 40),
    c=-2,
    shape=np.inf,
    beta=beta_true,
    groups=[[0, 1], [0, 2], [1, 2]],
    y_mode="bernoulli",
)

estimator = PolyadEstimator(loss="bernoulli", use_tqdm=True)
estimator.fit(df, columns[:-1], columns[-1], np.zeros(2), X=X)
estimator.summary()
```

## GMM (multiplicative-error models)

GMM losses require only conditional-mean assumptions (no Poisson distribution). Outcomes may be non-integer. For a two-way gravity panel, use `generate_jochmans_panel`:

```python
from src.polyads.data import generate_jochmans_panel
from src.polyads.model import PolyadEstimator

df, X = generate_jochmans_panel(
    seed=1, design=1, n=50, m=50, prob_non_zero=0.05,
)
columns = df.columns.tolist()

estimator = PolyadEstimator(loss="gmm_leveled", use_tqdm=True)
estimator.fit(
    df, columns[:-1], columns[-1],
    np.array([0.0], dtype=np.float32), X=X,
)
estimator.summary()
```

Use `gmm_unleveled` for the raw moment bracket and `gmm_leveled` when covariates are non-negative.

## The method

### Multi-way gravity model

Count data are modeled as

```
log λ_{i₁,…,iD} = β′X_{i₁,…,iD} + Σ_g θ^g_{g(i)}
```

GMM targets the multiplicative model `Y_i = exp(β′X_i) × (fixed effects) × ε_i` with `E[ε_i | X] = 1`, without specifying a Poisson distribution.

### The incidental parameter problem

When the number of fixed effects grows with sample size:
- **Two-way models (D = 2):** PPML is consistent
- **Three-way models (D ≥ 3):** PPML can yield biased estimates and unreliable confidence intervals

Polyad conditional likelihood and GMM estimators difference out fixed effects and avoid this problem by construction.

### The polyad solution

The method builds contrasts over structured subsets of edges (polyads). For count data it conditions on polyad-level orbits and maximizes a conditional logit that depends only on β. For GMM it uses product-moment conditions that cancel fixed effects algebraically. Both approaches enumerate active polyads over pairs of realized edges, giving `O(|E|²)` cost when the network is sparse.

## Basic usage

### Model setup

```python
from src.polyads.model import PolyadEstimator

estimator = PolyadEstimator(
    loss="poisson_multiclass",    # see table above
    max_iter=100,
    tol=1e-4,
    max_n_polyads=int(1e8),
    variance_threshold=0.0,       # 0 = exact pairwise variance; 1 = fast approximation
    use_tqdm=False,
)
```

You can override `loss` per fit: `estimator.fit(..., loss="gmm_leveled")`.

### Two-way example (trade)

```python
# Bilateral trade: log λ_ij = β′X_ij + u_i + v_j
estimator = PolyadEstimator(loss="poisson_multiclass")
estimator.fit(
    df=df,
    indices=["exporter", "importer"],
    values="flow",
    beta_init=np.zeros(p),
    X=features,
)
```

### Three-way example (panel)

```python
# Trade panel: log λ_ijt = β′X_ijt + u_ij + v_it + w_jt
estimator.fit(
    df=df,
    indices=["exporter", "importer", "year"],
    values="flow",
    beta_init=np.zeros(p),
    X=features,
)
```

## Advanced features

### Custom feature function

Compute features on-the-fly to save memory:

```python
def compute_features(indices):
    i, j, t = indices
    return np.array([
        np.log(distance[i, j]),
        fta_indicator[i, j, t],
        border[i, j],
    ])

estimator.fit(
    df=df,
    indices=["i", "j", "t"],
    values="y",
    beta_init=np.zeros(3),
    eval_X=compute_features,
)
```

### Results and inference

```python
beta_hat = estimator.beta_
se = np.sqrt(np.diag(estimator.var_))
estimator.summary(alpha=0.05)

print(f"Converged: {estimator.converged_}")
print(f"Iterations: {estimator.iterations_}")
print(f"Active polyads: {estimator.n_polyads_}")
```

## Data format

Supply a long-format `pandas.DataFrame` with index columns and one value column. Rows with zero outcomes may be omitted (they are dropped internally).

Validation depends on `loss`:

| Loss family | Value column |
|-------------|--------------|
| `poisson_*` | non-negative integers |
| `bernoulli` | integers in `{0, 1}` (before zero-drop) |
| `gmm_*` | finite non-negative values (float allowed) |

Feature array shape: `(n₁, …, nD, p)`, or pass a callable `eval_X(key)` instead of materializing the full tensor.

## Diagnostics

```python
if not estimator.converged_:
    print("Warning: did not converge")

if estimator.det_ < 1e-8:
    print("Singular Hessian — possible collinearity")
    estimator.summary()  # shows eigenstructure

print(f"Positive edges: {estimator.n_edges_}")
print(f"Active polyads: {estimator.n_polyads_}")
```

**No polyads found:** data too sparse or no variation within polyads.

**Singular Hessian:** collinear features or variation absorbed by fixed effects.

## Best practices

1. **Pick the right `loss`** for your outcome type (counts, binary, or general non-negative).
2. **Check sparsity:** this package is useful when \|E\| ≪ n.
3. **Scale features** for numerical stability, especially with GMM.
4. **Warm up:** run a small fit first (`max_n_polyads=10`) to compile Numba kernels.
5. **Validate:** check convergence, Hessian determinant, and `n_polyads_`.

## Examples

| Script | Content |
|--------|---------|
| `tests/basic_usage.py` | Three-way Poisson DGP |
| `tests/extensive_margin_example.py` | Bernoulli network |
| `tests/gmm_example.py` | Jochmans panel, GMM unleveled and leveled |
| `tests/polyad_dgp_monte_carlo.py` | Monte Carlo comparison of all loss families |
| `tests/jochmans_table1_replication.py` | GMM replication over Jochmans designs |
| `paper/replication_benchmark.py` | Poisson and Bernoulli timing benchmark |

## Limitations

- Assumes conditional independence given fixed effects and covariates
- Conditional logit losses require integer count or binary data as appropriate
- GMM is not convex and sensitive to the initial parameter
- Slower than PPML for very dense networks
- Requires sufficient within-group variation for identification

## Citation

```bibtex
@misc{resende2025polyads,
      title={Statistical Inference in Large Multi-way Networks},
      author={Lucas Resende and Guillaume Lecu{\'e} and Lionel Wilner and Philippe Chon{\'e}},
      year={2025},
      eprint={2512.02203},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2512.02203},
}
```
