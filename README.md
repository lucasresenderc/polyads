# Polyads: Statistical Inference in Large Multi-way Networks

A Python package for estimating multi-way gravity models with high-dimensional fixed effects. The polyad estimator addresses the incidental parameter problem in Poisson models by conditioning on sufficient statistics for fixed effects.

## Overview

Traditional PPML estimators may suffer from the incidental parameter problem when the number of fixed effects grows with sample size. This package implements a conditional likelihood approach that eliminates fixed effects from the estimation problem entirely.

**Key Features:**
- Handles arbitrary fixed effects structures (two-way, three-way, four-way, etc.)
- Computationally efficient for sparse network data
- Provides asymptotically valid inference


## Quick Start

```python
import numpy as np
import pandas as pd
from src.polyads.data import generate_data
from src.polyads.model import PolyadEstimator

# Generate synthetic three-way gravity data
beta_true = np.array([1.0, -0.5])
df, X = generate_data(
    seed=1,
    n_ds=(100, 100, 100),        # Dimensions: n1 × n2 × n3
    c=-5,                         # Baseline intensity
    shape=np.inf,                 # Poisson model
    beta=beta_true,               
    groups=[[0,1], [0,2], [1,2]] # Three-way fixed effects
)

# Fit the model
columns = df.columns.tolist()
estimator = PolyadEstimator(use_tqdm=True)
estimator.fit(
    df=df, 
    indices=columns[:-1],         # Index columns
    values=columns[-1],           # Count column
    beta_init=np.zeros(2),
    X=X
)

# Display results
estimator.summary()
```

## The Method

### Multi-Way Gravity Model

Consider count data indexed by D dimensions:

```
log λ_{i₁,...,iD} = β'X_{i₁,...,iD} + Σ_g θ^g_{g(i)}
```

where β are structural parameters and θ^g are fixed effects.

### The Incidental Parameter Problem

When the number of fixed effects grows with sample size:
- **Two-way models (D=2)**: PPML is consistent
- **Three-way models (D=3+)**: PPML yields unreliable confidence intervals.

### The Polyad Solution

The method conditions on node degrees (sufficient statistics for fixed effects) and maximizes a conditional likelihood that depends only on β. This eliminates the incidental parameter problem by removing fixed effects from the objective function.

## Basic Usage

### Model Setup

```python
from src.polyads.model import PolyadEstimator

estimator = PolyadEstimator(
    max_iter=100,                 # Maximum iterations
    tol=1e-4,                     # Convergence tolerance
    max_n_polyads=int(1e8),      # Max polyads to process
    use_tqdm=False                # Progress bar
)
```

### Two-Way Example (Trade)

```python
# Bilateral trade: log λ_ij = β'X_ij + u_i + v_j
df = pd.DataFrame({
    'exporter': [...],
    'importer': [...],
    'flow': [...]
})

estimator.fit(
    df=df,
    indices=['exporter', 'importer'],
    values='flow',
    beta_init=np.zeros(p),
    X=features
)
```

### Three-Way Example (Panel)

```python
# Trade panel: log λ_ijt = β'X_ijt + u_ij + v_it + w_jt
df = pd.DataFrame({
    'exporter': [...],
    'importer': [...],
    'year': [...],
    'flow': [...]
})

estimator.fit(
    df=df,
    indices=['exporter', 'importer', 'year'],
    values='flow',
    beta_init=np.zeros(p),
    X=features
)
```

## Advanced Features

### Custom Feature Function

Compute features on-the-fly to save memory:

```python
def compute_features(indices):
    i, j, t = indices
    return np.array([
        np.log(distance[i, j]),
        fta_indicator[i, j, t],
        border[i, j]
    ])

estimator.fit(
    df=df,
    indices=['i', 'j', 't'],
    values='y',
    beta_init=np.zeros(3),
    eval_X=compute_features
)
```

### Results and Inference

```python
# Point estimates
beta_hat = estimator.beta_

# Standard errors
se = np.sqrt(np.diag(estimator.var_))

# Confidence intervals
estimator.summary(alpha=0.05)  # 95% CI

# Diagnostics
print(f"Converged: {estimator.converged_}")
print(f"Iterations: {estimator.iterations_}")
print(f"Active polyads: {estimator.n_polyads_}")
```

## Data Format

### Input DataFrame

```python
df = pd.DataFrame({
    'i1': [0, 0, 1, ...],  # First dimension
    'i2': [0, 1, 0, ...],  # Second dimension
    'i3': [0, 0, 1, ...],  # Third dimension (if 3-way)
    'y': [5, 0, 3, ...]    # Non-negative counts
})
```

### Feature Matrix

```python
# 2-way: (n1, n2, p)
# 3-way: (n1, n2, n3, p)
# 4-way: (n1, n2, n3, n4, p)
X = np.random.randn(n1, n2, n3, p)
```

Or custom feature function, as described.

## Diagnostics

### Check Convergence

```python
if not estimator.converged_:
    print("Warning: Did not converge")
    
if estimator.det_ < 1e-8:
    print("Singular Hessian - possible collinearity")
    estimator.summary()  # Shows eigenstructure
```

### Common Issues

**No polyads found:** Data too sparse or no variation
```python
print(f"Positive edges: {estimator.n_edges_}")
print(f"Active polyads: {estimator.n_polyads_}")
```

**Singular Hessian:** Collinear features or absorbed by fixed effects
```python
# Check correlation
import pandas as pd
pd.DataFrame(X.reshape(-1, p)).corr()
```

## Comparison with PPML

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Polyads** | D≥3, sparse | No bias, valid inference | Slower for dense data |
| **PPML** | D=2, dense | Fast, familiar | IPP for D≥3 |


## Best Practices

1. **Start simple:** Fewer features initially
2. **Check sparsity:** Method works best when |E| ≪ n
3. **Scale features:** Normalize for numerical stability
4. **Warm-up:** Use small problem first for JIT compilation
5. **Validate:** Check convergence and Hessian determinant

## Limitations

- Assumes conditional independence given fixed effects and covariates
- Designed for count data (not continuous)
- Slower than PPML for very dense networks
- Requires sufficient within-group variation

## Citation

```bibtex
@misc{resende2025polyads,
      title={Statistical Inference in Large Multi-way Networks}, 
      author={Lucas Resende and Guillaume Lecué and Lionel Wilner and Philippe Choné},
      year={2025},
      eprint={2512.02203},
      archivePrefix={arXiv},
      primaryClass={econ.EM},
      url={https://arxiv.org/abs/2512.02203}, 
}
```