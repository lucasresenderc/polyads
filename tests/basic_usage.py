import numpy as np
import pandas as pd
from src.polyads.data import generate_data
from src.polyads.model import PolyadEstimator


# Generate synthetic data with p-dimensional features
beta = np.array([1.0, -.5], dtype=np.float32)

df, X = generate_data(
    seed=1,
    n_ds=(100, 100, 100),  # Dimensions of the data
    c=-5,         # Baseline intensity
    shape=np.inf,   # Use np.inf for a Poisson model
    beta=beta,      # True effect of covariates (vector)
    groups=[[0,1],[0,2],[1,2]], # Fixed effects for each dimension
)

# Get the column names
columns = df.columns.tolist()

# Initialize and fit the estimator warmup (just to compile)
estimator = PolyadEstimator(max_n_polyads=10, use_tqdm=False)
estimator.fit(df, columns[:-1], columns[-1], np.zeros(beta.size), X=X)

# Initialize and fit the estimator
estimator = PolyadEstimator(use_tqdm=True)
estimator.fit(df, columns[:-1], columns[-1], np.zeros(beta.size), X=X)

# Print the statistical summary of the fit
estimator.summary()