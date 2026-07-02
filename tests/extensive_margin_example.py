import numpy as np
import pandas as pd
from src.polyads.data import generate_data
from src.polyads.model import PolyadEstimator


# Generate synthetic Bernoulli network data with p-dimensional features
beta = np.array([1.0, -.5], dtype=np.float32)

df, X = generate_data(
    seed=1,
    n_ds=(40, 40, 40),  # use 40^3 for a runnable demo; 100^3 is very slow
    c=-2,
    shape=np.inf,  # ignored in Bernoulli mode
    beta=beta,
    groups=[[0, 1], [0, 2], [1, 2]],
    y_mode="bernoulli",
)

columns = df.columns.tolist()

# Warmup fit (compile numba kernels)
estimator = PolyadEstimator(max_n_polyads=10, use_tqdm=False, loss="bernoulli")
estimator.fit(df, columns[:-1], columns[-1], np.zeros(beta.size), X=X)

# Fit the extensive-margin estimator
estimator = PolyadEstimator(use_tqdm=True, loss="bernoulli")
estimator.fit(df, columns[:-1], columns[-1], np.zeros(beta.size), X=X)

estimator.summary()
