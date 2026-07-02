import numpy as np
from src.polyads.data import generate_jochmans_panel
from src.polyads.model import PolyadEstimator

# Jochmans (2015) Table 1 DGP: two-way exponential model, Design 1, n = m = 50
df, X = generate_jochmans_panel(seed=1, design=1, n=50, m=50, prob_non_zero=0.05)
columns = df.columns.tolist()
beta_init = np.array([0.0], dtype=np.float32)

for loss in ["gmm_unleveled", "gmm_leveled"]:
    estimator = PolyadEstimator(max_n_polyads=10, use_tqdm=False, loss=loss)
    estimator.fit(df, columns[:-1], columns[-1], beta_init, X=X)

    estimator = PolyadEstimator(use_tqdm=True, loss=loss)
    estimator.fit(df, columns[:-1], columns[-1], beta_init, X=X)

    print(f"\nLoss: {loss}  (Jochmans Design 1, psi0 = 1)")
    estimator.summary()
