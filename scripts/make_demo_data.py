"""
Generate a demo dataset for multivariate forecasting.
The dataset contains the following columns:
    - value: a noisy sine wave
    - lead: a noisy sine wave shifted 5 steps ahead
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
t = np.arange(1000)

values = np.sin(t / 20) + 0.1 * rng.standard_normal(len(t))
lead = np.sin((t + 5) / 20) + 0.1 * rng.standard_normal(len(t))

pd.DataFrame({"value": values, "lead": lead}).to_csv("data/demo.csv", index=False)
