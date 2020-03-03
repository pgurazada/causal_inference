# Code to replicate findings from the paper:
# Colonial Origins of Development (http://economics.mit.edu/files/4123)

import pandas as pd

from dowhy import CausalModel
from pathlib import Path

data_dir = Path("data", "colonial-origins")
data_file = data_dir / "colonial-origins-data.dta"

data_df = (pd.read_stata(data_file)
             .rename(columns={"risk": "protection"}))


# Model

model = CausalModel(data=data_df,
                    treatment="protection",
                    outcome="loggdp",
                    common_causes=["latitude", "asia", "africa", "other"],
                    instruments=["logmort0"])

# Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate through common causes

estimate_bd = model.estimate_effect(identified_estimand, 
                                    method_name="backdoor.linear_regression",
                                    test_significance=True)

print(estimate_bd)

# Estimate through instrumental variable (This is what they do in the paper)

estimate_iv = model.estimate_effect(identified_estimand,
                                    method_name="iv.instrumental_variable",
                                    test_significance=True)

print(estimate_iv)

# Refute

model.refute_estimate(identified_estimand, 
                      estimate_bd,
                      method_name="data_subset_refuter",
                      subset_fraction=0.9)

model.refute_estimate(identified_estimand,
                      estimate_iv,
                      method_name="data_subset_refuter",
                      subset_fraction=0.9)
