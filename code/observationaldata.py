import math
import dowhy

import numpy as np
import pandas as pd

import dowhy.datasets

from dowhy import CausalModel

# Consider the case where we want to probe the impact of a specific treatment on 
# the outcome in the presence of a potential common cause

rvar = 1 if np.random.uniform() > 0.5 else 0

data_dict = dowhy.datasets.xy_dataset(10000, effect=rvar, sd_error=0.2)
data_dict["df"].head()

# Model
model = CausalModel(data=data_dict["df"],
                    treatment=data_dict["treatment_name"],
                    outcome=data_dict["outcome_name"],
                    common_causes=data_dict["common_causes_names"],
                    instruments=data_dict["instrument_names"])

# Identify
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")

print(estimate.value)
print(rvar)

# Refute

model.refute_estimate(identified_estimand,
                      estimate,
                      method_name="random_common_cause")

model.refute_estimate(identified_estimand,
                      estimate,
                      method_name="placebo_treatment_refuter",
                      placebo_type="permute")

model.refute_estimate(identified_estimand,
                      estimate,
                      method_name="data_subset_refuter",
                      subset_fraction=0.9)

# Effect is robust to refutations

