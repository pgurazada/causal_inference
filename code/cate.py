import numpy as np
import pandas as pd

import dowhy.datasets

from dowhy import CausalModel

data_dict = dowhy.datasets.linear_dataset(beta=10,
                                          num_samples=10000,
                                          num_common_causes=4,
                                          num_treatments=1,
                                          num_instruments=2,
                                          num_effect_modifiers=2,
                                          treatment_is_binary=False)

data_dict["df"].head()                                          

data_dict["ate"] # True effect

# Model

model = CausalModel(data=data_dict["df"],
                    treatment=data_dict["treatment_name"],
                    outcome=data_dict["outcome_name"],
                    common_causes=data_dict["common_causes_names"],
                    instruments=data_dict["instrument_names"],
                    effect_modifiers=data_dict["effect_modifier_names"])

# Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate

estimate_regression = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression")

print(estimate_regression.value)

estimate_regression = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            control_value=0,
                                            treatment_value=1)

print(estimate_regression.value)

# Refute

model.refute_estimate(identified_estimand, estimate_regression,
                      method_name="random_common_cause")

model.refute_estimate(identified_estimand, estimate_regression,
                      method_name="data_subset_refuter",
                      subset_fraction=0.9)                      