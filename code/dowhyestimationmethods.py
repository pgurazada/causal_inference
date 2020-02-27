import numpy as np
import pandas as pd

import dowhy.datasets

from dowhy import CausalModel

# A tour of several estimation methods built into dowhy

data_dict = dowhy.datasets.linear_dataset(beta=10,
                                          num_common_causes=5,
                                          num_instruments=2,
                                          num_treatments=1,
                                          num_samples=10000,
                                          treatment_is_binary=True,
                                          outcome_is_binary=False)

data_dict["df"].head()
data_dict.keys()

# Model

model = CausalModel(data=data_dict["df"],
                    treatment=data_dict["treatment_name"],
                    outcome=data_dict["outcome_name"],
                    common_causes=data_dict["common_causes_names"],
                    instruments=data_dict["instrument_names"])

# Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimation methods

estimate_regression = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.linear_regression",
                                            test_significance=True)

print(estimate_regression)

estimate_stratification = model.estimate_effect(identified_estimand,
                                                method_name="backdoor.propensity_score_stratification",
                                                target_units="att")

print(estimate_stratification)

estimate_matching = model.estimate_effect(identified_estimand,
                                          method_name="backdoor.propensity_score_matching",
                                          target_units="atc")

print(estimate_matching)

estimate_iv = model.estimate_effect(identified_estimand,
                                    method_name="iv.instrumental_variable",
                                    method_params={"iv_instrument_name": "Z0"})
print(estimate_iv)

estimate_regression_discontinuity = model.estimate_effect(identified_estimand,
                                                          method_name="iv.regression_discontinuity",
                                                          method_params={'rd_variable_name':'Z1',
                                                                         'rd_threshold_value':0.5,
                                                                         'rd_bandwidth': 0.1})

print(estimate_regression_discontinuity)

# Refute

for estimate in [estimate_regression, estimate_stratification, estimate_matching,
                 estimate_iv, estimate_regression_discontinuity]:
                 
            print(model.refute_estimate(identified_estimand, estimate, 
                                        method_name="random_common_cause"))

for estimate in [estimate_regression, estimate_stratification, estimate_matching,
                 estimate_iv, estimate_regression_discontinuity]:
                 
            print(model.refute_estimate(identified_estimand, estimate, 
                                        method_name="data_subset_refuter",
                                        subset_fraction=0.9))