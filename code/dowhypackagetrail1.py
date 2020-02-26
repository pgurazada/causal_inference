import numpy as np
import pandas as pd

import dowhy.datasets

from dowhy import CausalModel

# The general method to estimate the causal effect of a treatment on the outcome 
# is: Model -> Identify -> Estimate -> Refute

# Common causes impact both the outcome and the treatment indicator.
# Instrumental variables impact only the treatment indicator.
# Effect modifiers directly impact the outcome
# Beyond these, we also assume the existence of unobserved confounders

data = dowhy.datasets.linear_dataset(beta=10,
                                     num_common_causes=5,
                                     num_instruments=2,
                                     num_effect_modifiers=1,
                                     num_samples=10000,
                                     treatment_is_binary=True)

df = data["df"]

print(data["gml_graph"])

causal_model = CausalModel(data=df,
                           treatment=data["treatment_name"],
                           outcome=data["outcome_name"],
                           graph=data["gml_graph"])


# First step is to utilize do-calculus to reduce the estimation to a 
# probabilistic expression

identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# Second step is to estimate the causal effect.
# There are three methods to do this

causal_estimate_matching = causal_model.estimate_effect(identified_estimand,
                                                        method_name="backdoor.propensity_score_matching")

print(causal_estimate_matching)
print(causal_estimate_matching.value)

causal_estimate_stratification = causal_model.estimate_effect(identified_estimand,
                                                              method_name="backdoor.propensity_score_stratification")

print(causal_estimate_stratification.value)

causal_estimate_weighting = causal_model.estimate_effect(identified_estimand,
                                                         method_name="backdoor.propensity_score_weighting")

print(causal_estimate_weighting.value)

# We can also build up the same graph by identifying all the named components 
# but not explicitly passing a graph object

model = CausalModel(data=df,
                    treatment=data["treatment_name"],
                    outcome=data["outcome_name"],
                    common_causes=data["common_causes_names"],
                    effect_modifiers=data["effect_modifier_names"])

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_weighting")

print(estimate.value)

# In refuting the estimate we stress test the validity of the estimate under a 
# variety of fail-safe conditions

# Add a random cause variable

res_random = model.refute_estimate(identified_estimand,
                                   estimate,
                                   method_name="random_common_cause")

print(res_random)

# Replace treatment with a random placebo

res_placebo = model.refute_estimate(identified_estimand, 
                                    estimate,
                                    method_name="placebo_treatment_refuter", 
                                    placebo_type="permute")
print(res_placebo)

# Remove a random subset of the data

res_subset = model.refute_estimate(identified_estimand, 
                                   estimate,
                                   method_name="data_subset_refuter", 
                                   subset_fraction=0.9)
print(res_subset)