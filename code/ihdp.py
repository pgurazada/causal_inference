import numpy as np
import pandas as pd

from dowhy import CausalModel

DATA_URL = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
DATA_COLS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]

for i in range(1, 26):
    DATA_COLS.append("x"+str(i))

data_df = pd.read_csv(DATA_URL, header=None)
data_df.columns = DATA_COLS

data_df["treatment"] = data_df.treatment.astype("bool")

data_df.head()

# Model

model = CausalModel(data=data_df,
                    treatment="treatment",
                    outcome="y_factual",
                    common_causes=["x"+str(i) for i in range(1, 26)])

# Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate

estimate_matching = model.estimate_effect(identified_estimand, 
                                          method_name="backdoor.propensity_score_matching")

print(estimate_matching)

estimate_regression = model.estimate_effect(identified_estimand, 
                                            method_name="backdoor.linear_regression",
                                            test_significance=True)

print(estimate_regression)

# Refute

model.refute_estimate(identified_estimand, 
                      estimate_regression, 
                      method_name="data_subset_refuter",
                      subset_fraction=0.9)