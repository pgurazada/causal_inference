import numpy as np
import pandas as pd

from dowhy import CausalModel

n_points = 1000
education_ability = 1
education_voucher = 0.5
income_ability = 2
income_education = 4

# common cause
ability = np.random.normal(0, 3, size=n_points)

# instrument

voucher = np.random.normal(2, 1, size=n_points)

# treatment
education = education_ability * ability +\
            education_voucher * voucher +\
            np.random.normal(5, 1, size=n_points)

# outcome
income = income_ability * ability +\
         income_education * education +\
         np.random.normal(10, 3, size=n_points)

df = pd.DataFrame(data={"ability": ability,
                        "education": education,
                        "income": income,
                        "voucher": voucher})

# Model

model = CausalModel(data=df,
                    treatment="education",
                    outcome="income",
                    common_causes=["ability"],
                    instruments=["voucher"])

 # Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand, 
                                 method_name="iv.instrumental_variable",
                                 test_significance=True)

print(estimate)