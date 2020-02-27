import random

import pandas as pd

from dowhy import CausalModel

z = [i for i in range(10)]
random.shuffle(z)

df = pd.DataFrame(data={'Z': z, 'X': range(0, 10), 'Y': range(0, 100, 10)})

# Model

model = CausalModel(data=df,
                    treatment='X',
                    outcome='Y',
                    common_causes=['Z'])

# Identify

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate

estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.linear_regression") 

print(estimate.value)