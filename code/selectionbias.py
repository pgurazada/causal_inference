import pandas as pd

from pathlib import Path

data_dir = Path("data")
data_file = data_dir / "toy_data.csv"

data_df = pd.read_csv(data_file)

# SDO = E[Y1_i | D_i = 1] - E[Y0_i | D_i = 0]

# When the randomization does not work

SDO = (data_df.query("D_i == 1").Y1_i.mean() - 
       data_df.query("D_i == 0").Y0_i.mean())

ATE = (data_df.Y1_i.mean() - 
       data_df.Y0_i.mean())

bias = SDO - ATE
print(bias)

(data_df.groupby(["D_i"])
        .agg(E_y1 = ("Y1_i", "mean"),
             E_y0 = ("Y0_i", "mean")))

# When randomization works

SDO_random = (data_df.query("D_i_random == 1").Y1_i.mean() - 
              data_df.query("D_i_random == 0").Y0_i.mean())

print(SDO_random - ATE)

# Selection bias = E[Y0_i | D_i = 1] - E[Y0_i | D_i = 0]

selection_bias = (data_df.query("D_i == 1").Y0_i.mean() - 
                  data_df.query("D_i == 0").Y0_i.mean())

print(selection_bias)

heterogenous_treatment_effect = bias - selection_bias
print(heterogenous_treatment_effect)

print("SDO:", SDO, '\n')
print("ATE:", ATE, '\n')
print("Selection Bias:", selection_bias, '\n')
print("Heterogenous treatment bias: ", heterogenous_treatment_effect, '\n')

assert SDO == (ATE+selection_bias+heterogenous_treatment_effect)

# Rough work