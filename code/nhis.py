import pandas as pd

import causalinference

from dowhy import CausalModel
from pathlib import Path

data_dir = Path("data")
data_file = data_dir / "NHIS2009_clean.dta"

data_df = (pd.read_stata(data_file)
             .assign(age=lambda x: pd.to_numeric(x.age, errors="coerce"))
             .query("perweight != 0")
             .query("adltempl >= 1")
             .query("age > 26 and age < 59"))

data_df.columns

cleaned_df = (data_df.loc[:, ["age", "empl", "famsize", "hlth", "inc", "nwhite",
                              "yedu", "fml", "incmp"]])

Y = cleaned_df.incmp
D = cleaned_df.fml
X = cleaned_df.loc[:, ["age", "empl", "famsize", "hlth", "inc", "nwhite", "yedu"]]

causal_model = causalinference.CausalModel(Y, D, X)

print(causal_model.summary_stats)

# Data seem resonably balanced