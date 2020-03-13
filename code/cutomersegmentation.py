"""
The aim is to find the price elasticity of demand across various subsets of 
people defined by their income ranges
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor

from econml.dml import LinearDMLCateEstimator, ForestDMLCateEstimator


# Import the sample pricing data
file_url = "https://msalicedatapublic.blob.core.windows.net/datasets/Pricing/pricing_sample.csv"
train_df = pd.read_csv(file_url)

# Y = g(T, X, W, e) -> Model 1
# T = f(X, W, n) -> Model 2

Y = train_df.demand # Outcome
T = train_df.price # Treatment
X = train_df.loc[:, ["income"]].values # Features
W = train_df.drop(columns=["demand", "price", "income"]).values # common causes (confounders)

log_Y = np.log(Y)
log_T = np.log(T)

estimator = LinearDMLCateEstimator(model_y=GradientBoostingRegressor(),
                                   model_t=GradientBoostingRegressor(),
                                   featurizer=PolynomialFeatures(degree=2, 
                                                                 include_bias=False))

estimator.fit(log_Y, log_T, X, W, inference="statsmodels")


nonparam_estimator = ForestDMLCateEstimator(model_y=GradientBoostingRegressor(),
                                            model_t=GradientBoostingRegressor())

nonparam_estimator.fit(log_Y, log_T, X ,W, inference="blb")


# Rough work
