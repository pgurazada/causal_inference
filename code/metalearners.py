import pandas as pd

from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor
from causalml.inference.meta import XGBRRegressor

from econml.dml import DMLCateEstimator
from econml.metalearners import TLearner

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

from causalml.propensity import ElasticNetPropensityModel


file_url = "https://msalicedatapublic.blob.core.windows.net/datasets/Pricing/pricing_sample.csv"
train_df = pd.read_csv(file_url)

Y = train_df["demand"].values
T = (train_df["price"] > 0.9).values
X = train_df.drop(columns=["demand", "price"]).values

# Theoretically we expect the treatment effect to be negative, that is, when the 
# price increases, demand should fall
# Conceptually, the T-model uses two models, one to predict treatment and 
# another to predict outcome. Both can be any arbitrary machine learning models.

learner_s = LRSRegressor()
ate_s = learner_s.estimate_ate(X, T, Y)
print(ate_s)

learner_t = XGBTRegressor()
ate_t = learner_t.estimate_ate(X, T, Y, 
                               bootstrap_ci=True,
                               n_bootstraps=1000)
print(ate_t)


# Building a T-learner

learner_t_manual = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                                    model_t=GradientBoostingRegressor(),
                                    model_final=LassoCV())

learner_t_manual.fit(Y, T, X, inference="bootstrap")
learner_t_manual.score_
learner_t_manual.const_marginal_effect(X)

learner_t_manual = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                                    model_t=GradientBoostingRegressor(),
                                    model_final=GradientBoostingRegressor())

learner_t_manual.fit(Y, T, X, inference="bootstrap")
learner_t_manual.score_
learner_t_manual.const_marginal_effect(X)

models = GradientBoostingRegressor(n_estimators=100, 
                                   max_depth=6, 
                                   min_samples_leaf=10)
auto_learner_t = TLearner(models)
auto_learner_t.fit(Y, T, X, inference="bootstrap")

auto_learner_t.effect(X).mean()
auto_learner_t.effect_interval(X)

propensity_model = ElasticNetPropensityModel(n_fold=5, 
                                             random_state=20130810)
propensity_score = propensity_model.fit_predict(X, T)

# Rough work
