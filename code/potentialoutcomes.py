"""
Direct duplication of the excellent series of blog posts here:

http://www.degeneratestate.org/posts/2018/Mar/24/causal-inference-with-python-part-1-potential-outcomes/

"""


import pandas as pd
import numpy as np

import datagenerators as dg

from causalinference import CausalModel
from scipy.stats import chi2_contingency


def estimate_uplift(ds: pd.DataFrame):
    """
    Estimates the treatment effect between two groups
    Estimates the standard error assuming a normal distribution 
    """
    control = ds.loc[ds.x == 0]
    treatment = ds.loc[ds.x == 1]

    delta = (treatment.y.mean() - control.y.mean())
    delta_err = 1.96 * np.sqrt(treatment.y.var()/treatment.shape[0] + \
                               control.y.var()/control.shape[0])

    return {"estimated_effect": delta, "standard_error": delta_err}


def run_ab_test(datagenerator, n_samples=10000, filter_=None):
    """
    Generates n_samples from datagenerator with the value of X randomized
    so that 50% of the samples recieve treatment X=1 and 50% receive X=0,
    and feeds the results into `estimate_uplift` to get an unbiased 
    estimate of the average treatment effect.
    """

    n_samples_a = int(n_samples/2)
    n_samples_b = n_samples - n_samples_a

    set_X = (np.concatenate([np.ones(n_samples_a), np.zeros(n_samples_b)])
               .astype(np.int64))

    ds = datagenerator(n_samples=n_samples, set_X=set_X)

    if filter_ != None:
        ds = ds[filter_(ds)].copy()

    return estimate_uplift(ds)


# In this data, the hypothesis is to check if wearing cool hats ("x") is related
# to the productivity of the hat-wearer ("y") 

observed_data_0 = dg.generate_dataset_0()
estimate_uplift(observed_data_0)

contingency_table = (observed_data_0.assign(placeholder=1)
                                    .pivot_table(index="x",
                                                 columns="y",
                                                 values="placeholder",
                                                 aggfunc="sum")
                                    .values)

chi2_contingency(contingency_table, lambda_="log-likelihood")

run_ab_test(dg.generate_dataset_0)

# Now we assume that there is a underlying third factor Z that drives both X and 
# Y
observed_data_0_with_confounders = dg.generate_dataset_0(show_z=True)

estimate_uplift(observed_data_0_with_confounders.loc[lambda x: x.z == 0])
estimate_uplift(observed_data_0_with_confounders.loc[lambda x: x.z == 1])

# One way to tackle counterfactuals is to model the outcomes directly. Doing so 
# we can then measure the average treatment effect by using these estimators 
# learned from the model

observed_data_1 = dg.generate_dataset_1()

(observed_data_1.loc[lambda df: df.x == 0]
                .y
                .mean())

(observed_data_1.loc[lambda df: df.x == 1]
                .y
                .mean())

estimate_uplift(observed_data_1)
run_ab_test(dg.generate_dataset_1)

# We can fit a linear model *assuming* that the data generating process is so 
# and involves the exact same predictors. Bad things can happen when these 
# assumptions dont hold. Worse still, there is no way to verify if these are 
# true

causal_model = CausalModel(Y=observed_data_1.y.values,
                           D=observed_data_1.x.values,
                           X=observed_data_1.z.values)

causal_model.est_via_ols(adj=1)
print(causal_model.estimates)

# We now look at an example where our assumptions alone are insufficient

observed_data_2 = dg.generate_dataset_2()  
estimate_uplift(observed_data_2)
run_ab_test(dg.generate_dataset_2)

causal_model = CausalModel(Y=observed_data_2.y.values,
                           D=observed_data_2.x.values,
                           X=observed_data_2.z.values)

causal_model.est_via_ols(adj=1)
print(causal_model.estimates)

# One way to counter this is to use a matching process that maps each unit that 
# received the treatment with one that did not by comparing its distance in the 
# covariate space
 
causal_model.est_via_matching()
print(causal_model.estimates)

# This approach becomes difficult when there is imbalance in covariates since 
# this involves extrapolation of the "similarity" metric beyond what can be 
# observed

observed_data_3 = dg.generate_dataset_3()

estimate_uplift(observed_data_3)
run_ab_test(dg.generate_dataset_3)

causal_model = CausalModel(Y=observed_data_3.y.values,
                           D=observed_data_3.x.values,
                           X=observed_data_3.z.values)

causal_model.est_via_ols()

print(causal_model.estimates)

causal_model.est_via_matching()

print(causal_model.estimates)

print(causal_model.summary_stats)

# A nice method that formalizes the search of groups that are most similar to 
# the treatment and control groups is the propensity-score method. Here we are 
# interested in estimating the probability that the subject would end up with 
# the treatment, given the covariates

estimate_uplift(observed_data_1)

causal_model = CausalModel(Y=observed_data_1.y.values,
                           D=observed_data_1.x.values,
                           X=observed_data_1.z.values)

causal_model.est_propensity_s()

propensity = causal_model.propensity["fitted"]

observed_data_1["ips"] = np.where(observed_data_1.x == 1, 
                                  1/propensity,
                                  1/(1-propensity))

observed_data_1["ipsw"] = observed_data_1.y * observed_data_1.ips

ipse = (observed_data_1[observed_data_1.x == 1]["ipsw"].sum() - 
        observed_data_1[observed_data_1.x == 0]["ipsw"].sum())/observed_data_1.shape[0]

# Do not use skllearn's logistic regression since it uses regularization by 
# default

# Sometimes there can be more than one covariate variable

df = dg.generate_exercise_dataset_2()

estimate_uplift(df)

zs = [c for c in df.columns if c.startswith('z')]

causal_model = CausalModel(Y=df.y.values,
                           D=df.x.values,
                           X=df[zs].values)

causal_model.est_via_ols()
causal_model.est_via_matching()
causal_model.est_propensity_s()
causal_model.est_via_weighting()

causal_model.stratify_s()
causal_model.est_via_blocking()

print(causal_model.estimates)

# Ultimately blindly controlling for everything we an measure, however good our 
# method might be, without a reasonable causal graph underlying the process is 
# guaranteed to produce unreliable results