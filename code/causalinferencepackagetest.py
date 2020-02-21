import numpy as np

from causalinference import CausalModel
from causalinference.utils import vignette_data

# Example 1: Dummy data
# The terminology is: 
# Y is the outcome variable (N x 1)
# D is the treatment status indicator (N x 1)
# X is the matrix of covariates (N x K)
# 
# In this example, the data generating process for the data is as follows:
# Y0 = X1^2 + X2^2 + e0
# Y1 = 10 + X1^2 + X2^2 + e1
# e0, e1 ~ U(0, 10)
# So, the subject level treatment effect is:
# Y1 - Y0 = 10 + (e1 - e0)

Y, D, X = vignette_data()
Y.shape, D.shape, X.shape # Verify shapes

causal_model = CausalModel(Y, D, X)

# First thing to do is to check the degree of imbalance in covariates among the 
# treatment and control groups. If Nor-diff is 0.5 this is a sign of worry and 
# indicates that the randomization did not work

print(causal_model.summary_stats)

# The default method is to run a linear regression with interaction with the 
# following specification
# Y_i = a + bD_i + p'(X_i - X_mean) + q'(X_i - X_mean)D_i + e_i 

causal_model.est_via_ols()
print(causal_model.estimates)

# ATE = average treatment effect
# ATC = average treatment effect for controls
# ATT = average treatment effect for treated
# In this example, ATE = 3.672 while we know that the true ATE = 10
#
# To disable the interaction terms included in the regression:

causal_model.est_via_ols(adj=1)
print(causal_model.estimates)

# To not include covariates at all:

causal_model.est_via_ols(adj=0)
print(causal_model.estimates)
print(causal_model.summary_stats)

# OLS is not a reliable estimate of the causal effect since the Nor-diff is 
# more than 0.5 and there is scant evidence that data is not randomly drawn.

# Method1: Matching
# The idea here is to non-parametrically match subjects based on the covariate 
# values so that each person who received a treatment is now compared to a 
# person who is equal in all respects except the treatment. This method hinges 
# heavily on the strength of our conviction that the covariates capture 
# everything that is different between the subjects

causal_model.est_via_matching()
print(causal_model.estimates)

# The estimates from matching are true to the extent that the matching based on 
# the observed covariates is exact. However, there might be inefficiencies in 
# this matching that introduces bias. Note that this is distinct from the 
# sufficiency of the covariates in capturing the characteristics of the sample 
# in entirety

causal_model.est_via_matching(bias_adj=True)
print(causal_model.estimates)

# Method 2: Propensity scores
# If (Y(0), Y(1)) is independent of treatment (D) given the set of covariates, 
# then (Y(0), Y(1)) is independent of treatment (D) given p(X) = P(D = 1 | X)

causal_model.est_propensity_s()
print(causal_model.propensity)
print(causal_model.estimates)

# If there is imbalance in covariates (i.e., nor-diff > 0.5)

print(causal_model.summary_stats)

# It might be a good idea to drop the units with extreme values of propensity 
# scores (i.e., < 0.1 or > 0.9)

causal_model.cutoff
causal_model.trim()
print(causal_model.summary_stats)

# Now notice that the nor-diff is reduced; the covariates are now much more 
# balanced

causal_model.reset()
print(causal_model.summary_stats)
causal_model.est_propensity_s()
causal_model.trim_s() # Optimal cutoff is automatically determined
causal_model.cutoff
print(causal_model.summary_stats)

# Stratifying the data into blocks with similar propensity scores is also useful 
# to understand how the treatment effect varies

causal_model.reset()
causal_model.est_propensity_s()
causal_model.blocks
causal_model.stratify()
print(causal_model.strata) 

# The choice of number of bins could be done automatically to ensure that the 
# propensity scores are significantly different between strata

causal_model.reset()
causal_model.est_propensity_s()
causal_model.stratify_s()
print(causal_model.strata)

for stratum in causal_model.strata:
    stratum.est_via_ols(adj=1)

[stratum.estimates["ols"]["ate"] for stratum in causal_model.strata]

# This procedure can be executed in one step:

causal_model.est_via_blocking()
print(causal_model.estimates)

# Method 3: Doubly robust estimation
# Here we run a weighted regression where if at least one of the propensity 
# score specification or the regression function is correct, we get better 
# estimates of ATE

causal_model.reset()
causal_model.est_propensity_s()
causal_model.est_via_weighting()
print(causal_model.estimates)

# Note that if we specify both the propensity and regression incorrectly, we get 
# wild results

# The following general method emerges when one has to do causal inference with 
# observational data. Our core problem is to observe the treatment effect of D 
# on Y given a set of covariates X

# 1. Examine the data for covariate imbalance. Correct extreme propensity units 
#    using trim. Stratify units by propensity score
# 2. Estimate ATE using the blocking estimator or the matching estimator

Y, D, X = vignette_data()
causal_model = CausalModel(Y, D, X)

print(causal_model.summary_stats)

causal_model.est_propensity_s()
causal_model.trim_s()
causal_model.stratify_s()

causal_model.est_via_blocking()
print(causal_model.estimates)

causal_model.est_via_matching()
print(causal_model.estimates)

