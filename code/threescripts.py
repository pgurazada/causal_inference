"""
https://www.inference.vc/causal-inference-2-illustrating-interventions-in-a-toy-example/
"""

import numpy as np

from scipy import stats

NUM_SAMPLES = 10000

# Consider random draws from Gaussians using three different methods. In all 
# three cases E[X] = 0, E[Y] = 1, Cov(X, Y) = 1

# Method 1
# X ~ N(0,1)
# Y ~ 1 + X + sqrt(3) * N(0,1)

xs_1 = np.array([np.random.randn() for _ in range(NUM_SAMPLES)])
ys_1 = np.array([(x + 1 + np.sqrt(3) * np.random.randn()) for x in xs_1])

xs_1.mean(), ys_1.mean()
np.cov(xs_1, ys_1)

# Method 2
# Y ~ 1 + 2 * N(0,1)
# X ~ (Y - 1)/4 + sqrt(3)*randn()/2 

ys_2 = np.array([1 + 2*np.random.randn() for _ in range(NUM_SAMPLES)])
xs_2 = np.array([((y - 1)/4 + np.sqrt(3)*np.random.randn()/2) for y in ys_2])

xs_2.mean(), ys_2.mean()
np.cov(xs_2, ys_2)

# Method 3
# Z ~ N(0, 1)
# Y ~ Z + 1 + sqrt(3) * N(0, 1)
# X = Z

zs_3 = np.array([np.random.randn() for _ in range(NUM_SAMPLES)])
ys_3 = np.array([z + 1 + np.sqrt(3) * np.random.randn() for z in zs_3])
xs_3 = zs_3

xs_3.mean(), ys_3.mean()
np.cov(xs_3, ys_3)

stats.ttest_ind(xs_1, xs_2)
stats.ttest_ind(xs_1, xs_3)

stats.ttest_ind(ys_1, ys_2)
stats.ttest_ind(ys_1, ys_3)

# Now, we intervene and set X = 3 in each case

xs_1 = np.array([3 for _ in range(NUM_SAMPLES)])
ys_1 = np.array([(x + 1 + np.sqrt(3) * np.random.randn()) for x in xs_1])

ys_2 = np.array([1 + 2*np.random.randn() for _ in range(NUM_SAMPLES)])
xs_2 = np.array([3 for y in ys_2])

zs_3 = np.array([np.random.randn() for _ in range(NUM_SAMPLES)])
ys_3 = np.array([z + 1 + np.sqrt(3) * np.random.randn() for z in zs_3])
xs_3 = np.array([3 for z in zs_3])

ys_1.mean(), ys_2.mean(), ys_3.mean()

# This illustrates that the joint distribution is not enough to predict Y under 
# intervention. Using do-calculus we can derive that case 2 and 3 should have 
# the same mean since P(Y|do(X)) = P(Y) in both these cases. For case 1, 
# P(Y|do(X)) = P(Y|X)