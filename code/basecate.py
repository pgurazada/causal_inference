import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

from econml.dml import LinearDMLCateEstimator

n_samples = 1000
n_controls = 100
n_instruments = 1
n_features = 1 
n_treatments = 1

# data generating process

alpha = np.random.normal(size=(n_controls, 1))
beta = np.random.normal(size=(n_instruments, 1))
gamma = np.random.normal(size=(n_treatments, 1))
delta = np.random.normal(size=(n_treatments, 1))
zeta = np.random.normal(size=(n_controls, 1))

W = np.random.normal(size=(n_samples, n_controls))
Z = np.random.normal(size=(n_samples, n_instruments))
X = np.random.normal(size=(n_samples, n_features))
eta = np.random.normal(size=(n_samples, n_treatments))
epsilon = np.random.normal(size=(n_samples, 1))

T = np.dot(W, alpha) + np.dot(Z, beta) + eta
y = np.dot(T**2, gamma) + np.dot(np.multiply(T, X), delta) + np.dot(W, zeta) + epsilon


estimator = LinearDMLCateEstimator()

estimator.fit(y, T, X, W, inference="statsmodels")
estimator.score_

lb, ub = estimator.const_marginal_effect_interval(X, alpha=.05)


# Rough work
W.shape