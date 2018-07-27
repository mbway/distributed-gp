#!/usr/bin/env python3

import sklearn.gaussian_process as sk_gp

# local imports


class SciKitGPExpert:
    def __init__(self, X, y, kernel):
        self.model = sk_gp.GaussianProcessRegressor(kernel=kernel, alpha=0, optimizer=None, copy_X_train=False, normalize_y=False)
        self.X = X
        self.y = y
        self.model.fit(X, y) # doesn't do any optimisation of hyperparameters

    def set_theta(self, theta):
        self.model.kernel.theta = theta
        # have to recompute and cache the Cholesky of K
        self.model.fit(self.X, self.y)

    def log_marginal_likelihood(self, theta, eval_gradient=True):
        return self.model.log_marginal_likelihood(theta, eval_gradient)

    def predict(self, X_new):
        assert X_new.ndim == 2 and X_new.shape[1] == self.X.shape[1]
        mu, sigma = self.model.predict(X_new, return_std=True)
        sigma = sigma.reshape(-1, 1)
        assert mu.ndim == sigma.ndim == 2 and mu.shape[0] == sigma.shape[0] and sigma.shape[1] == 1
        return mu, sigma
