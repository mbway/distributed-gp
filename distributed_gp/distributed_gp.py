#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import sklearn
import warnings

# local imports
from .utils import unzip
from .scikit_expert import SciKitGPExpert


class DistributedGP:
    """ Distributed Gaussian Process Regression

    References:
        [1] M. P. Deisenroth, D. Fox, and C. E. Rasmussen,
            “Gaussian Processes for Data-Efficient Learning in Robotics and Control,”
            IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 2, pp. 408–423, Feb. 2015.
    """
    def __init__(self, X, y, num_experts, kernel, expert_class=SciKitGPExpert):
        """
        Args:
            X: the input data to fit to
            y: the output data to fit to
            num_experts: the number of experts to use
            kernel: the kernel to use in each of the expert models. The parameters of the kernel are shared between every expert
            expert_class: the type of GPExpert to use
        """
        assert X.ndim == y.ndim == 2 and X.shape[0] == y.shape[0] and y.shape[1] == 1
        self.X = X
        self.y = y
        self.kernel = kernel
        # split the data evenly between the experts without replacement
        # if the data doesn't evenly divide then some data sets will have different lengths
        indices = np.random.permutation(np.arange(X.shape[0]))
        self.data_set_indices = np.array_split(indices, num_experts)
        self.experts = [expert_class(X[ids], y[ids], sklearn.base.clone(kernel)) for ids in self.data_set_indices]


    def optimise_params(self, iterations=1, randomize_theta=True, quiet=True):
        """ Optimise the hyperparameters of the model by maximising the marginal log likelihood

        Args:
            iterations: the number of optimisation iterations to perform
            randomize_theta: whether to randomize theta on the first
                optimisation iteration or to take the current
                `self.kernel.theta` instead.
            quiet: whether to display warnings when optimisations fail
        """
        k = self.kernel
        start_thetas = np.random.uniform(k.bounds[:, 0], k.bounds[:, 1], size=(iterations, len(k.theta)))
        if not randomize_theta:
            start_thetas[0] = self.kernel.theta

        thetas = []
        likelihoods = []

        def objective(theta):
            return tuple(-v for v in self.log_marginal_likelihood(theta, eval_gradient=True))

        for i in range(iterations):
            result = scipy.optimize.minimize(
                fun=objective,
                x0=start_thetas[i],
                jac=True, # whether the objective function returns calculated gradients
                bounds=self.kernel.bounds, # note: log-transformed bounds (because theta is log transformed)
                method='L-BFGS-B'
            )
            if not result.success and not quiet:
                warnings.warn("optimisation iteration {}/{} failed".format(i+1, iterations))
            else:
                thetas.append(result.x)
                likelihoods.append(result.fun)

        if not thetas:
            raise RuntimeError('every optimisation iteration failed!')

        best_theta = thetas[np.argmin(likelihoods)]
        self.kernel.theta = best_theta

        for e in self.experts:
            e.set_theta(best_theta)


    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """ calculate the log marginal data likelihood for the model

        see equation 5 of [1]
        since the likelihood is just a sum, the gradients can be summed as well
        to obtain the overall gradient.
        """
        theta = theta if theta is not None else self.kernel.theta
        expert_res = [e.log_marginal_likelihood(theta, eval_gradient) for e in self.experts]
        pk, dpk_dtheta = unzip(expert_res) if eval_gradient else (expert_res, None)
        p = sum(pk)
        if eval_gradient:
            dp_dtheta = sum(dpk_dtheta)
            return p, dp_dtheta
        else:
            return p


    def predict_POE(self, X_new):
        """ predict the mean and variance at the given points using Product of Experts

        see equations 7-9 of [1]
        """
        assert X_new.ndim == 2 and X_new.shape[1] == self.X.shape[1]
        mus, sigmas = unzip([e.predict(X_new) for e in self.experts])
        mus, sigmas = np.hstack(mus), np.hstack(sigmas)
        sigmas = np.clip(sigmas, 1e-10, np.inf)

        r_var = np.power(sigmas, -2) # the reciprocal of the predicted variance of each expert

        var = np.reciprocal(np.sum(r_var, axis=1).reshape(-1, 1))
        mu = var * np.sum(r_var * mus, axis=1).reshape(-1, 1)

        return mu, var


