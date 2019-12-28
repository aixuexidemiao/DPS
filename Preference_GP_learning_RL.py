
"""
"""
import os
import scipy.io as io

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from ValueIteration import value_iteration


def advance(num_policies, dirichlet_posterior, GP_posterior_model, epsilon, num_states,
            num_actions, time_horizon, finite_horizon_iteration):
    """
    Draw a specified number of samples from the preference GP Bayesian model
    posterior.

    Inputs:
        1) num_policies: the number of samples to draw from the posterior; a
           positive integer.

        3) GP_posterior_model: this is the model posterior, represented as a
           dictionary of the form {'mean': post_mean, 'cov_evecs': evecs,
           'cov_evals': evals}; post_mean is the posterior mean, a length-n
           NumPy array in which n is the number of points over which the
           posterior is to be sampled. cov_evecs is an n-by-n NumPy array in
           which each column is an eigenvector of the posterior covariance,
           and evals is a length-n array of the eigenvalues of the posterior
           covariance.

    Outputs:
        1) A num_samples-length NumPy array, in which each element is the index
           of a sample.
        2) A num_samples-length list, in which each entry is a sampled reward
           function. Each reward function sample is a length-n vector (see
           above for definition of n).

    """
    policies = []
    reward_models = []

    # Unpack model posterior:
    mean = GP_posterior_model['mean']
    cov_evecs = GP_posterior_model['cov_evecs']
    cov_evals = GP_posterior_model['cov_evals']


    for i in range(num_policies):

        # Sample state transition dynamics from Dirichlet posterior:
        dynamics_sample = []

        for state in range(num_states):

            dynamics_sample_ = []

            for action in range(num_actions):

                dynamics_sample_.append(np.random.dirichlet(dirichlet_posterior[state][action]))

            dynamics_sample.append(dynamics_sample_)

        # Sample reward function from GP model posterior:
        X = np.random.normal(size = len(mean))
        R = mean + cov_evecs @ np.diag(np.sqrt(cov_evals)) @ X
        reward_sample = np.real(R)

        # Determine horizon to use for value iteration
        if finite_horizon_iteration:
            H = time_horizon
        else:
            H = np.infty

        # Value iteration to determine policy:
        policies.append(value_iteration(dynamics_sample, reward_sample, num_states,
                                        num_actions, epsilon = epsilon,
                                        H = H)[0])
        reward_models.append(R)

    return policies, reward_models


def feedback(visitation_difference_matrix, labels, GP_prior_cov_inv, preference_noise,
             r_init = []):
    """
    Function for updating the GP preference model given data.

    Inputs:
        1) visitation_difference_matrix: num_preference-by-num_state_action_pair NumPy array
        2) labels: num_data_points NumPy array (all elements should be zeros
           and ones) 0 = pt1 preferred; 1 = pt2 preferred.
        3) GP_prior_cov_inv: n-by-n NumPy array, where n is the number of
           points over which the posterior is to be sampled
        4) preference_noise: positive scalar parameter. Higher values indicate
           larger amounts of noise in the expert preferences.
        5) (Optional) initial guess for convex optimization; length-n NumPy
           array when specified.

    Output: the updated model posterior, represented as a dictionary of the
           form {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals};
           post_mean is the posterior mean, a length-n NumPy array in which n
           is the number of points over which the posterior is to be sampled.
           cov_evecs is an n-by-n NumPy array in which each column is an
           eigenvector of the posterior covariance, and evals is a length-n
           array of the eigenvalues of the posterior covariance.

    """
    num_features = GP_prior_cov_inv.shape[0]

    # Solve convex optimization problem to obtain the posterior mean reward
    # vector via Laplace approximation:
    if r_init == []:
        r_init = np.zeros(num_features)    # Initial guess

    res = minimize(preference_GP_objective, r_init, args = (visitation_difference_matrix, labels,
                   GP_prior_cov_inv, preference_noise), method='L-BFGS-B',
                   jac=preference_GP_gradient)

    # The posterior mean is the solution to the optimization problem:
    post_mean = res.x

    # Obtain inverse of posterior covariance approximation by evaluating the
    # objective function's Hessian at the posterior mean estimate:
    post_cov_inverse = preference_GP_hessian(post_mean, visitation_difference_matrix, labels,
                   GP_prior_cov_inv, preference_noise)

    # Calculate the eigenvectors and eigenvalues of the inverse posterior
    # covariance matrix:
    evals, evecs = np.linalg.eigh(post_cov_inverse)

    # Invert the eigenvalues to get the eigenvalues corresponding to the
    # covariance matrix:
    evals = 1 / evals

    # Return the model posterior:
    return {'mean': post_mean, 'cov_evecs': evecs, 'cov_evals': evals}


def preference_GP_objective(f, visitation_difference_matrix, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the optimization objective function for finding the posterior
    mean of the GP preference model (at a given point); the posterior mean is
    the minimum of this (convex) objective function.

    Inputs:
        1) f: the "point" at which to evaluate the objective function. This is
           a length-n vector, where n is the number of points over which the
           posterior is to be sampled.
        2) visitation_difference _matrix
        3) labels: num_data_points NumPy array (all elements should be zeros
           and ones; 1: trajectory 1 is preferred; 0: trajectory 0 is preferred)
        2)-5): same as the descriptions in the feedback function.

    Output: the objective function evaluated at the given point (f).
    """

    obj = 0.5 * f @ GP_prior_cov_inv @ f

    num_preferences = visitation_difference_matrix.shape[0]

    for k in range(num_preferences):   # Go through each pair of preferences
        label = int(labels[k])
        visits = visitation_difference_matrix[k]
        if label == 0.5 or not np.any(visits): # if visit is not all zeros
            continue

        sum = 0
        sqr = 0
        for i in range(visitation_difference_matrix.shape[1]):
            sum += visits[i] * f[i]
            sqr += visits[i] ** 2

        z = sum / (preference_noise * np.sqrt(sqr))

        obj -= np.log(norm.cdf(z))

    return obj


def preference_GP_gradient(f, visitation_difference_matrix, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the gradient of the optimization objective function for finding
    the posterior mean of the GP preference model (at a given point).

    Inputs:
        1) f: the "point" at which to evaluate the gradient. This is a length-n
           vector, where n is the number of points over which the posterior
           is to be sampled.
        2)-5): same as the descriptions in the feedback function.

    Output: the objective function's gradient evaluated at the given point (f).
    """

    grad = GP_prior_cov_inv @ f    # Initialize to 1st term of gradient

    num_preferences = visitation_difference_matrix.shape[0]
    num_sa_pair = visitation_difference_matrix.shape[1]

    for m in range(grad.shape[0]):

        value = 0
        for k in range(num_preferences):   # Go through each pair of preference

            label = int(labels[k])
            visits = visitation_difference_matrix[k]

            if label == 0.5 or not np.any(visits):
                continue

            sum = 0
            sqr = 0
            for i in range(num_sa_pair):
                sum += visits[i] * f[i]
                sqr += visits[i]**2
            z = sum / (preference_noise * np.sqrt(sqr))

            value += (norm.pdf(z) / norm.cdf(z)) / (sqr * preference_noise) * visits[m]

        grad[m] -= value
    return grad

def preference_GP_hessian(f, visitation_difference_matrix, labels, GP_prior_cov_inv, preference_noise):
    """
    Evaluate the Hessian matrix of the optimization objective function for
    finding the posterior mean of the GP preference model (at a given point).

    Inputs:
        1) f: the "point" at which to evaluate the Hessian. This is
           a length-n vector, where n is the number of points over which the
           posterior is to be sampled.
        2)-5): same as the descriptions in the feedback function.

    Output: the objective function's Hessian matrix evaluated at the given
            point (f).
    """

    num_preferences = visitation_difference_matrix.shape[0]
    num_sa_pair = visitation_difference_matrix.shape[1]

    Lambda = np.zeros(GP_prior_cov_inv.shape)

    for k in range(num_preferences):   # Go through each pair of preference
        label = int(labels[k])
        visits = visitation_difference_matrix[k]

        if label == 0.5 or not np.any(visits):
            continue
        sum = 0
        sqr = 0
        for i in range(num_sa_pair):
            sum += visits[i] * f[i]
            sqr += visits[i] **2
        z = sum / (preference_noise * np.sqrt(sqr))

        c = (norm.pdf(z) / norm.cdf(z)) / sqr * (z + norm.pdf(z) / norm.cdf(z))

        Lambda += c * visits.reshape((num_sa_pair,1)) @ visits.reshape((1,num_sa_pair))

    return GP_prior_cov_inv + Lambda


