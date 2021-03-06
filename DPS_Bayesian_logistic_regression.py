# -*- coding: utf-8 -*-
"""
DPS algorithm using Bayesian logistic regression credit assignment to infer 
state-action rewards from trajectory preferences.

"""

import numpy as np
import scipy as sc
from collections import defaultdict
import scipy.io as io
from scipy.optimize import minimize

from RandomMDPEnv import RandomMDPPreferenceEnv
from ValueIteration import value_iteration


"""
This function contains the code of the DPS algorithm with a Bayesian 
logistic regression model over state rewards.
"""
def DPS_Bayes_log_reg(inputs):
    
    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1]
    num_policy_runs = inputs[2]
    epsilon = inputs[3]
    run_num = inputs[4]
    save_folder = inputs[5]
    seed = inputs[6]
    finite_horizon_iteration = inputs[7]
    prior_covariance = inputs[8]
    preference_noise = inputs[9]
    num_states = inputs[10]
    num_actions = inputs[11]
        
    print('Log reg, noise param %.1f, eps %1.2f, prior cov = %i, run %i: hello!' % \
          (preference_noise, epsilon, prior_covariance, run_num))
    
    np.random.seed(seed)

    # Initialize Random MDP environment:
    env = RandomMDPPreferenceEnv(num_states, num_actions)
    
    num_features = num_states * num_actions

    # Initialize prior mean and covariance for Bayesian logistic regression model:
    prior_mean = np.zeros(num_features)
    prior_cov = prior_covariance * np.eye(num_features)
    
    # Initially, model is just the prior, since we don't have any data:
    LR_prior_model = {'mean': prior_mean, 'cov': prior_cov}
    LR_model = {'mean': prior_mean, 'cov': prior_cov}
    
    # Initialize dictionary object to hold state transition information. This
    # is initially set to the Dirichlet prior (all ones = uniform distribution),
    # and is updated as state transition data is accumulated.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: np.ones(num_states)))    

    # Store data on how often each trajectory visits each state:
    feature_matrix = np.empty((0, num_features))
    
    # Trajectory return labels:
    trajectory_return_labels = np.empty((0, 1))

    """
    Here is the main loop of the Self-Sparring-VI algorithm.
    """
    
    # Stopping condition variables:
    converged = False    # Convergence flag
    max_iter = 250

    # To store results:
    step_rewards = -np.ones(max_iter * num_policies * num_policy_runs * time_horizon)
    
    iteration = 0
    step_count = 0

    while iteration < max_iter and not converged:
        
        print('Log reg, noise param %.1f, eps %1.2f, prior cov = %i, run %i: count = %i' \
              % (preference_noise, epsilon, prior_covariance, run_num, iteration + 1))
 
        # Sample policies:
        policies, reward_models = advance(num_policies, dirichlet_posterior, 
                LR_model, epsilon, num_states, num_actions, time_horizon, 
                finite_horizon_iteration)
    
        for i in range(num_policy_runs):
    
            # Sample trajectories using these policies:
            trajectories = []
            
            for policy in policies:    # Roll out 2 action sequences
        
                env.reset()
                        
                state_sequence = np.empty(time_horizon + 1)
                action_sequence = np.empty(time_horizon)
                
                for t in range(time_horizon):        
                    
                    state = env.state
                    
                    if finite_horizon_iteration:
                        action = np.random.choice(num_actions, p = policy[t, state, :])
                    else:
                        action = np.random.choice(num_actions, p = policy[state, :])
                    
                    next_state, _, _ = env.step(action)
                    next_state = next_state[0]
                    
                    state_sequence[t] = state
                    action_sequence[t] = action
                    
                    # Update state transition information:
                    dirichlet_posterior[state][action][next_state] += 1
                    
                    # Tracking rewards for evaluation purposes
                    step_rewards[step_count] = env.get_step_reward(state, 
                                action, next_state)
                    step_count += 1
                
                state_sequence[-1] = next_state
                trajectories.append([state_sequence, action_sequence]) 
                
            # Obtain an expert preference between the 2 trajectories:
            preference = env.get_trajectory_preference(trajectories[0], 
                                trajectories[1], True, preference_noise)
            
            # Store state visitation information corresponding to the 2 
            # trajectories.  
            visitations_1 = get_state_action_visit_counts(trajectories[0], 
                                        num_states, num_actions)
            visitations_2 = get_state_action_visit_counts(trajectories[1], 
                                        num_states, num_actions)
            features = visitations_1 - visitations_2
            
            feature_matrix = np.vstack((feature_matrix, features))
            
            # Store preference information. Preference gets mapped to 1 for the 
            # preferred trajectory and -1 for the dominated trajectory.
            trajectory_return_labels = np.vstack((trajectory_return_labels, 
                            np.reshape([-2*preference + 1], (1, 1))))
       
        # Call feedback function to update the Bayesian logistic regression model:
        if len(trajectory_return_labels) > 0:
            LR_model = feedback(LR_prior_model, feature_matrix, 
                                trajectory_return_labels)
        
        iteration += 1
        
        # Save results from this iteration:       
        if np.mod(iteration, max_iter) == 0:
            
            save_filename = save_folder + 'Iter_' + str(iteration) + '_param_' + \
                str(prior_covariance) + '_noise_' + str(preference_noise) + \
                '_run_' + str(run_num) + '.mat'
            
            io.savemat(save_filename, {'step_rewards': step_rewards,
                       'iteration': iteration})
                       
"""
This function returns a vector of how many times each state is visited in a 
trajectory.
"""
def get_state_visit_counts(trajectory, num_states):
    
    states = trajectory[0]
    
    return np.array([np.sum(states == i) for i in range(num_states)])

"""
This function returns a vector of how many times each state-action pair is 
visited in a trajectory.
"""
def get_state_action_visit_counts(trajectory, num_states, num_actions):
    
    states = trajectory[0].astype(int)
    actions = trajectory[1].astype(int)
    
    visit_counts = np.zeros(num_states * num_actions)
    
    for t in range(len(actions)):
        
        visit_counts[num_actions * states[t] + actions[t]] += 1
    
    return visit_counts

"""
This function samples from the state transition and reward posteriors, and 
returns the requested number of policies. 
"""
def advance(num_policies, dirichlet_posterior, LR_model, epsilon, num_states, 
            num_actions, time_horizon, finite_horizon_iteration):

    policies = []
    reward_models = []
    
    for i in range(num_policies):
        
        # Sample state transition dynamics from Dirichlet posterior:
        dynamics_sample = []
        
        for state in range(num_states):
            
            dynamics_sample_ = []
            
            for action in range(num_actions):
        
                dynamics_sample_.append(np.random.dirichlet(dirichlet_posterior[state][action]))
                
            dynamics_sample.append(dynamics_sample_)
    
        # Sample reward function from Bayesian logistic regression posterior:

        # Sample reward function from posterior:
        X = np.random.normal(size = len(LR_model['mean']))
        evals, evecs = np.linalg.eig(LR_model['cov'])
        R = LR_model['mean'] + evecs @ np.diag(np.sqrt(evals)) @ X
        
        R = R.reshape([num_states, num_actions])

        # Determine horizon to use for value iteration
        if finite_horizon_iteration:
            H = time_horizon
        else:
            H = np.infty
              
        # Value iteration to determine policy:             
        policies.append(value_iteration(dynamics_sample, R, num_states, 
                                        num_actions, epsilon = epsilon,
                                        H = H)[0])
        reward_models.append(R)
        
    return policies, reward_models   

"""
This function updates the conjugate Bayesian linear regression posterior over 
rewards based on the new preference data.
"""
def feedback(LR_prior_model, feature_matrix, labels):
      
    prior_mean = LR_prior_model['mean']
    prior_cov_inv = np.linalg.inv(LR_prior_model['cov'])

    num_features = feature_matrix.shape[1]

    # Solve convex optimization problem to obtain the posterior mean reward 
    # vector via Laplace approximation:    
    r_init = 0.5 * np.ones(num_features)    # Initial guess

    res = minimize(logreg_objective, r_init, args = (feature_matrix, labels, 
                   prior_mean, prior_cov_inv), method='L-BFGS-B', 
                   jac=logreg_obj_gradient)
    
    post_mean = res.x

    # Obtain posterior covariance approximation by evaluating the objective
    # function's Hessian at the posterior mean estimate:
    post_cov = np.linalg.inv(logreg_obj_hessian(post_mean, feature_matrix,
                                                labels, prior_cov_inv))

    # Helps if any eigenvalues have miniscule imaginary components that 
    # exist due to numerical errors
    L = np.linalg.cholesky(post_cov)
    post_cov = np.dot(L, L.T.conj())
    
    # Return the model posterior:
    return {'mean': post_mean, 'cov': post_cov}


"""
Objective function obtained via Laplace approximation, to be solved via
Bayesian logistic regression.

The label vector y is expected to contain -1's and 1's only.
"""
def logreg_objective(w, X, y, w0, V0_inv):
    
    obj = 0.5 * (w - w0) @ V0_inv @ (w - w0)
    
    for i in range(len(y)):
        
        obj += np.log(1 + np.exp(-y[i] * np.dot(X[i, :], w)))
        
    return obj
            

"""
Gradient of the objective function obtained via Laplace approximation.

The label vector y is expected to contain -1's and 1's only.
"""
def logreg_obj_gradient(w, X, y, w0, V0_inv):
    
    grad = V0_inv @ (w - w0)
    
    for i in range(len(y)):
        
        yi = y[i]; xi = X[i, :]
        
        grad -= yi * xi / (1 + np.exp(yi * np.dot(xi, w)))
        
    return grad  
    

"""
Hessian of the objective function obtained via Laplace approximation.

The label vector y is expected to contain -1's and 1's only.
"""
def logreg_obj_hessian(w, X, y, V0_inv):
    
    hess = V0_inv
    
    for i in range(len(y)):
        
        yi = y[i]; xi = X[i, :]
        xi_vec = xi.reshape((len(xi), 1))
        
        hess += (xi_vec @ np.transpose(xi_vec)) * np.exp(yi * np.dot(xi, w)) /   \
                (np.exp(yi * np.dot(xi, w)) + 1)**2
        
    return hess

