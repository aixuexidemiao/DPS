# -*- coding: utf-8 -*-
"""
DPS algorithm with Gaussian process regression credit assignment to infer 
state-action rewards from trajectory preference.

This version is intended to be used with a one-dimensional state-space,
but can be easily extended to higher-dimensional ones.

"""

import numpy as np
from collections import defaultdict
import scipy.io as io
import itertools

from RandomMDPEnv import RandomMDPPreferenceEnv
from ValueIteration import value_iteration


"""
This function contains the code of the DPS algorithm with a GP model over
state-action rewards.
"""
def DPS_GP_reg(inputs):
    
    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1]
    num_policy_runs = inputs[2]
    epsilon = inputs[3]
    run_num = inputs[4]
    save_folder = inputs[5]
    seed = inputs[6]
    finite_horizon_iteration = inputs[7]
    
    hyperparams = inputs[8]
    kernel_variance = hyperparams['var']
    kernel_lengthscales = hyperparams['length']
    GP_noise_var = hyperparams['noise']

    preference_noise = inputs[9]
    num_states = inputs[10]
    num_actions = inputs[11]
        
    print('GPR, noise param %.1f, run %i: hello!' % (preference_noise, 
                                                       run_num))
    
    np.random.seed(seed)

    # Initialize Random MDP environment:
    env = RandomMDPPreferenceEnv(num_states, num_actions)
    num_s_a = num_states * num_actions
    
    # Initialize prior mean and covariance for GP model:
    GP_prior_mean = 0.5 * np.ones(num_s_a)
	
    # Map for converting state-action pair index to indices within each state
    # and action dimension:
    ranges = []
    ranges.append(np.arange(num_states))
    ranges.append(np.arange(num_actions))
    
    state_action_map = list(itertools.product(*ranges))
    
    # Initialize covariance for GP model:
    GP_prior_cov = kernel_variance * np.ones((num_s_a, num_s_a))
        
    for i in range(num_s_a):
        
        x1 = state_action_map[i]
        
        for j in range(num_s_a):
            
            x2 = state_action_map[j]
            
            for dim, lengthscale in enumerate(kernel_lengthscales): # 2 dimensions both are 0 for River Swim
                
                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((x2[dim] - x1[dim]) / \
                                lengthscale)**2)
                    
                elif lengthscale == 0 and x1[dim] != x2[dim]:
                    
                    GP_prior_cov[i, j] = 0 # 0 out of diagonal # RiverSwim Identity matrix
                    
    GP_prior_cov += GP_noise_var * np.eye(num_s_a)	
                    
    GP_prior = {'mean': GP_prior_mean, 'cov': GP_prior_cov}
    
    # Initially, GP model is just the prior, since we don't have any data:
    GP_model = {'mean': GP_prior_mean, 'cov': GP_prior_cov}
    
    # Initialize dictionary object to hold state transition information. This
    # is initially set to the Dirichlet prior (all ones = uniform distribution),
    # and is updated as state transition data is accumulated.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: np.ones(num_states)))    

    # Store data on how often each trajectory visits each state:
    visitation_matrix = np.empty((0, num_states * num_actions))
    
    # Trajectory return labels:
    trajectory_return_labels = np.empty((0, 1))

    """
    Here is the main loop of the PB-PSRL algorithm.
    """
    
    # Stopping condition variables:
    converged = False    # Convergence flag
    max_iter = 200

    # To store results:
    step_rewards = -np.ones(max_iter * num_policies * num_policy_runs * time_horizon)
    
    iteration = 0
    step_count = 0

    while iteration < max_iter and not converged:
        
        print('GPR, noise param %.1f, run %i: count = %i' % (preference_noise, 
                                                       run_num, iteration + 1))
 
        # Sample policies:
        policies, reward_models = advance(num_policies, dirichlet_posterior, 
                GP_model, epsilon, num_states, num_actions, time_horizon, 
                finite_horizon_iteration, run_num)
    
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
                    step_rewards[step_count] = env.get_step_reward(state, action, next_state)
                    step_count += 1
                
                state_sequence[-1] = next_state #？
                trajectories.append([state_sequence, action_sequence]) 
                
            # Obtain an expert preference between the 2 trajectories:
            preference = env.get_trajectory_preference(trajectories[0], 
                        trajectories[1], True, preference_noise)
            
            # Store state visitation information corresponding to the 2 
            # trajectories:   
            for tr in trajectories:
                
                state_action_visit_counts = get_state_action_visit_counts(tr, 
                                        num_states, num_actions).reshape((1, 
                                                num_states * num_actions))
                visitation_matrix = np.vstack((visitation_matrix, 
                                               state_action_visit_counts)) #concatenation
            
            # Store preference information:
            trajectory_return_labels = np.vstack((trajectory_return_labels, 
                            np.reshape([1 - preference, preference], (2, 1))))
       
        # Call feedback function to update the GP model:
        GP_model = feedback(GP_prior, visitation_matrix, 
                            trajectory_return_labels, run_num)
        
        iteration += 1
        
        # Save results from this iteration:       
        if np.mod(iteration, max_iter) == 0:
            
            save_filename = save_folder + 'Iter_' + str(iteration) + '_RBF_' + \
                str(kernel_variance) + '_' + str(kernel_lengthscale) + '_' + \
                str(GP_noise_var) + '_noise_' + str(preference_noise) + '_eps_' + str(epsilon) + '_run_' \
                 + str(run_num) + '.mat'
            
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
def advance(num_policies, dirichlet_posterior, GP_model, epsilon, num_states, 
            num_actions, time_horizon, finite_horizon_iteration, run_num):

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
    
        # Sample reward function from GP posterior:
        R = np.random.multivariate_normal(GP_model['mean'], GP_model['cov'])
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
This function updates the GP posterior over rewards based on the new preference 
data.
"""
def feedback(GP_prior, visitation_matrix, trajectory_return_labels, run_num):
            
    prior_mean = GP_prior['mean']
    prior_cov = GP_prior['cov']

    num_samples = len(trajectory_return_labels)

    # Calculate the covariance between the rewards and trajectory returns:
    K_rR = prior_cov @ np.transpose(visitation_matrix)
    
    # Calculate covariance matrix of the trajectory returns:
    K_R = visitation_matrix @ K_rR
            
    # Calculate the posterior mean:
    #print(visitation_matrix)
    intermediate_term = K_rR @ np.linalg.inv(K_R + 1e-5 * np.eye(num_samples))
    post_mean = prior_mean + intermediate_term @ \
        (trajectory_return_labels.flatten() - visitation_matrix @ prior_mean)
    
    # Calculate the posterior covariance matrix:
    post_cov = prior_cov - intermediate_term @ np.transpose(K_rR)
    
    # Return the GP posterior:
    return {'mean': post_mean, 'cov': post_cov}
    
