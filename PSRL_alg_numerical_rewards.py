# -*- coding: utf-8 -*-
"""
The PSRL algorithm, as described in Osband et al. (2013), with numerical
reward feedback.

Test PSRL with the RiverSwim environment.
"""

import numpy as np
from collections import defaultdict
import os
import scipy.io as io

from RiverSwim import RiverSwimEnv
from ValueIteration import value_iteration


# Folder for saving results:
save_folder = 'PSRL_numerical_rewards/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    
# Finite or infinite-horizon value iteration?
finite_horizon_iteration = True

# Define constants:
time_horizon = 50
num_policies = 1  # Num. policies to request from advance function at a time
# Number of times to run each policy:
num_policy_runs = 1

epsilon = 0      # Do not use epsilon-greedy control

"""
This function contains the code of the PSRL algorithm.
"""
def PB_PSRL_Self_Sparring(inputs):
    
    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1]
    num_policy_runs = inputs[2]
    epsilon = inputs[3]
    run_num = inputs[4]
    save_folder = inputs[5]
    seed = inputs[6]
    finite_horizon_iteration = inputs[7]
    diri_prior = inputs[8]
    NG_prior_params = inputs[9]
        
    print('Run %i: hello!' % run_num)
    
    np.random.seed(seed)

    # Initialize RiverSwim environment:
    env = RiverSwimEnv(num_states = 6)
    num_states = env.nS
    num_actions = env.nA

    # Initialize dictionary object to hold state transition information. This
    # is initially set to the Dirichlet prior (all ones = uniform distribution),
    # and is updated as state transition data is accumulated.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: \
                                            diri_prior * np.ones(num_states)))
    
    # Initialize posterior parameters used to sample from reward model:
    NG_params = np.tile(NG_prior_params, (num_states, num_actions, 1))

    # Store data on how many times each state-action pair gets visited:
    visit_counts = np.zeros((num_states, num_actions))
    
    # Reward observations associated with each state-action:
    reward_samples = defaultdict(lambda: [])

    """
    Here is the main loop of the PSRL algorithm.
    """
    
    # Stopping condition variables:
    converged = False    # Convergence flag
    max_iter = 400

    # To store results:
    step_rewards = -np.ones(max_iter * num_policies * num_policy_runs * time_horizon)
    
    iteration = 0
    step_count = 0

    while iteration < max_iter and not converged:
        
        print('Run %i: new iteration! Count = %i' % (run_num, iteration + 1))
 
        # Sample policies:
        policies, reward_models = advance(num_policies, dirichlet_posterior, 
                    NG_params, epsilon, num_states, num_actions, time_horizon, 
                    finite_horizon_iteration)
    
        for i in range(num_policy_runs):
            
            for policy in policies:    # Roll out 2 action sequences
        
                env.reset()
                
                for t in range(time_horizon):        
                    
                    state = env.state
                    
                    if finite_horizon_iteration:
                        action = np.random.choice(num_actions, p = policy[t, state, :])
                    else:
                        action = np.random.choice(num_actions, p = policy[state, :])
                    
                    next_state, reward, _, _ = env.step(action)
                    next_state = next_state[0]
                    
                    # Update state transition information:
                    dirichlet_posterior[state][action][next_state] += 1
                    
                    # Update state/action visits information:
                    visit_counts[state][action] += 1
                    
                    # Store reward information:
                    reward_samples[state, action].append(reward)
                    
                    # Tracking rewards for evaluation purposes
                    step_rewards[step_count] = reward
                    step_count += 1
       
        # Call feedback function to update the reward posterior:
        NG_params = feedback(NG_prior_params, visit_counts, reward_samples,
                             num_states, num_actions)
        
        iteration += 1
        
        # Save results from this iteration:       
        if np.mod(iteration, max_iter) == 0:
            
            save_filename = save_folder + 'Iter_' + str(iteration) + '_params_' \
                            + str(diri_prior) + '_' + str(NG_prior_params[0]) + \
                            '_' + str(NG_prior_params[1]) + '_' + \
                            str(NG_prior_params[2]) + '_' + \
                            str(NG_prior_params[3]) + '_run_' + str(run_num) + \
                            '.mat'
            
            io.savemat(save_filename, {'policies': policies, 
                       'rewards': reward_models, 'step_rewards': step_rewards,
                       'iteration': iteration})

"""
This function samples from the state transition and reward posteriors, and 
returns the requested number of policies. 
"""
def advance(num_policies, dirichlet_posterior, NG_params, epsilon, num_states, 
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
    
        # Sample reward function from Normal-Gamma posterior:
        R = np.empty((num_states, num_actions))
        
        for s in range(num_states):
            for a in range(num_actions):
                
                gamma_sample = np.random.gamma(NG_params[s, a, 2], 1 / NG_params[s, a, 3])
                R[s, a] = np.random.normal(NG_params[s, a, 0], 
                         (NG_params[s, a, 1] * gamma_sample)**(-0.5))
                
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
This function updates the Normal-Gamma reward posterior.
"""
def feedback(NG_prior_params, visit_counts, reward_samples, num_states, 
             num_actions):
            
    # Calculate posterior parameters of Normal-Gamma model:    
    NG_params = np.empty((num_states, num_actions, 4))
    
    mu0 = NG_prior_params[0]     # Unpack prior parameters
    k0 = NG_prior_params[1]
    alpha0 = NG_prior_params[2]
    beta0 = NG_prior_params[3]
    
    for s in range(num_states):
        for a in range(num_actions):
            
            n = visit_counts[s, a]
            
            if n == 0:
                NG_params[s, a] = NG_prior_params
                continue
            
            samples = np.array(reward_samples[s, a])
            avg = np.mean(samples)
            
            NG_params[s, a, 0] = (k0 * mu0 + n * avg) / (k0 + n)
            NG_params[s, a, 1] = k0 + n 
            NG_params[s, a, 2] = alpha0 + n/2
            NG_params[s, a, 3] = beta0 + 0.5 * np.sum((samples - avg)**2) + \
                                 k0 * n * (avg - mu0)**2 / (2 * (k0 + n))
    
    return NG_params
    

"""
Make a list of parameters for calling the Self-Sparring algorithm. Then, either
launch a multiprocessing pool to calculate them in parallel, or process them
sequentially.
"""    
args_list = []      # List of arguments for the processes that we will run

run_nums = np.arange(1)    # Number of runs for each set of parameter values

# Prior parameters:
diri_prior = 1
NG_prior_params = [[0.5, 0.5, 1, 1], [1, 1, 1, 1]]

num_seeds = len(run_nums) * len(NG_prior_params)

# Generate random seeds, using the full range of possible values and making 
# sure that no two are the same:
seeds = np.empty(num_seeds)

seeds[0] = np.random.choice(2**32)

for i in range(1, num_seeds):
    
    seed = np.random.choice(2**32)
    
    while np.any(seed == seeds[:i]):
        
        seed = np.random.choice(2**32)
        
    seeds[i] = seed
    
seeds = seeds.astype(int)

count = 0

for NG_prior in NG_prior_params:

    for run in run_nums:
    
        args_list.append((time_horizon, num_policies, num_policy_runs, epsilon, 
                          run, save_folder, seeds[count], finite_horizon_iteration,
                          diri_prior, NG_prior))
        
        count += 1
        
# Run processes sequentially:
for args in args_list:
    PB_PSRL_Self_Sparring(args)
