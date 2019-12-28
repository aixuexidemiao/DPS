# -*- coding: utf-8 -*-
"""
Implementation of EPMC algorithm (Wirth and Furnkranz, 2013) with probabilistic
temporal credit assignment.
"""

import numpy as np
from collections import defaultdict
import sys
import scipy.io as io
import os
from multiprocessing import Pool

from RiverSwim import RiverSwimPreferenceEnv


"""
Decide on a deterministic action to take in each state, choosing uniformly
over the actions.
"""
def initialize_policy(num_states, num_actions):

    policy = np.zeros((num_states, num_actions))

    # Random actions to take in each state:
    actions = np.random.choice(num_actions, size = num_states)
    
    for state in range(num_states):
        
        policy[state, actions[state]] = 1
        
    return policy
    
"""
Return a deterministic policy that acts greedily with respect to an inputted
Q-function.
"""
def get_deterministic_policy(Q):
    
    [num_states, num_actions] = Q.shape
    
    policy = np.zeros((num_states, num_actions))
    
    # Select action in each state that has the highest Q-value. Break ties
    # randomly.
    for state in range(num_states):
        
        max_idxs = np.where(Q[state, :] == np.max(Q[state, :]))[0]
        
        policy[state, np.random.choice(max_idxs)] = 1
        
    return policy

"""
Return EXP-3 policy based on inputted Q-function. Also, return updated values 
of G, to be used in future calls to this function.
"""
def get_EXP3_policy(Q, eta, G_previous):
    
    num_actions = Q.shape[1]
    
    # Update the policy:
    policy = np.exp((eta / num_actions) * G_previous)    
    policy = (policy.T / policy.sum(axis=1)).T
    policy = eta / num_actions + (1 - eta) * policy
    
    # Update G:
    G = G_previous + Q / policy

    return policy, G

"""
Convert from probabilities P(a' > a | s) to Q-function.
"""
def prob_to_Q(preference_probs):
    
    [num_states, num_actions, _] = preference_probs.shape
    
    Q = np.ones((num_states, num_actions))
    
    for state in range(num_states):
        for action in range(num_actions):
            
            denom = 2 - num_actions
            
            for action_ in range(num_actions):
                
                if action_ != action:
                    
                    if action > action_:
                        prob = preference_probs[state, action, action_]
                    else:
                        prob = 1 - preference_probs[state, action_, action]
                    
                    denom += 1 / prob
            
            Q[state, action] /= denom
    
    return Q
    
    
"""
Perform policy improvement step.
"""
def policy_improvement(P_prev, P_sampled, alpha):
    
    return (1 - alpha) * P_prev + alpha * P_sampled
    

"""
Determine new state-action preference information given a newly-sampled 
preference.
"""
def sampled_state_action_preference_probs(T1, T2, preference, P_prev, 
                                          method = 2):
    
    [num_states, num_actions, _] = P_prev.shape
    
    # Unpack the data:
    states_T1 = T1[0][:-1]
    actions_T1 = T1[1]
    states_T2 = T2[0][:-1]
    actions_T2 = T2[1]
    
    # Set of overlapping states:
    S = np.intersect1d(states_T1, states_T2)
    
    n = len(S)     # Number of overlapping states
    
    # Values to use as samples:
    sample_vals = [(n + 1) / (2*n), (n - 1) / (2*n)]

    # Use a defaultdict object to keep track of all preference samples:
    samples = defaultdict(lambda: [])
    
    for overlapping_state in S:  # Consider each overlapping state.
        
        # Indices where this overlapping state occurs:
        T1_idxs = np.where(states_T1 == overlapping_state)[0]
        T2_idxs = np.where(states_T2 == overlapping_state)[0]
        
        if method == 1:
        
            # Consider each pair:
            for idx1 in T1_idxs:
                
                a1 = actions_T1[idx1]
                
                for idx2 in T2_idxs:
                    
                    a2 = actions_T2[idx2]
                    
                    if a1 > a2:
                        
                        samples[overlapping_state, a1, a2].append(\
                               sample_vals[preference])
                        
                    elif a1 < a2:
                        
                        samples[overlapping_state, a2, a1].append(\
                               sample_vals[1 - preference])
                        
        elif method == 2:
            
            actions_1 = np.unique(actions_T1[T1_idxs])
            actions_2 = np.unique(actions_T2[T2_idxs])
            
            for a1 in actions_1:
                for a2 in actions_2:
                    
                    if a1 > a2:
                        
                        samples[overlapping_state, a1, a2].append(\
                               sample_vals[preference])
                        
                    elif a1 < a2:
                        
                        samples[overlapping_state, a2, a1].append(\
                               sample_vals[1 - preference])                   
            
        else:
            
            print('ERROR: Method should be either 1 or 2.')
            sys.exit()
                    
    # Initialize probabilities to their previous values, so that for non-
    # updated values, the probabilities will not change in the policy 
    # improvement step:
    P = np.copy(P_prev)            
            
    # Take mean of sampled values in each case.
    for s in range(num_states):
        for a1 in range(num_actions):
            for a2 in range(num_actions):
                
                vals = samples[s, a1, a2]
                
                if len(vals) > 0:
                    P[s, a1, a2] = np.mean(vals)
    
    return P


def EPMC_probabilistic_credit_assignment(args):
    
    num_iterations = args[0]
    eta = args[1]
    alpha = args[2]
    env = args[3]
    time_horizon = args[4]
    noise = args[5]
    save_folder = args[6]
    seed = args[7]
    run_num = args[8]

    print('Eta %2.1f, alpha %2.1f, noise %3.1f, run %i: hello!' % (eta,
                                                    alpha, noise, run_num))
    
    np.random.seed(seed)
    
    num_states = env.nS
    num_actions = env.nA
    
    # Sample initial deterministic policy:
    determ_policy = initialize_policy(num_states, num_actions)
    
    # Initialize G values for use with EXP3:
    G = np.zeros((num_states, num_actions))
    
    # Initialize state-action preference information:
    pref_probs = 0.5 * np.tril(np.ones((num_states, num_actions, num_actions)))
    
    # Initial EXP3 policy prefers all actions equally:
    EXP3_policy = (1/num_actions) * np.ones((num_states, num_actions))

    # To store performance results:
    step_rewards = -np.ones(num_iterations * time_horizon * 2)
    step_count = 0
    
    for i in range(num_iterations):
        
        trajectories = []    # Trajectories to be sampled in this iteration
        
        already_visited_states = np.empty(0)
        
        for j in range(2):   # Roll out 2 trajectories per iteration

            # State and action sequence to be sampled in this trajectory:
            state_sequence = np.empty(time_horizon + 1)
            action_sequence = np.empty(time_horizon)
        
            env.reset()
            state = env.state
            
            for t in range(time_horizon):        
                
                # Select next action via EXP3 policy if 1st trajectory sample
                # or visiting an overlapping state:
                if j == 1 or state in already_visited_states:
                    
                    action = np.random.choice(num_actions, p = EXP3_policy[state, :])
                    
                else:    # Otherwise, use deterministic policy.
                    
                    action = np.argmax(determ_policy[state, :])
                
                next_state, _, _ = env.step(action)
                next_state = next_state[0]
                
                state_sequence[t] = state
                action_sequence[t] = action

                state = next_state
                
                # Tracking rewards for evaluation purposes
                step_rewards[step_count] = env.get_step_reward(state, 
                            action, next_state)
                step_count += 1
            
            state_sequence[-1] = next_state
            trajectories.append([state_sequence, action_sequence])  
            
            # Update list of visited states:
            already_visited_states = np.unique(state_sequence[:-1])
            
        # Obtain a preference between the two trajectories:
        preference = env.get_trajectory_preference(trajectories[0], 
                                                trajectories[1], True, noise)
        
        # Determine new state-action preference information given newly-
        # sampled preference.
        pref_probs_sampled = sampled_state_action_preference_probs(trajectories[0], 
                                    trajectories[1], preference, pref_probs) 
        
        # Policy improvement step:
        pref_probs = policy_improvement(pref_probs, pref_probs_sampled, alpha)
    
        # Convert from probabilities P(a' > a | s) to Q-function.
        Q = prob_to_Q(pref_probs)    

        # Deterministic policy for next iteration:
        determ_policy = get_deterministic_policy(Q)
        
        # EXP3 policy for next iteration:
        EXP3_policy, G = get_EXP3_policy(Q, eta, G)
        
        
    # Save output file:
    save_filename = save_folder + 'Iter_' + str(num_iterations) + \
        '_alpha_' + str(alpha) + '_eta_' + str(eta) + '_noise_' + str(noise) + \
        '_run_' + str(run_num) + '.mat'
    
    io.savemat(save_filename, {'step_rewards': step_rewards,
               'iteration': num_iterations, 'Q': Q})        
        
        
# Define constants:
time_horizon = 50
num_iterations = 400

# Hyperparameters:
etas = np.arange(0.1, 1, 0.1)
alphas = np.arange(0.1, 1, 0.1)
noise_params = [0.1, 0.5, 1, 1000]

run_nums = np.arange(100)

save_folder = 'EPMC_probabilistic/' 

if not os.path.isdir(save_folder):
    os.mkdir(save_folder) 
    
# Initialize RiverSwim environment:
env = RiverSwimPreferenceEnv(num_states = 6)

# Sample random seeds:
num_seeds = len(run_nums) * len(etas) * len(alphas) * len(noise_params)

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

# Arguments for processes:
count = 0
args_list = []

for eta, alpha, noise_param, run in [(eta, alpha, noise_param, run) \
                                     for eta in etas for alpha in alphas \
                                     for noise_param in noise_params 
                                     for run in run_nums]:
    
    args_list.append((num_iterations, eta, alpha, env, time_horizon, 
                          noise_param, save_folder, seeds[count], run))
    
    count += 1

# Run processes sequentially:
for args in args_list:
    EPMC_probabilistic_credit_assignment(*args)

# Launch a pool of worker processes to calculate results for these cases:
#pool = Pool(processes = 6)
#pool.map(EPMC_probabilistic_credit_assignment, args_list)

