# -*- coding: utf-8 -*-
"""
current
"""

import numpy as np
from collections import defaultdict
import scipy.io as io
import itertools
import os

from SimpleMountainCarPreferenceEnv import SimpleMountainCarPreferenceEnv
from Preference_GP_learning_RL_sigmoid import advance, feedback


"""
This function contains the code of the DPS algorithm with a GP model over
state-action rewards in River Swim Environment.
"""
def DPS_Preference_MountainCar(inputs):
    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1] # no use in face, compare the previous trajectary and current trajectory
    epsilon = inputs[2]
    run = inputs[3]
    save_folder = inputs[4]
    finite_horizon_iteration = inputs[5]
    max_iter = inputs[6]

    hyperparams = inputs[7]
    kernel_variance = hyperparams['var']
    kernel_lengthscales = hyperparams['length']
    kernel_noise = hyperparams['noise']

    alg_pref_noise = inputs[8]
    noisy_preference = inputs[9] # Boolean whether user give deterministic preference
    user_noise = inputs[10]

    diri_prior = inputs[11]

    np.random.seed(2020)

    print('MountainCar, user noise %.2f, alg_pref_noise:%.4f, run %i: hello!' % (user_noise, alg_pref_noise, run))

    #np.random.seed(seed)

    # Initialize Mountain Car Preference environment:
    env = SimpleMountainCarPreferenceEnv()

    num_actions = env.nA

    # State space ranges
    high = env.high
    low = env.low

    # Number of states in each discretized state space dimension (1st dimension
    # corresponds to position; 2nd dimension is velocity):
    # Wirth：discretize the state space into 10 equal width bins for each dimension
    states_per_dim = [10, 10]
    state_dims = len(states_per_dim)

    thresholds = []  # Thresholds to use for discretizing state space

    for i in range(state_dims):
        thresholds.append(np.linspace(low[i], high[i], num = states_per_dim[i] + 1)[:-1])

    num_states = np.prod(states_per_dim) + 1 # plus one goal state
    num_s_a = num_states * num_actions     # Number of state-action pairs

    # Map for converting overall index of state-action pair to the indices
    # within each state dimension, and the action:
    ranges = []
    for i in range(len(states_per_dim)):
        ranges.append(np.arange(states_per_dim[i]))

    ranges.append(np.arange(num_actions))

    state_action_map = list(itertools.product(*ranges))

    # add goal state to state_action_map
    goal_state = (10,10)
    for a in range(num_actions):
        state_action_map.append(goal_state + (a,))

    # Initialize prior covariance for GP model:

    GP_prior_cov = kernel_variance * np.ones((num_s_a, num_s_a))

    for i in range(num_s_a):

        x1 = state_action_map[i]

        for j in range(num_s_a):

            x2 = state_action_map[j]

            for dim, lengthscale in enumerate(kernel_lengthscales):

                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((x2[dim] - x1[dim]) / lengthscale)**2)

                elif lengthscale == 0 and x1[dim] != x2[dim]:

                    GP_prior_cov[i, j] = 0

    GP_prior_cov += kernel_noise * np.eye(num_s_a)

    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)


    # Initialize dictionary object to hold state transition information. This
    # is initially set to the Dirichlet prior (all ones = uniform distribution),
    # and is updated as state transition data is accumulated.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda:  diri_prior * np.ones(num_states)))
    # every single state-action pair maintains a dirichlet model. The number of param is num_states for each model


    """
    Here is the main loop of the algorithm.
    """

    # Stopping condition variables:
    converged = False    # Convergence flag (not used)


    # Store data on how often each trajectory visits each state:
    visitation_matrix = np.empty((0, num_s_a))
    # Store n(t1, ) - n(t2, )
    visitation_difference_matrix = np.zeros((max_iter * num_policies, num_s_a)).astype(int)

    # Trajectory return labels:
    trajectory_return_labels = np.empty((0, 2))

    # To store results (for evaluation purposes only):
    episode_rewards = np.empty(max_iter)

    iteration = 0
    episode_count = 0
    pref_count = 0  # Keeps track of how many preferences are in the dataset

    new_preference = True # check whether there is a preference

    while iteration < max_iter and not converged:

        print('user_noise %f, alg_noise %f, run %i: iter = %i,pref_count: %d'% (user_noise, alg_pref_noise,
                                                                           run, iteration, pref_count))
        X = visitation_difference_matrix[:pref_count,:]
        y = trajectory_return_labels[:pref_count,1]     # index of preferred trajectory, 0 or 1

        if new_preference:
            # to save time, if new_preference, same data to feedback
            # Call feedback function to update the GP model:
            GP_model = feedback(X, y, GP_prior_cov_inv, alg_pref_noise)
            # dirichlet_posterior = dirichlet_posterior

        # Sample policy:
        policies, reward_models = advance(1, dirichlet_posterior,
                GP_model, epsilon, num_states, num_actions, time_horizon,
                finite_horizon_iteration)


        # Sample trajectories using these policies:
        trajectory = []



        # run the sampled policy
        policy = policies[0]
        state = env.reset() #[p,v]
        state = convert_state_values_to_state_idx(state, thresholds)  # converts the state into a discretized state index

        state_sequence = np.empty(time_horizon + 1)
        action_sequence = np.empty(time_horizon)

        for t in range(time_horizon):

            if finite_horizon_iteration:

                if len(policy[t, state, :])== 2:
                    print(policy[t, state, :])

                action = np.random.choice(num_actions, p = policy[t, state, :])
            else:
                action = np.random.choice(num_actions, p = policy[state, :])

            next_state = env.step(action) #[position,velocity]
            if env.goaled:
                next_state = 100 # index of goal_state
            else:
                next_state = convert_state_values_to_state_idx(next_state, thresholds)

            state_sequence[t] = state
            action_sequence[t] = action

            # Update state transition information using discretized state:
            dirichlet_posterior[state][action][next_state] += 1

            state = next_state

        state_sequence[-1] = next_state # add the last state into state_sequence
        trajectory = [state_sequence, action_sequence]
        # policy finished, add visitation_matrix and episode reward

        # Tracking rewards for evaluation purposes
        # print(env.done)
        episode_rewards[episode_count] = env.get_episode_return()

        # Store state visitation information corresponding to the current trajectory:
        cur_state_action_visit_count = get_state_action_visit_counts(trajectory, num_states, num_actions)
        visitation_matrix = np.vstack((visitation_matrix, cur_state_action_visit_count)) #concatenation

        iteration += 1

        episode_count += 1
        if episode_count == 1:
            continue

        # Run following if episode > 2
        # Obtain an expert preference between the last 2 trajectories:
        preference = env.get_preference(episode_rewards[episode_count-2],episode_rewards[episode_count-1],noisy_preference, user_noise)


        # new_preference the same trajectories 不存相关visitation and keep the pref_count data不变

        if preference == 0.5:
            new_preference = False
            continue
        else:
            new_preference = True
            pref_count += 1

        # Store preference information:
        trajectory_return_labels = np.vstack((trajectory_return_labels,[1 - preference, preference]))


        # store comparison result into visitation_difference_matrix
        state_action_visit_counts = np.empty((num_policies, num_states * num_actions))
        state_action_visit_counts[0] = visitation_matrix[episode_count-2]
        state_action_visit_counts[1] = cur_state_action_visit_count

        visitation_difference_matrix[pref_count,:] = \
            [state_action_visit_counts[preference][i] - state_action_visit_counts[1-preference][i] for i in range(num_s_a)]


        # Save results from this iteration:
        if np.mod(iteration, max_iter) == 0:

            # # Convert dynamics to a form that can be saved in a .mat file:
            # M = np.empty((num_states, num_actions, num_states))
            #
            # for i in range(num_states):
            #     for j in range(num_actions):
            #
            #         M[i, j, :] = dirichlet_posterior[i][j]

            save_filename = save_folder + 'userNoise_' + str(user_noise) + '_algNoise_' + str(alg_pref_noise) + '_run_' + str(run) + '.mat'

            io.savemat(save_filename, {'episode_rewards': episode_rewards})

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
    # time_horizen
    for t in range(len(actions)):

        visit_counts[num_actions * states[t] + actions[t]] += 1

    return visit_counts.reshape((1,num_states * num_actions))

"""
This function converts the state into a discretized state (with bins defined
according to the given thresholds). Then, it returns the state's scalar index,
which is used for updating the dynamics and reward models.
"""
def convert_state_values_to_state_idx(state, thresholds):

    # Convert to integer bins:
    num_dims = len(thresholds)
    state_bins = np.empty(num_dims)

    for i in range(num_dims):

        state_bins[i] = np.where(state[i] >= thresholds[i])[0][-1]

    # Convert to the state's scalar index:
    state_idx = 0
    prod = 1

    for i in range(num_dims - 1, -1, -1):

        state_idx += state_bins[i] * prod
        prod *= len(thresholds[i])

    return int(state_idx)



'''
run the optimize in SimpleMountainCar
'''


time_horizon = 500
num_policies = 2  # Num. policies to request from advance function at a time

epsilon = 0       # Do not use epsilon-greedy control
num_run = 1    # Number of runs for each set of parameter values


# Finite or infinite-horizon value iteration
finite_horizon_iteration = True
max_iter = 200

# GPR hyperparameters to use:
kernel_variance = 1e-7
kernel_lengthscales = [2, 2, 0] #use this in RiverSwim
kernel_noise = 0.005

GP_hyperparams = {'var': kernel_variance, 'length': kernel_lengthscales, 'noise': kernel_noise}

alg_pref_noises = [2]

noisy_preference = False # Boolean whether user give deterministic preference
user_noises = [0.1,0.5,1,1000]

if not noisy_preference:
    user_noises = [-99]

diri_prior = 0.0005

args_list = []      # List of arguments for the processes that we will run


# Folder for saving results:
save_folder = 'GP_Pref_MountainCar_sigmoid/' + 'Iter_' + str(max_iter) + '_Nrun' + str(num_run) + '_RBF_' + \
                str(kernel_variance) + '_' + str(kernel_lengthscales) + '_' + \
                str(kernel_noise) + '_algNoises_' + str(alg_pref_noises) +'/'
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)


for user_noise in user_noises:

    for alg_pref_noise in alg_pref_noises:

            for run in range(num_run):

                args_list.append((time_horizon, num_policies, epsilon, run, save_folder, finite_horizon_iteration, max_iter,
                                  GP_hyperparams, alg_pref_noise,noisy_preference, user_noise,diri_prior))

# Run processes sequentially:
for args in args_list:
    DPS_Preference_MountainCar(args)
