# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from collections import defaultdict
import scipy.io as io
import itertools
import os
import matplotlib.pyplot as plt

from RiverSwim import RiverSwimPreferenceEnv
from Preference_GP_learning_RL_sigmoid import advance, feedback


"""
This function contains the code of the DPS algorithm with a GP model over
state-action rewards in River Swim Environment.
"""
def DPS_Preference(inputs):

    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1]
    num_policy_runs = inputs[2]
    epsilon = inputs[3]
    run = inputs[4]
    save_folder = inputs[5]
    finite_horizon_iteration = inputs[6]
    max_iter = inputs[7]

    hyperparams = inputs[8]
    kernel_variance = hyperparams['var']
    kernel_lengthscales = hyperparams['length']
    kernel_noise = hyperparams['noise']

    alg_pref_noise = inputs[9]

    noisy_preference = inputs[10] # Boolean whether user give deterministic preference
    user_noise = inputs[11]
    num_states = inputs[12]
    num_actions = inputs[13]

    print('alg_pref_noise = %f, run %i: hello!' % (alg_pref_noise, run))

    # Initialize RiverSwim environment:
    env = RiverSwimPreferenceEnv(num_states = 6)
    num_s_a = num_states * num_actions

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

            for dim, lengthscale in enumerate(kernel_lengthscales):

                if lengthscale > 0:
                    GP_prior_cov[i, j] *= np.exp(-0.5 * ((x2[dim] - x1[dim]) / \
                                lengthscale)**2)

                elif lengthscale == 0 and x1[dim] != x2[dim]:
                    GP_prior_cov[i, j] = 0
    GP_prior_cov += kernel_noise * np.eye(num_s_a)

    GP_prior_cov_inv = np.linalg.inv(GP_prior_cov)


    # Initialize dictionary object to hold state transition information. This
    # is initially set to the Dirichlet prior (all ones = uniform distribution),
    # and is updated as state transition data is accumulated.
    dirichlet_posterior = defaultdict(lambda: defaultdict(lambda: np.ones(num_states)))
    # every single state-action pair maintains a dirichlet model. The number of param is num_states for each model


    """
    Here is the main loop of the algorithm.
    """

    # Stopping condition variables:
    converged = False    # Convergence flag


    # Store data on how often each trajectory visits each state:
    visitation_matrix = np.empty((0, num_states * num_actions))
    # Store n(t1, ) - n(t2, )
    visitation_difference_matrix = np.zeros((max_iter * num_policies, num_states * num_actions)).astype(int)

    # Trajectory return labels:
    trajectory_return_labels = np.empty((0, 2))

    # To store results:
    step_rewards = -np.ones(max_iter * num_policies * num_policy_runs * time_horizon)
    step_count = 0

    iteration = 0
    pref_count = 0  # Keeps track of how many preferences are in the dataset

    new_preference = True; # check whether there is a preference

    while iteration < max_iter and not converged:

        print('user_noise %f, alg_noise %f, run %i: iter = %i,pref_count: %d'% (user_noise, alg_pref_noise,
                                                                           run, iteration + 1, pref_count))
        X = visitation_difference_matrix[:pref_count,:]
        y = trajectory_return_labels[:pref_count,1]     # index of preferred trajectory, 0 or 1

        if new_preference:
            # to save time, if not new_preference, same data to feedback
            # Call feedback function to update the GP model:
            GP_model = feedback(X, y, GP_prior_cov_inv, alg_pref_noise)
            # plot_posterior(GP_model,num_s_a,save_folder,pref_count,alg_pref_noise)
            # dirichlet_posterior = dirichlet_posterior

        # Sample policies:
        policies, reward_models = advance(num_policies, dirichlet_posterior,
                GP_model, epsilon, num_states, num_actions, time_horizon,
                finite_horizon_iteration)

        # execute the optimal policies
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

                state_sequence[-1] = next_state # add the last state into state_sequence
                trajectories.append([state_sequence, action_sequence])

            # Obtain an expert preference between the 2 trajectories:
            preference = env.get_trajectory_preference(trajectories[0],
                                                       trajectories[1], noisy_preference, user_noise)

            if preference == 0.5:
                new_preference = False
                break
            else:
                new_preference = True

            # Store preference information:
            trajectory_return_labels = np.vstack((trajectory_return_labels,[1 - preference, preference]))

            # Store state visitation information corresponding to the 2 trajectories:
            state_action_visit_counts = np.empty((num_policies, num_states * num_actions))
            for tr in range(num_policies):

                state_action_visit_counts[tr] = get_state_action_visit_counts(trajectories[tr],
                                        num_states, num_actions)
                visitation_matrix = np.vstack((visitation_matrix,
                                               state_action_visit_counts[tr])) #concatenation
            # visitation_difference_matrix
            visitation_difference_matrix[pref_count,:] = \
                [state_action_visit_counts[preference][i] - state_action_visit_counts[1-preference][i] for i in range(num_s_a)]

        iteration += 1
        if new_preference:
            pref_count += 1

        # Save results from this iteration:
        if np.mod(iteration, max_iter) == 0:
            save_filename = save_folder + 'userNoise_' + str(user_noise) + '_algNoise_' + str(alg_pref_noise) + '_run_' \
                            + str(run) + '.mat'

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
    # time_horizen
    for t in range(len(actions)):

        visit_counts[num_actions * states[t] + actions[t]] += 1

    return visit_counts.reshape((1,num_states * num_actions))

"""
This function plots the posterior GP, num_sample_points = num_s_a
"""
def  plot_posterior(GP_model,num_sample_points,save_folder,pref_count,alg_pref_noise):
    mean = GP_model['mean']
    evecs = GP_model['cov_evecs']
    evals = GP_model['cov_evals']

    cov = np.dot(np.linalg.inv(evecs),np.dot(np.diag(evals),evecs))
    std_deviation = np.square(np.diag(cov))
    sample_points = np.linspace(0, num_sample_points-1, num_sample_points)

    fig = plt.figure()
    # Plot mean and deviation
    plt.errorbar(sample_points, mean, std_deviation, linestyle='None', marker='^')
    plt.plot(sample_points, mean)
    plt.xlabel('state_action pair')
    plt.ylabel('latent utility')
    plt.title('GP posterior, number of data = '+str(pref_count))

    folder = save_folder + '/posterior_' + 'alg_pref_noise_' + str(alg_pref_noise)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fig.savefig(folder + '/pref_count_'+str(pref_count)+'.png')


'''
run the optimize
'''
num_states = 6
num_actions = 2



time_horizon = 50
num_policies = 2  # Num. policies to request from advance function at a time
# Number of times to run each policy and request a comparison:
num_policy_runs = 1 # no use

epsilon = 0       # Do not use epsilon-greedy control
num_run = 100    # Number of runs for each set of parameter values


# Finite or infinite-horizon value iteration
finite_horizon_iteration = True
max_iter = 120

# GPR hyperparameters to use:
kernel_variance = 1
kernel_lengthscales = [0,0] #use this in RiverSwim
kernel_noise = 0.001

GP_hyperparams = {'var': kernel_variance, 'length': kernel_lengthscales, 'noise': kernel_noise}

alg_pref_noises = [2]
noisy_preference = True # Boolean whether user give deterministic preference
user_noises = [0.1,0.5,1,1000]
if not noisy_preference:
    user_noise = 'NA'


args_list = []      # List of arguments for the processes that we will run


# Folder for saving results:
save_folder = 'DPS_Preference_RiverSwim/sigmoid/' + 'Iter_' + str(max_iter) + '_Nrun' + str(num_run) + '_RBF_' + \
                str(kernel_variance) + '_' + str(kernel_lengthscales) + '_' + \
                str(kernel_noise) + '_algNoises_' + str(alg_pref_noises) +'/'
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

for user_noise in user_noises:

    for alg_pref_noise in alg_pref_noises:

            for run in range(num_run):

                args_list.append((time_horizon, num_policies, num_policy_runs, epsilon,
                                  run, save_folder, finite_horizon_iteration,max_iter,
                                  GP_hyperparams, alg_pref_noise,noisy_preference,
                                  user_noise,num_states,num_actions))

# Run processes sequentially:
for args in args_list:
    DPS_Preference(args)
