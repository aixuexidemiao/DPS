# -*- coding: utf-8 -*-
"""
current
"""

import numpy as np
from collections import defaultdict
import scipy.io as io
import itertools
import os

from RiverSwim import RiverSwimPreferenceEnv
from Preference_GP_learning_RL import advance, feedback


"""
This function contains the code of the DPS algorithm with a GP model over
state-action rewards in River Swim Environment.
"""
def DPS_Preference(inputs):
    '''
    # Unpack input parameters:
    time_horizon = 20
    num_policies = 2
    num_policy_runs = 1 # no use
    epsilon = 0 # deterministic policy
    run_num = 1 # to store the data for different runs

    save_folder = 'DPS_Preference_RiverSwim/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    seed = 19961202
    finite_horizon_iteration = True

    # Hyperparams in kernel definition
    kernel_variance = 1        # signal_variance
    kernel_lengthscales = [0,0]       # use 0 in River_Swim; kenel_lengthscales for each dim in state-action pair
    GP_noise_var = 0.03     # noise_variance

    preference_noise = 0.01
    num_states = 6
    num_actions = 2
    '''

    # Unpack input parameters:
    time_horizon = inputs[0]
    num_policies = inputs[1]
    num_policy_runs = inputs[2]
    epsilon = inputs[3]
    run_num = inputs[4]
    save_folder = inputs[5]
    seed = inputs[6]
    finite_horizon_iteration = inputs[7]
    max_iter = inputs[8]

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    hyperparams = inputs[9]
    kernel_variance = hyperparams['var']
    kernel_lengthscales = hyperparams['length']
    GP_noise_var = hyperparams['noise']

    preference_noise = inputs[10]
    num_states = inputs[11]
    num_actions = inputs[12]

    print('GPR, noise param %.2f, run %i: hello!' % (preference_noise,
                                                       run_num))

    np.random.seed(seed)

    # Initialize RiverSwim environment:
    env = RiverSwimPreferenceEnv(num_states = 6)
    num_s_a = num_states * num_actions # 根据环境设置参数

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
    GP_prior_cov += GP_noise_var * np.eye(num_s_a)

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

    while iteration < max_iter and not converged:

        skip = False; # check whether there is a preference

        print('GPR, noise param %.2f, run %i: count = %i,pref_count: %d'% (preference_noise,
                                                       run_num, iteration + 1,pref_count))
        X = visitation_difference_matrix[:pref_count,:]
        y = trajectory_return_labels[:pref_count,1]     # index of preferred trajectory, 0 or 1

        # Call feedback function to update the GP model:
        GP_model = feedback(X, y, GP_prior_cov_inv, preference_noise)
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
                        trajectories[1], False, preference_noise)

            # skip the same trajectories 不存相关visitation and keep the pref_count data不变
            if preference == 0.5:
                skip = True;
                break

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
        if not skip:
            pref_count += 1

        # Save results from this iteration:
        if np.mod(iteration, max_iter) == 0:

            save_filename = save_folder + 'Iter_' + str(iteration) + '_RBF_' + \
                str(kernel_variance) + '_' + str(kernel_lengthscales) + '_' + \
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
    # time_horizen
    for t in range(len(actions)):

        visit_counts[num_actions * states[t] + actions[t]] += 1

    return visit_counts.reshape((1,num_states * num_actions))

