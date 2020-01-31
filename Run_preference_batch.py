#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run batch of DPS tests with River Swim environment.
- 1. execute DPS_Preference(), which stores the result
- 2. execute plot_cumulative_reward(), which extract the stored result
"""

import numpy as np
from Optimize_preference_GP_RiverSwim import DPS_Preference
#from Arxiv_paper_plots import plot_cumulative_reward

num_states = 6
num_actions = 2

# Finite or infinite-horizon value iteration?
finite_horizon_iteration = True
iteration = 600

time_horizon = 50
num_policies = 2  # Num. policies to request from advance function at a time
# Number of times to run each policy and request a comparison:
num_policy_runs = 1 # no use

epsilon = 0       # Do not use epsilon-greedy control

run_nums = np.arange(1)    # Number of runs for each set of parameter values

noise_params = [0.01]     # Preference noise parameters

"""Run GP Preference cases."""

# Folder for saving results:
save_folder = 'DPS_Preference_RiverSwim/'


# GPR hyperparameters to use:
kernel_variance = 1
kernel_lengthscales = [0,0] #use this in RiverSwim
GP_noise_var = [0.03]

GP_hyperparams = {'var': kernel_variance, 'length': kernel_lengthscales, 'noise': GP_noise_var}

# Sample the random seeds:
num_seeds = len(run_nums) * len(noise_params)

# Generate random seeds, using the full range of possible values and making
# sure that no two are the same. This is necessary if using multi-processing
# pool to launch processes simultaneously; otherwise, processes that launch
# at the same time will have the same random seed.
seeds = np.empty(num_seeds)

seeds[0] = np.random.choice(2**32)

for i in range(1, num_seeds):

    seed = np.random.choice(2**32)

    while np.any(seed == seeds[:i]):

        seed = np.random.choice(2**32)

    seeds[i] = seed

seeds = seeds.astype(int)

args_list = []      # List of arguments for the processes that we will run

count = 0

for noise_param in noise_params:

        for run in run_nums:

            args_list.append((time_horizon, num_policies, num_policy_runs, epsilon,
                              run, save_folder, seeds[count], finite_horizon_iteration,iteration,
                              GP_hyperparams, noise_param, num_states,
                              num_actions))

            count += 1

# Run processes sequentially:
for args in args_list:
    DPS_Preference(args)


''' 2 Plot

preference_noise = 0.01

num_steps = iteration * num_policy_runs * time_horizon * num_policies
fig_num = 1

filename_part1 = save_folder + 'Iter_' + str(iteration) + '_RBF_' + \
                str(kernel_variance) + '_' + str(kernel_lengthscales) + '_' + \
                str(GP_noise_var) + '_noise_' + str(preference_noise) + '_eps_' + str(epsilon) + '_run_'
filename_part2 = '.mat'

#plot_cumulative_reward(filename_part1, filename_part2, num_steps, run_nums, fig_num, True, False, 'blue')
'''
