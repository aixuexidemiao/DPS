#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run batch of DPS tests with random MDP environment.
"""

import numpy as np
import os
from multiprocessing import Pool

from DPS_Bayesian_logistic_regression import DPS_Bayes_log_reg
from DPS_conjugate_Bayes_linreg import DPS_Bayes_lin_reg
from DPS_GPR import DPS_GP_reg

# Define constants for random MDP environment:
num_states = 50
num_actions = 5

# Finite or infinite-horizon value iteration?
finite_horizon_iteration = True

time_horizon = 20
num_policies = 2  # Num. policies to request from advance function at a time
# Number of times to run each policy and request a comparison:
num_policy_runs = 1

epsilon = 0       # Do not use epsilon-greedy control

run_nums = np.arange(100)    # Number of runs for each set of parameter values

noise_params = [0.1, 1, 1000]     # Preference noise parameters

"""Run GPR cases."""

# Folder for saving results:
save_folder = 'GPR/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    
# GPR hyperparameters to use:   
kernel_variance = 1
kernel_lengthscale = 0 #use this in RiverSwim
GP_noise_var = [0.03]

GP_hyperparams = {'var': kernel_variance, 'length': [kernel_lengthscale,
                                kernel_lengthscale], 'noise': GP_noise_var}

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
                              run, save_folder, seeds[count], finite_horizon_iteration,
                              GP_hyperparams, noise_param, num_states, 
                              num_actions))
            
            count += 1

# Run processes sequentially:
for args in args_list:
    DPS_GP_reg(args)
    

# Launch a pool of worker processes to calculate results for these cases:
#pool = Pool(processes = 8)
#pool.map(Self_Sparring_VI_GP_reg, args_list)




"""Run Bayesian linear regression cases."""

# Folder for saving results:
save_folder = 'Bayes_lin_reg/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)  

# Bayesian linear regression hyperparameters to use:
sigma = 0.1
lambd = 0.1

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
                          run, save_folder, seeds[count], finite_horizon_iteration,
                          [sigma, lambd], noise_param, num_states, num_actions))
        
        count += 1

# Run processes sequentially:
for args in args_list:
    DPS_Bayes_lin_reg(args)


"""Run Bayesian logistic regression cases."""

# Folder for saving results:
save_folder = 'Bayes_log_reg/'

if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

prior_covariance = 20

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
                          run, save_folder, seeds[count], finite_horizon_iteration,
                          prior_covariance, noise_param, num_states,
                          num_actions))

        count += 1


# Run processes sequentially:
#for args in args_list:
#    Self_Sparring_VI_Bayes_log_reg(args)

pool = Pool(processes = 6)
pool.map(DPS_Bayes_log_reg, args_list)
                
