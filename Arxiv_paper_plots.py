# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:21:05 2019

@author: Ellen
"""
'''
1.读取mat文件们，同一参数(alg_pref_noise)的不同runs,
2.计算mean，variance
3.plot
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from matplotlib import rcParams

plt.close('all')

rcParams.update({'font.size': 14})


# Function to calculate means and standard deviations of the cumulative step
# rewards, and add them to the given figure. Also, includes an option for just
# plotting each cumulative sum sequence separately.

def plot_cumulative_reward(filename_part1, filename_part2, num_steps, num_runs, 
                           fig_num, plot_mean_SD = True, line_plot = False,
                           color = 'blue', scale_self = False, scale_vals = [],
                           make_plot = True, alpha = 0.5):

    plt.figure(fig_num, figsize = (7.5, 5.5))
    
    # Obtain the cumulative step rewards over the runs.
    step_vals = np.empty((num_steps, num_runs))
    step_vals_scaled = np.empty((num_steps, num_runs))
    
    for run in range(num_runs):
        
        # Load and unpack results:
        results = io.loadmat(filename_part1 + str(run) + filename_part2)
        rewards = results['step_rewards'].flatten()[: num_steps]
        
        # Check if scaling:
        if scale_self:
            rewards_scaled = rewards / np.sum(rewards)
        elif len(scale_vals) > 0:
            rewards_scaled = rewards / scale_vals[run]
        else:
            rewards_scaled = rewards
        
        # Obtain cumulative sum of rewards. Add to plot if in line_plot mode.
        step_vals[:, run] = np.cumsum(rewards)
        step_vals_scaled[:, run] = np.cumsum(rewards_scaled)
        
        if line_plot and make_plot:
            plt.plot(np.arange(1, num_steps + 1), step_vals_scaled[:, run],
                     color = color)
    
    if plot_mean_SD and make_plot:
        mean = np.mean(step_vals_scaled, axis = 1)
        stdev = np.std(step_vals_scaled, axis = 1)
        
        # Plot the step rewards:    
        plt.plot(np.arange(1, num_steps + 1), mean, color = color)
        plt.fill_between(np.arange(1, num_steps + 1), mean - stdev, 
                         mean + stdev, alpha = 0.5, color = color)
        
    # Return average reward for this set of runs:
    return np.mean(step_vals[-1, :]), step_vals[-1, :]
        

fig_num = 0

# Constants:
time_horizon = 50
num_policies = 1  # Num. policies to request from advance function at a time
# Number of times to run each policy and request a comparison:
num_policy_runs = 1

noise_params = [0.1, 0.5, 1, 1000]    # Noise parameters

# PSRL prior parameters:
diri_prior = 1
NG_prior = [1, 1, 1, 1]

# Load matlab file with optimal EPMC parameters:
EPMC_params = io.loadmat('EPMC_best_hyperparameters.mat')
best_alphas = EPMC_params['best_alphas'][0]
best_etas = EPMC_params['best_etas'][0]

filename_part2 = '.mat'
num_runs = 100

for i, noise_param in enumerate(noise_params):
    
    fig_num += 1
    legend_labels = []
        
    # We will normalize by the total reward achieved by the optimal policy.
    # Folder for saving results:
    save_folder = 'PB_PSRL_Self_Sparring_RiverSwim_optimal/'
    
    max_iter = 200
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_optimal_policy_run_'
    
    num_steps_plot = 10000
    _, scale_vals = plot_cumulative_reward(filename_part1, filename_part2, 
           num_steps_plot, num_runs, fig_num, True, False, 'white', scale_self = True,
           make_plot = False)
    
    # Scale all runs by the same quantity: take the mean of cumulative rewards
    # for the optimal policy.
    scale_vals = np.mean(scale_vals) * np.ones(num_runs)
    
    # Plot PSRL results.
    save_folder = 'PSRL_numerical_rewards/'
    max_iter = 400
    num_steps = max_iter * num_policy_runs * time_horizon * num_policies
          
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_params_' \
                            + str(diri_prior) + '_' + str(NG_prior[0]) + \
                            '_' + str(NG_prior[1]) + '_' + str(NG_prior[2]) + \
                            '_' + str(NG_prior[3]) + '_run_'

    plot_cumulative_reward(filename_part1, filename_part2, num_steps, 
           num_runs, fig_num, True, False, 'blue', scale_vals = scale_vals)
        
    legend_labels += ['PSRL']    
    
    # Add GPR results to plots.
    save_folder = 'Noisy_preferences/GPR/'
    
    max_iter = 200
    kernel_variance = 0.03
    kernel_lengthscale = 0
    noise_variance = 0.05
    
    num_steps = max_iter * num_policy_runs * time_horizon * num_policies
        
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_RBF_' + \
    str(kernel_variance) + '_' + str(kernel_lengthscale) + '_' + \
    str(noise_variance) + '_noise_' + str(noise_param) + '_run_'

    plot_cumulative_reward(filename_part1, filename_part2, num_steps, 
           num_runs, fig_num, True, False, 'green', scale_vals = scale_vals)
        
    legend_labels += ['GPR']    
    
    # Add Bayesian logistic regression results to plots.      
    save_folder = 'Noisy_preferences/Bayes_log_reg/'
    
    max_iter = 200
    param = 20
    
    num_steps = max_iter * num_policy_runs * time_horizon * num_policies
        
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_param_' + \
    str(param) + '_noise_' + str(noise_param) + '_run_'

    plot_cumulative_reward(filename_part1, filename_part2, num_steps, 
           num_runs, fig_num, True, False, 'red', scale_vals = scale_vals)
        
    legend_labels += ['Bayes. log. reg.']  
    
    # Add Bayesian linear regression results to plots.
    save_folder = 'Noisy_preferences/Bayes_lin_reg/'
    
    max_iter = 200
    sigma = 0.5
    lambd = 0.1
    
    num_steps = max_iter * num_policy_runs * time_horizon * num_policies
    
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_params_' + \
    str(sigma) + '_' + str(lambd) + '_noise_' + str(noise_param) + '_run_'

    plot_cumulative_reward(filename_part1, filename_part2, num_steps, 
           num_runs, fig_num, True, False, 'black', alpha = 0.7, scale_vals = scale_vals)
        
    legend_labels += ['Bayes. lin. reg.']  
    
    # Add EPMC results to plots.   
    max_iter = 400
    num_steps = max_iter * num_policy_runs * time_horizon * num_policies
    
    save_folder = 'EPMC_probabilistic/'
 
    alpha = best_alphas[i]
    eta = best_etas[i]
    
    if i == 1:
        eta = '0.7000000000000001'
    
    filename_part1 = save_folder + 'Iter_' + str(max_iter) + '_alpha_' + \
            str(alpha) + '_eta_' + str(eta) + '_noise_' + str(noise_param) + \
            '_run_'

    plot_cumulative_reward(filename_part1, filename_part2, num_steps, 
           num_runs, fig_num, True, False, 'orange', scale_vals = scale_vals)
    
    legend_labels += ['EPMC']    

    # Tidy plots:
    plt.xlim([0, num_steps_plot])
    plt.ylim([-0.15, 1])
    
    plt.xlabel('Number of steps taken in environment')
    plt.ylabel('Cumulative reward (normalized)')
    
    plt.legend(legend_labels, loc = 'best')
    
    # Save figure:
    # plt.savefig('Plots/NeurIPS_Revision/RiverSwim_noise_' + str(noise_param) + '.png') 

