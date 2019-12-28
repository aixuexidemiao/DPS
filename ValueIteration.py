# -*- coding: utf-8 -*-
"""
Infinite-horizon value iteration:
  Adapted from Denny Britz's course, see:
    https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    http://www.wildml.com/2016/10/learning-reinforcement-learning/
    
Finite-horizon value iteration: modified from the infinite-horizon case.
"""

import numpy as np

def value_iteration(P, R, num_states, num_actions, theta=0.0001, 
                    discount_factor=1.0, epsilon = 0, H = np.infty):

    """
    Value Iteration Algorithm--infinite and finite-horizon.
    
    Args:
        P[s][a] is an array of length num_states, with the probability of 
        transitioning to each state after taking action a in state s.
        R is a num_states x num_actions matrix (or num_states-length vector), 
        where R[s, a] (or R[s]) is the expected reward for taking action a in 
        state s (or for landing in state s).
        num_states is a number of states in the environment. 
        num_actions is a number of actions in the environment.
        theta: For infinite-horizon case only. We stop evaluation once our 
            value function change is less than theta for all states. Ignored if
            a finite H is specified.
        discount_factor: Gamma discount factor.
        epsilon: make policy epsilon-greedy with parameter epsilon.
        H: episode horizon. Set to np.infty for infinite horizon case (this is
        the default), or to a positive integer for the finite-horizon case.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
        
        policy is (num_states * num_actions) for infinite horizon case, and 
        (H * num_states * num_actions) for finite horizon case.
        
        V is a num_states-length vector for infinite horizon case, and 
        (H * num_states) for finite horizon case.
    """
    
    if H == np.infty:    # Infinite-horizon value iteration
        
        policy, V = value_iteration_inf_horizon(P, R, num_states, num_actions, 
                        theta, discount_factor, epsilon)
        
    else:    # Finite-horizon value iteration
        
        policy, V = value_iteration_finite_horizon(P, R, num_states, 
                    num_actions, H, discount_factor, epsilon)
        
    return policy, V


def value_iteration_inf_horizon(P, R, num_states, num_actions, theta=0.0001, 
                    discount_factor=1.0, epsilon = 0):
    """
    Value Iteration Algorithm--infinite horizon.
    
    Args:
        P[s][a] is an array of length num_states, with the probability of 
        transitioning to each state after taking action a in state s.
        R is a num_states x num_actions matrix, where R[s, a] is the expected
        reward for taking action a in state s.
        num_states is a number of states in the environment. 
        num_actions is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        epsilon: make policy epsilon-greedy with parameter epsilon.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    # Only state rewards, as opposed to state-action rewards?
    states_only = (len(R.shape) == 1 or R.shape[1] == 1)
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length num_states
        
        Returns:
            A vector of length num_actions containing the expected value of each action.
        """
        A = np.zeros(num_actions)
        for a in range(num_actions):
            for next_state, prob in enumerate(P[state][a]):
                if states_only:
                    reward = R[next_state]
                else:
                    reward = R[state, a]
                
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(num_states)
    
    #count = 0
    
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
        
        #count += 1
        #if count < 40:
         #   print(V)
    
    # Create an epsilon-greedy policy using the optimal value function:
    policy = (epsilon / num_actions) * np.ones([num_states, num_actions])
    
    for s in range(num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] += 1 - epsilon
    
    return policy, V

def value_iteration_finite_horizon(P, R, num_states, num_actions, H = 50, 
                    discount_factor=1.0, epsilon = 0):
    """
    Value Iteration Algorithm--finite horizon.
    
    Args:
        P[s][a] is an array of length num_states, with the probability of 
        transitioning to each state after taking action a in state s.
        R is a num_states x num_actions matrix, where R[s, a] is the expected
        reward for taking action a in state s.
        num_states is a number of states in the environment. 
        num_actions is a number of actions in the environment.
        H: number of steps in a learning episode.
        discount_factor: Gamma discount factor.
        epsilon: make policy epsilon-greedy with parameter epsilon.
        
    Returns:
        The optimal policy and corresponding value function.
    """

    R = R.flatten()
    
    Q = np.zeros((H + 1, num_states * num_actions))
    V = np.zeros((H + 1, num_states))
    
    # Create probability transition matrix:
    prob_matrix = np.empty((num_states * num_actions, num_states))
    
    count = 0
    
    for state in range(num_states):
        for action in range(num_actions):
            
            prob_matrix[count, :] = P[state][action]
            count += 1
    
    # Create an epsilon-greedy policy using the optimal action-value function:
    policy = (epsilon / num_actions) * np.ones([H, num_states, num_actions])

    for t in np.arange(H - 1, -1, -1):
        
        Q[t, :] = R + discount_factor * prob_matrix @ V[t + 1, :]
                    
        # Update value function for this time step:
        Q_matrix = Q[t, :].reshape((num_states, num_actions))
        V[t, :] = np.max(Q_matrix, axis = 1)
                
        # Best actions at this time step for each state:
        best_actions = np.argmax(Q_matrix, axis = 1)
        
        # Update policy for this time step:
        for state, best_action in enumerate(best_actions):
            policy[t, state, best_action] += 1 - epsilon

    return policy, V

