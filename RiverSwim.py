# -*- coding: utf-8 -*-
"""
RiverSwim simulation environment, including either numerical or preference feedback.

"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class RiverSwimEnv(gym.Env):
    """
    There are 2 available actions: left (0) and right (1). See "(More) Efficient 
    Reinforcement Learning via Posterior Sampling" by Ian Osband, Benjamin
    Van Roy, and Daniel Russo, for details of the MDP.
    """
    
    def __init__(self, num_states = 6):
        
        # Initialize state and action spaces.
        self.nA = 2     # Actions: left and right
        self.action_space = spaces.Discrete(self.nA) # Set with 2 elements {0, 1}
        self.observation_space = spaces.Discrete(num_states) # Set with 6 elements {0, 1, 2, 3, 4, 5}
        self.nS = num_states
        
        self._seed()
        
        # Create transition probability matrix.
        # self.P[s][a] is a list of transition tuples (prob, next_state, reward)
        self.P = {}
        
        for s in range(self.nS):
            
            self.P[s] = {a : [] for a in range(self.nA)}
            
            for a in range(self.nA):
                
                if a == 0:  # Left action
                    
                    next_state = np.max([s - 1, 0])
                    
                    reward = 5/1000 if (s == 0 and next_state == 0) else 0
                    
                    self.P[s][a] = [(1, next_state, reward)]
                    
                elif s == 0:  # Leftmost state, and right action
                    
                    self.P[s][a] = [(0.4, s, 0), (0.6, s + 1, 0)]
                    
                elif s == self.nS - 1:   # Rightmost state, and right action
                    
                    self.P[s][a] = [(0.4, s - 1, 0), (0.6, s, 1)]
                    
                else:   # Intermediate state, and right action
                    
                    self.P[s][a] = [(0.05, s - 1, 0), (0.6, s, 0), 
                          (0.35, s + 1, 0)]

        # Start the first game
        self._reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        return self._reset()
    
    def step(self, action):
        return self._step(action)
    
    def _reset(self): # We always start at the leftmost state.
        
        self.state = 0
        return (self.state, )
    
    """
    Take a step using the transition probability matrix specified in the 
    constructor.
    """
    def _step(self, action):
        
        transition_probs = self.P[self.state][action]
        
        num_next_states = len(transition_probs)

        next_state_probs = [transition_probs[i][0] for i in range(num_next_states)]
            
        outcome = np.random.choice(np.arange(num_next_states), p = next_state_probs)
        
        self.state = transition_probs[outcome][1]    # Update state
        reward = transition_probs[outcome][2]
        
        return (self.state, ), reward, False, {}   # done = False, since finite-horizon setting
    
    """
    Return the reward associated with a specific action from a state-action 
    pair, and the resulting next state.
    """
    def get_step_reward(self, state, action, next_state):
        
        transition_probs = self.P[state][action]
        
        reward = 0
        
        for i in range(len(transition_probs)):
            
            if transition_probs[i][1] == next_state:
                reward = transition_probs[i][2]
                break
            
        return reward
    
    
    """
    Return the total reward accrued in a particular trajectory.
    """    
    def get_trajectory_return(self, tr):
        
        states = tr[0]
        actions = tr[1]
        
        # Sanity check:        
        if not len(states) == len(actions) + 1:
            print('Invalid input given to get_trajectory_return.')
            print('State sequence expected to be one element longer than corresponding action sequence.')      
        
        total_return = 0
        
        for i in range(len(actions)):
            
            total_return += self.get_step_reward(states[i], actions[i], \
                                         states[i + 1])
            
        return total_return

         
      
class RiverSwimPreferenceEnv(RiverSwimEnv):
    """
    This class extends the RiverSwim environment to handle trajectory-
    preference feedback.
    
    The following changes are made to the RiverSwimEnv class defined above:
        1) Step function no longer returns reward feedback.
        2) Add a function that calculates a preference between 2 inputted
            trajectories.
    """

    def __init__(self, num_states = 6):
    
        super().__init__(num_states)        # RiverSwimPreferenceEnv, self

    """
    Take a step using the transition probability matrix specified in the 
    constructor. This is identical to the RiverSwim class, except that now we 
    no longer return the reward.
    """
    def _step(self, action):

        state, _, done, info = super()._step(action)    # RiverSwimPreferenceEnv, self
        return state, done, info
       
    """
    Return a preference between two given state-action trajectories, tr1 and tr2.
    
    Format of inputted trajectories: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
    
    Preference information: 0 = trajectory 1 preferred; 1 = trajectory 2 
    preferred; 0.5 = trajectories preferred equally.
    
    Preferences are determined by comparing the rewards accrued in the 2 
    trajectories.
    
    CONSIDER: later adding a noise parameter to create noisy preferences. E.g.,
    with probability epsilon, the preference is either random or flipped from its
    correct value.
    """    
    def get_trajectory_preference(self, tr1, tr2, noise = False, noise_param = 1):
    
        trajectories = [tr1, tr2]
        num_traj = len(trajectories)
        
        returns = np.empty(num_traj)  # Will store cumulative returns for the 2 trajectories
        
        for i in range(num_traj):
            
            returns[i] = self.get_trajectory_return(trajectories[i])
            
        if not noise:  # Deterministic preference:
            
            if returns[0] == returns[1]:  # Compare returns to determine the preference
                preference = 0.5
            elif returns[0] > returns[1]:
                preference = 0
            else:
                preference = 1
                
        else:   # Logistic noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-noise_param * (returns[1] - returns[0])))  #np.exp(noise_param * (returns[1] - returns[0])) / (1 + np.exp(noise_param * (returns[1] - returns[0])))
            # larger noise_paran, less noisy
            
            preference = np.random.choice([0, 1], p = [1 - prob, prob])
                
        
        return preference

    """
    Return a preference between two actions in a given state. Returned value is 
    0 if action1 is preferred, 1 if action2 is preferred, and 0.5 if the 2 
    actions are equally preferred.
    
    The preference is based upon knowledge of the optimal policy; in this case,
    going right is always preferable to going left.
    
    CONSIDER: later adding a noise parameter to create noisy preferences. E.g.,
    with probability epsilon, the preference is either random or flipped from its
    correct value.
    """    
    def get_state_action_preference(self, state, action1, action2):
    
        if action1 == action2:
            return 0.5
        elif action1 > action2:
            return 0
        else:
            return 1

