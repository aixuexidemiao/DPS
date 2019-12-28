"""
Random MDP simulation environment, including either numerical or preference feedback.
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class RandomMDPEnv(gym.Env):
    """
    Sample a Random MDP, as described in See "(More) Efficient 
    Reinforcement Learning via Posterior Sampling" by Ian Osband, Benjamin
    Van Roy, and Daniel Russo.
    """
    
    def __init__(self, num_states = 10, num_actions = 5):
        
        # Initialize state and action spaces.
        self.nA = num_actions
        self.action_space = spaces.Discrete(self.nA)
        self.nS = num_states
        self.observation_space = spaces.Discrete(self.nS)
        
        self._seed()
        
        # Prior for Dirichlet model (for sampling the dynamics):
        dirichlet_prior = np.ones(num_states)
        
        # Prior parameters for normal-gamma model (for sampling the rewards):
        NG_prior = [1, 1, 1, 1]
        
        # Create transition probability matrix.
        # self.P[s][a] is a list of transition tuples (prob, next_state, reward)
        self.P = {}
        
        for s in range(self.nS):
            
            self.P[s] = {a : [] for a in range(self.nA)}
            
            for a in range(self.nA):
                
                # Sample transition dynamics:
                transition_probs = np.random.dirichlet(dirichlet_prior)
                
                # Sample rewards:
                rewards = np.empty(num_states)
                
                for s_ in range(num_states):
                        
                    gamma_sample = np.random.gamma(NG_prior[2], 1 / NG_prior[3])
                    rewards[s_] = np.random.normal(NG_prior[0], 
                             (NG_prior[1] * gamma_sample)**(-0.5))                
                
                # Normalize rewards to be between 0 and 1:
                rewards = (rewards - np.min(rewards)) /   \
                            (np.max(rewards) - np.min(rewards))
                
                for s_next in range(self.nS):
                    
                    self.P[s][a].append((transition_probs[s_next], 
                            s_next, rewards[s_next]))

        # Sample initial state distribution:
        self.P0 = np.random.dirichlet(dirichlet_prior)

        # Start the first game
        self._reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        return self._reset()
    
    def step(self, action):
        return self._step(action)
    
    def _reset(self):   # Draw a sample from the initial state distribution.
        
        outcome = np.random.choice(np.arange(self.nS), p = self.P0)
        
        self.state = outcome
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


class RandomMDPPreferenceEnv(RandomMDPEnv):
    """
    This class extends the RandomMDP environment to handle trajectory-
    preference feedback.
    
    The following changes are made to the RandomMDPEnv class defined above:
        1) Step function no longer returns reward feedback.
        2) Add a function that calculates a preference between 2 inputted
            trajectories.
    """

    def __init__(self, num_states = 10, num_actions = 5):
    
        super().__init__(num_states)        # RandomMDPPreferenceEnv, self

    """
    Take a step using the transition probability matrix specified in the 
    constructor. This is identical to the RandomMDP class, except that now we 
    no longer return the reward.
    """
    def _step(self, action):

        state, _, done, info = super()._step(action)    # RandomMDPPreferenceEnv, self
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
            prob = 1 / (1 + np.exp(-noise_param * (returns[1] - returns[0])))
            
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

        
        