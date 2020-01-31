# -*- coding: utf-8 -*-
"""
Simple MountainCar Preference Environment (Wirth et all).
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math

class SimpleMountainCarEnv(gym.Env):
    """
    This is the Simple MountainCar environment described by Wirth.
    """

    def __init__(self):

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.horizon = 500
        self.num_step = 0
        self.goaled = False
        self.done = False

        self.force=0.001
        self.gravity=0.0023

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.nA = self.action_space.n

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    '''
    For one step, reward is 1 if reaching the goal state, otherwise 0
    Action: 0 left 1 stay 2 right
    '''
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state

        velocity += (action-1+np.random.uniform(-0.2,0.2))*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        self.num_step += 1

        self.done = bool(self.num_step == self.horizon)

        if position >= self.goal_position:
            reward = 1
            self.goaled = True
        else:
            reward = 0
            self.goaled = False

        if not self.goaled:
            self.state = (position, velocity)
        # otherwise state does not change
        return np.array(self.state), reward

    '''
    Starting space is randomly sampled from the complete space
    '''
    def reset(self):
        self.num_step = 0
        self.goaled = False
        self.done = False
        self.state = np.array([self.np_random.uniform(low=self.min_position, high=self.max_position),\
                               self.np_random.uniform(low=-self.max_speed, high=self.max_speed)])
        return np.array(self.state)

    """
    Return the total reward accrued in a particular trajectory.
    trajectory = [state_sequence, action_sequence]
    state = np.array(position, velocity)
    states : 2D-array time_horizaon x dimension of space, state sequence
    """
    def get_trajectory_return(self, tr):

        states = tr[0]
        actions = tr[1]
        positions = states[:][0]

        # Sanity check:
        if not len(states) == len(actions) + 1:
            print('Invalid input given to get_trajectory_return.')
            print('State sequence expected to be one element longer than corresponding action sequence.')

        rewards = [1 for p in positions if p >= self.goal_position]

        return sum(rewards)




class SimpleMountainCarPreferenceEnv(SimpleMountainCarEnv):
    """
    Extends SimpleMountainCarEnv (defined above) to include preference feedback.
    """

    def __init__(self):

        super().__init__()

    def step(self, action):

        _, reward = super().step(action)

        self.episode_return += reward

        return self.state

    def reset(self):
        """
        We will keep track of the total reward for the current episode and the
        previous one; these are both initialized here.
        """

        _ = super().reset()

        self.episode_return = 0

        return self.state

    """
    Determine user's preference given the total returns of the last 2 trajectories
    """
    def get_preference(self, r1, r2, noise = False, user_noise = 1):

        if not noise:  # Deterministic preference:

            if r1 == r2:  # Compare returns to determine the preference
                preference = 0.5
            elif r1 > r2:
                preference = 0
            else:
                preference = 1

        else:   # Logistic noise model

            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-user_noise * (r2 - r1)))
            # larger noise_paran, less noisy

            preference = np.random.choice([0, 1], p = [1 - prob, prob])

        return preference


    def get_episode_return(self):

        if not self.done:
            print('Still running! Cannot check the total return')
            return

        return self.episode_return


'''
test
'''
# env = SimpleMountainCarPreferenceEnv()
# print(env.reset())
#
# time_horizon = 10
# state_dimension = 2
#
# action_sequence = np.empty(time_horizon)
# state_sequence = np.empty((time_horizon + 1,state_dimension))
# tras = []
#
# for _ in range(2):
#     for t in range(10):
#         state = env.state
#         action = np.random.choice([0,1,2])
#         next_state = env.step(action)
#
#         state_sequence[t] = state
#         action_sequence[t] = action
#
#     tras.append([state_sequence,action_sequence])
#
# pre = env.get_trajectory_preference(tras[0],tras[1],False,0)
# print(pre)

