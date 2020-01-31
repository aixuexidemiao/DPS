import gym
import numpy as np

time_horizon = 5

env = gym.envs.make("MountainCar-v0")
env.reset()

state_dimension = 2
state_sequence = np.empty((time_horizon + 1,state_dimension))

action = 2

for t in range(time_horizon):
    state = env.state
    next_state, _, _, _ = env.step(action)

    state_sequence[t] = state

position = state_sequence[:,0]
print(state_sequence)
