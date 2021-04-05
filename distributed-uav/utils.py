import numpy as np
import gym
from collections import deque
import random
import math

# Ornstein-Ulhenbeck Process
# Taken from #https://www.mathworks.com/help/reinforcement-learning/ref/rlddpgagentoptions.html#mw_2875b71d-bfb0-4be4-b0d3-a44592c3cb30_head
class OUNoise:
    def __init__(self, mean=0.0, mean_attraction_constant=0.6, variance=0.6, decay_rate=5e-4):
        self.mean                     = mean
        self.mean_attraction_constant = mean_attraction_constant
        self.variance                 = variance
        self.variance_min             = 0
        self.decay_rate               = decay_rate
        self.action_dim               = 10
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mean
        
    def evolve_state(self, t):
        x  = self.state
        dx = self.mean_attraction_constant * (self.mean - x) * t + self.variance * np.random.randn(self.action_dim) * math.sqrt(t)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state(t)
        decayed_variance = self.variance * (1 - self.decay_rate)
        self.variance = max(decayed_variance, self.variance_min)
        return np.clip(action + ou_state, self.variance_min, self.variance)
        
class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

# Greedy-epsilon 
class EpsilonGreedy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
