'''
Worker function is to run the simulation and collect buffer memory 
- Push experience to global buffer
- Download weights from global param server
'''
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ray
import torch
import gym
import researchGym
import random
from models import *
from utils import *

@ray.remote
class Worker:

    def __init__(self, workerID, param_server, remote_buffer, training_episode, batch_size, max_timestep=100):
        super().__init__()
        import researchGym
        self.id = workerID
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_episode = training_episode
        np.random.seed(self.id )
        random.seed(self.id)   
        self.noise = OUNoise()
        self.env = gym.make('uav-v0')
        self.max_timestep = max_timestep
        self.batch_size = batch_size

        # Actor
        self.actor = Actor().double()

        # remote service
        self.param_server = param_server
        self.remote_buffer = remote_buffer

    def update(self):
        new_params = ray.get(self.param_server.return_params.remote())['actor']
        for param, new_param in zip(self.actor.parameters(), new_params):
            new_param = torch.Tensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def run(self, average_reward_per=10):
        episode = 0
        rewards = []
        avg_rewards = []
        while ray.get(self.remote_buffer.get_episode.remote()) < self.training_episode:
            episode += 1
            np.random()
            state = self.env.reset()
            self.noise.reset()
            episode_reward = 0
            internal_buffer = []
            for step in range(self.max_timestep):
                state_cpu = torch.from_numpy(state).to(self.device)
                action = self.actor.forward(state_cpu.reshape(-1))
                action = action.cpu().detach().numpy()
                action = self.noise.get_action(action, step)
                # Make sure action is lower or equal to 1
                action[action > 1] = 1
                new_state, reward, done, _ = self.env.step(action) 
                exp = (state.reshape(-1), action.reshape(-1), reward, new_state.reshape(-1), done)
                internal_buffer.append(exp)

                state = new_state
                episode_reward += reward

                if done or step == self.max_timestep-1:
                    break

            ### AFTER FINISH AN EPISODE
            # Add to global buffer memory
            for st, a, r, nst, d in internal_buffer:
                self.remote_buffer.push.remote(st, a, r, nst, d)
            # Update param
            if ray.get(self.remote_buffer.get_length.remote()) < self.batch_size: 
                self.update()
                # Update number of total episide
            self.remote_buffer.add_episode.remote()
            sys.stdout.write("Worker: {}, episode: {}, reward: {}, average _reward: {} , steps: {}\n".format(self.id, episode, 
                                np.round(episode_reward, decimals=2), np.mean(rewards[-10:]), step))
            
            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-average_reward_per:]))
        # After training
        with open(f'output/worker{self.id}_experience.txt', 'w') as filehandle:
            for rew in rewards:
                filehandle.write('%s\n' % rew)
            for arew in avg_rewards:
                filehandle.write('%s\n' % arew)

        print(f'Worker {self.id} stops')
