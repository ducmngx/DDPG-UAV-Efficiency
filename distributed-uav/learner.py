import os 
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import ray
from models import *
from utils import *

@ray.remote(num_gpus=1)
class Learner:

    def __init__(self, buffer_remote, param_server, training_episode, batch_size, actor_learning_rate=1e-6, critic_learning_rate=1e-6, 
    l2_regularization_rate =1e-4, gradient_threshold=1, gamma=0.9, target_smooth=1e-3, 
    max_memory_size=1000000):
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Params
        self.training_episode = training_episode
        self.gamma = gamma
        self.tau = target_smooth
        self.gradient_threshold = gradient_threshold
        self.batch_size = batch_size
        
        # Networks
        self.actor = Actor().double()
        self.actor.to(self.device)
        self.actor_target = Actor().double()
        self.actor_target.to(self.device)
        self.critic = Critic().double()
        self.critic.to(self.device)
        self.critic_target = Critic().double()
        self.critic_target.to(self.device)
        self.param_dict = {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
        self.param_server = param_server

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # ray remote objects
        self.buffer_remote = buffer_remote
        self.update_step = 0
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate, weight_decay=l2_regularization_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate, weight_decay=l2_regularization_rate)

    # A small save function
    def save(self):
        torch.save(self.actor.state_dict(), "weights/best_actor.pth")
        torch.save(self.critic.state_dict(), "weights/best_critic.pth")

    # Update gradient weight of actor-critic network
    def optimize_parameters(self, batch_size):
        states, actions, rewards, next_states, _ = ray.get(self.buffer_remote.sample.remote(batch_size))

        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states).reshape(-1,10)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states).reshape(-1,10)).mean()
        
        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.gradient_threshold)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.gradient_threshold) 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    
    # to take advantage of Ray's zero-cost reading of numpy arrays,
    # we convert the learner parameters from pytorch tensors to 
    # numpy arrays, and then send those over to the remote paramter server object
    def update_param_server(self):
        params = {'actor': [], 'critic': []}
        state_dict = {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
        for param in list(self.param_dict['actor']):
            params['actor'].append(state_dict['actor'][param].cpu().numpy())
        for param in self.param_dict['critic']:
            params['critic'].append(state_dict['critic'][param].cpu().numpy())
        self.param_server.update_params.remote(params)
        # Update number of steps
        self.update_step += 1
    
    def return_param_dict(self):
        return self.param_dict

    def return_update_step(self):
        return self.update_step
    
    def run(self):
        print("Learner starts")
        while ray.get(self.buffer_remote.get_length.remote()) < self.batch_size:
            continue

        while ray.get(self.buffer_remote.get_episode.remote()) < self.training_episode:
            # update parameters
            self.optimize_parameters(self.batch_size)

            # sync with global
            self.update_param_server()
            self.update_step += 1

            # if self.update_step % 100 == 0:
            #     print(f"learner update step: {self.update_step}")
        # save param
        self.save()
        print("Learner exits")