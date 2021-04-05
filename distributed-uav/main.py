import numpy as np
import ray
from buffer import BufferMemory
from learner import Learner
from parameter_server import ParameterServer
from worker import Worker
import gym 
import researchGym

ray.init()

# Declare parameter
training_episode = 2000
batch_size = 512
average_reward_per = 50
number_of_worker = 5

### SPAWN
# Create gobal service
remote_buffer = BufferMemory.remote(max_size = 1000000)
param_server = ParameterServer.remote()
learner = Learner.remote(remote_buffer, param_server, training_episode, batch_size=batch_size)

# Create workers
workers = []
for i in range(number_of_worker):
    w = Worker.remote(i+1, param_server, remote_buffer, training_episode, batch_size=batch_size)
    workers.append(w)

### Train
actors = workers + [learner]
ray.wait([actor.run.remote() for actor in actors])