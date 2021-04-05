import numpy as np
from os import listdir
import matplotlib.pyplot as plt

# Parameters
number_of_plots = 5
EXPERIMENT = []

# Load data
for i in range(number_of_plots):   
    raw = []
    with open(f'output/worker{i+1}_experience.txt','r') as f:
        for line in f:
            raw.append(line.split())
    experiment = np.array(raw).astype(np.float).reshape(1,-1)
    EXPERIMENT.append(experiment)

# Creat graph
for i in range(number_of_plots):
    plt.plot(EXPERIMENT[i][0][int(len(EXPERIMENT[i][0])/2):], label =f'Worker{i+1}')
# plt.plot(np.ones(1000) * 50, label ='BaseLine(50)')
plt.legend() 
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title("Comparision among worker' outcome")
plt.savefig("output/worker_performance.png")