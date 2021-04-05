import gym
from gym import spaces
import numpy as np
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display

class UAVEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    ##### FIELD #####

    ## Parameter Configuration
    squre_length = 10 # 10 unit, each is 100 meters

    ## Ground User Settings
    N_U = 100 # 100 users
    # User Location distribution: 4 hots spots, each having N-U/5 users, the others are uniformly distributed
    hotspots = np.matrix([[squre_length*0.2, squre_length*0.2], [squre_length*0.8, squre_length*0.8], [squre_length*0.3, squre_length*0.7], [squre_length*0.7, squre_length*0.3]])
    # Create a matrix to store user location distributionzaAAAAAAAAAAAAAAAAAAAAAAAAAAA
    u_loc = np.zeros((N_U, 2))

    ## UAV Settings
    N_UAV = 5 # 5 uavs available
    # power for Hovering and Level Flight of UAV
    power_hover = 1/100;  # 1 unit = 9.428W
    v_uav = 100/9; # unit is m/s
    power_level = 0.723/100; # 0.723 units, power_level tends to be smaller than power_hover due to velocity
    # UAV altitude
    uav_h = 3;  # unit
    fc = 2e9;  # unit is Hz
    uav_range = uav_h * math.tan(30/360*2*math.pi)
    pt = 10**((-49.5-30)/10)  # -49.5dBm
    po = 10**((-174-30)/10)  # -174dBm
    thru = 1e6  #300kbps
    d_max = 1 # maximum move by 2 units per step for each UAV
    w_rb = 180e3 # total bandwidth per Resource Block (RB): 180kHz per RB
    w = 5e6 # total bandwidth per UAV
    n_rb = w*0.9/w_rb  # Remove the guard bands, actual total bandwidth of one UAV

    # Time slot duration
    timeslot = 10;  # time between 2 timestep2

    ## Math notation
    beta = 2
    theta = 2*math.pi/N_UAV

    # Load custom environment
    raw = []
    with open('settings/UserLocations.txt','r') as f:
        for line in f:
            raw.append(line.split())
    myULoc = np.array(raw).astype(np.float)
    # myULoc = pd.read_excel('settings/U_loc.xlsx', header=None,engine='openpyxl').to_numpy()

    def __init__(self):
        super(UAVEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=0, 
                                        high=1, shape=(1, 10), dtype=np.float32)
        # Obs space
        self.observation_space = spaces.Box(low=0, high=15, shape=(1, 15), dtype=np.float32)
        # # Orignal init
        # for i in range(4):
        #     temp = 3.5*np.random.rand(int(self.N_U/5),2)-1.75
        #     self.u_loc[int(self.N_U/5)*i:int(self.N_U/5)*(i+1),:] = np.column_stack([(temp[:,0]+self.hotspots[i,0]), (temp[:,1]+self.hotspots[i,1])])

        # self.u_loc[int(self.N_U/5)*4:,:] = 10*np.random.rand(int(self.N_U/5),2)
        self.u_loc = self.myULoc
        self.state = np.zeros((self.N_UAV,3))

    def step(self, action):
        # reshape action
        # Execute one time step within the environment
        # counter of out-boundaries 
        flag = 0
        isDone = False
        quit_ind = (self.state[:, 2] <= 0)

        # UAV movement and energy change
        for i in range(self.N_UAV):
            tmp_x = self.state[i][0]
            tmp_y = self.state[i][1]
            # take action
            self.state[i][0] = self.state[i][0] + action[0][i + self.N_UAV] * self.d_max * math.cos(action[0][i]*2*math.pi)
            self.state[i][1] = self.state[i][1] + action[0][i + self.N_UAV] * self.d_max * math.sin(action[0][i]*2*math.pi)

            # modify energy
            if self.state[i][2] > 0:  # if UAV i not quits
                t_level = action[0][i + self.N_UAV] * self.d_max * 100 / self.v_uav
                eng = t_level * self.power_level + (10 - t_level) * self.power_hover
                self.state[i][2] -= eng

            # For out-of-boundary cases, stay still (cancel movement), flagged as 1 to give punishment
            if self.state[i][0] < 0 or self.state[i][0] > self.squre_length or self.state[i][1] < 0 or self.state[i][1] > self.squre_length:
                self.state[i][0] = tmp_x
                self.state[i][1] = tmp_y

                if quit_ind[i] == False:
                    self.state[i][2] += eng - 10 * self.power_hover
                    flag += 1
        
        # End of loop
        cvg = np.zeros((self.N_U, self.N_UAV))
        serv = np.zeros((self.N_U, self.N_UAV))

        for i in range(self.N_U):
            for j in range(self.N_UAV):
                tmp = math.sqrt( (self.u_loc[i,0] - self.state[j][0])**2 + (self.u_loc[i,1] - self.state[j][1])**2 )  # in units
                if (tmp < self.uav_range) and (quit_ind[j] == False): # user is within range and the UAV does not quit
                    cvg[i,j] = math.sqrt(tmp**2 + self.uav_h**2) * 100 # distance in meters between UAV j and user i
        
        # User association and bandwidth assignment
        UAV_CHs = np.zeros(self.N_UAV)
        for i in range(self.N_U):
            # Find the UAV with best SINR
            tmp_v = np.where(cvg[i,:] != 0) # index of UAVs covering user i, which has non-zero value
            if len(tmp_v[0]) > 0: # If there is at least 1 uav covering user i
                sinr_list = np.zeros(len(tmp_v[0])) # create a SINR list storing the SINR from all the UAVs covering user i
                idx = 0
                # Calcualte SINR for candidates UAVs
                for j in tmp_v[0]:   # sweep over the candidate UAVs to calculate the respective SINR
                    pathloss = 20*math.log10(4*math.pi*self.fc*cvg[i,j]/(3e8)) + 1; # unit is in dB
                    pathloss = 10**(pathloss/20);  # check how to convert between dB and a decimal value
                    rx_p = self.pt/pathloss
                    intf = 0;    # interference
                    for k in range(self.N_UAV):
                        if k == j:
                            continue
                        # Calculate throughput
                        if cvg[i,k] > 0:
                            pathloss = 20*math.log10(4*math.pi*self.fc*cvg[i,k]/(3e8)) + 1
                            pathloss  = 10**(pathloss/20)
                            intf += self.pt/pathloss
                    sinr_list[idx] = rx_p/(self.po + intf)
                    idx += 1
                # Sort SINR of candidate UAVs
                indexes = np.argsort(sinr_list)[::-1] 
                # Check from the UAV with best SINR, if has enough bandwidth
                # available, attach user to this UAV, if not, try UAV with
                # second best SINR, repeat
                for index in indexes:
                    idx = tmp_v[0][index]
                    n_tmp = math.ceil(self.thru / (self.w_rb*math.log2(1+sinr_list[index])))  # Calculate the minimum RBs needed by the user to satisfy the minimum thruput requirement given the SINR
                    if UAV_CHs[idx] + n_tmp <= self.n_rb:   # Check whether this UAV has enough available bandwidth
                        UAV_CHs[idx] = UAV_CHs[idx] + n_tmp  # if yes, update the occupied number of RBs
                        serv[i,idx] = 1   # attach this user to this UAV
                        break
        
        # Calculate the Reward: total number of users being served
        reward = (sum(sum(serv))/self.N_U)**self.beta

        if flag != 0:
            isDone = True

        return np.copy(self.state).reshape(1, self.N_UAV * 3), reward, isDone, f"{flag} uavs out of system"

    def reset(self):
        # Reset the state of the environment to an initial state
        # for i in range(self.N_UAV):
        #     self.state[i,0] = 5 + 1*math.cos(i*self.theta); # x
        #     self.state[i,1] = 5 + 1*math.sin(i*self.theta) # y
        # # Reset energy level
        # self.state[:, 2] = [1500,100,1200,1100,1100] # initial energy levels for 5 UAVs
        # # Normalize energy by dividing by 10
        # self.state[:, 2] /= 100

        # New state
        stateList = [6.9614, 8.0340, 2.5376, 2.8778, 6.8462,    5.0003, 8.1943, 6.6484, 1.8510, 1.8724]
        trueList = []
        for i in range(5):
            trueList.append(stateList[i])
            trueList.append(stateList[i+5])
        self.state[:,0:2] = np.array(trueList).astype(np.float).reshape(5,2)
        # Reset energy level
        self.state[:, 2] = [1500,100,1200,1100,1100] # initial energy levels for 5 UAVs
        # Normalize energy by dividing by 10
        self.state[:, 2] /= 100

        # Return
        return np.copy(self.state).reshape(1, self.N_UAV * 3)
        
    def render(self, mode='human', close=False):
        if mode == 'human':
            colors = ['red', 'blue', 'green', 'purple', 'yellow']
            position = self.state[:,0:2]
            # Render the environment to the screen
            plt.xlim([0, 10])
            plt.ylim([0, 10])
            for i in range(self.N_UAV):
                plt.scatter(position[i][0], position[i][1],color=colors[i])
                plt.text(position[i][0], position[i][1], f"uav-{i+1}")
            plt.xlabel("X cordinate")
            plt.ylabel("Y cordinate")
            plt.pause(0.001)
            plt.show()
            if is_ipython: display.clear_output(wait=True)
            # Done graph
        else:
            print(f"{sum(self.state[:, 2] > 0)}/{self.N_UAV} UAVs in the system")

        