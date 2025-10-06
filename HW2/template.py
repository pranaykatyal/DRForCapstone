import numpy as np
import matplotlib.pyplot as plt
import random
from network_agent import NetworkAgent, DynamicAgent
import networkx as nx
from distributed_algorithm import run_ctl_sim_circle, dist_ccw, dist_cw, plot_agents
from math import pi, sin, cos

# Network -- S_circle first-order agents in S^1 with absolute sensing of own position, communication range r
# Algorithm -- Agree & Pursue
# Alphabet -- A=S^1x{c,cc}xIU{null}

# Processor State: w=(dir,max-id), where
# dir in {c,cc}; initially, dir[i] unspecified
# max-id in I; initially, max-id[i]=i for all i

class AgreePursueAgent(DynamicAgent):

    def __init__(self, id=0, n=5, theta=0, speed=0.05, dt=0.01):
        self.id = id
        self.theta = theta
        self.n = n
        self.msgs = []
        self.dir = random.choice([-1, 1])
        self.max_id = id
        self.speed = speed
        self.dt = dt
        self.theta_dot = 0

    def step(self):
        self.theta = self.theta + self.theta_dot * self.dt
        self.theta = self.theta % (2 * pi)

    def msg(self):
        return (self.id, self.theta, self.dir, self.max_id)

    def stf(self):
        for msg in self.msgs:
            msg_id, msg_theta, msg_dir, msg_max_id = msg
            
            if msg_max_id > self.max_id:
                self.max_id = msg_max_id
                self.dir = msg_dir

    def ctl(self):
        base_velocity = self.dir * self.speed
        
        if not self.msgs:
            self.theta_dot = base_velocity
            return
        
        # Find closest neighbor in direction of travel
        min_dist = float('inf')
        for msg in self.msgs:
            msg_id, msg_theta, msg_dir, msg_max_id = msg
            
            if self.dir == 1:  # Moving CCW
                dist = dist_ccw(self.theta, msg_theta)
            else:  # Moving CW
                dist = dist_cw(self.theta, msg_theta)
            
            if 0 < dist < min_dist:
                min_dist = dist
        
        if min_dist == float('inf'):
            self.theta_dot = base_velocity
            return
        
        # Target spacing for uniform distribution
        target_spacing = 2 * pi / self.n
        
        # Modulate speed based on spacing
        if min_dist < target_spacing:
            speed_factor = min_dist / target_spacing
        else:
            speed_factor = min(min_dist / target_spacing, 1.5)
        
        self.theta_dot = base_velocity * speed_factor

    def clear_msgs(self):
        self.msgs = []

    def add_msg(self, msg):
        self.msgs.append(msg)
        


if __name__=="__main__":

    # create any parameters you need here (e.g., number of agents, communication radius, etc.)
    n = 45
    r_comm = 2*pi/20
    max_iter = 1000
    alpha = 0.05
    speed = 0.1
    dt = 0.1
    # generate positions randomly from 0 to 2 pi and directions randomly between clockwise and counterclockwise
    
    
    # initialize array of agents with -1 for direction and i for i-max
    agents = []
    for i in range(n):
        theta_init = random.uniform(0, 2 * pi)
        agents.append(AgreePursueAgent(id=i, theta=theta_init, n=n, speed=speed, dt=dt))
    # run_ctl_sim_circle from distributed_algorithm, passing in your agents as input
    run_ctl_sim_circle(agents, max_iter, r_comm=r_comm)
    # sim needs to simulate continuous dynamics with discrete time messages and updates
