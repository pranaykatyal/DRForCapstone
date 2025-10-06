import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from network_agent import NetworkAgent
from distributed_algorithm import run_sim

class FormationAgent(NetworkAgent):

    def __init__(self,d,id=0,val=0,alpha=0.05):
        self.my_id = id
        self.val = val
        self.d = {}
        for key, value in d.items():
            if list(key)[0]==id: 
                self.d[list(key)[1]]=value
        self.msgs = []
        self.val_hist = [val]
        self.alpha = alpha

    def msg(self):
        return (self.my_id,self.val)
    
    def stf(self):
        error = 0
        for msg in self.msgs:
            error += msg[1]-self.val-self.d[msg[0]]
        self.val = self.val + self.alpha*error
        self.val_hist.append(self.val)

    def clear_msgs(self):
        self.msgs = []

    def add_msg(self,msg):
        self.msgs.append(msg)

def make_agents(n,init_vals,d,alpha=0.05):
    agents = []

    for ii in range(n):
        agents.append(FormationAgent(d,ii,init_vals[ii],alpha))

    return agents

if __name__=="__main__":
    
    # number of agents
    n = 4
    max_iter = 50
    env_max = 50
    env_lims = [0, env_max, 0, env_max]
    alpha = 0.05 #update rate

    # edge list (undirects)
    edges = [[0,1],[1,3],[0,2],[2,3]]

    # prescribe distances in a square
    d = {(0,1): [5,0], 
         (1,0): [-5,0],
         (1,3): [0,-5], 
         (3,1): [0,5],
         (0,2): [0,-5],
         (2,0): [0,5],
         (2,3): [5,0],
         (3,2): [-5,0]}

    G = nx.Graph()

    # add nodes to graph
    for ii in range(n):
        G.add_node(ii)

    # add edges
    for e in edges:
        G.add_edge(e[0],e[1])

    # initialize agents
    init_vals = np.random.uniform(0,env_max,(n,2))
    agents = make_agents(n,init_vals,d,alpha=alpha)

    # run the simulation
    agents = run_sim(agents,G,max_iter,return_agents=True,plotYN=True,env_lims=env_lims) # uncomment to run sim and return the agents