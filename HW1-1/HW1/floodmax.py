import networkx as nx

from network_agent import NetworkAgent
from distributed_algorithm import run_sim

class FloodMaxAgent(NetworkAgent):

    def __init__(self,id=0,round=0,leader=None,diam=1):
        self.my_id = id
        self.round = round
        self.max_id = id
        self.leader = leader
        self.diam = diam
        self.msgs = []

    def msg(self):
        if self.round < self.diam:
            return self.max_id
        else:
            return []
    
    def stf(self):

        self.max_id = max(self.max_id,max(self.msgs))
        if self.round<self.diam:
            self.leader = None
        elif self.round == self.diam and self.max_id == self.my_id:
            self.leader = True
        elif self.round == self.diam and self.max_id > self.my_id:
            self.leader = False

        self.round += 1

    def clear_msgs(self):
        self.msgs = []

    def add_msg(self,msg):
        self.msgs.append(msg)

def make_agents(n,diam):
    agents = []

    for ii in range(n):
        agents.append(FloodMaxAgent(ii,0,None,diam))

    return agents

def print_max_id(agents):
    for agent in agents:
        print(agent.max_id)

if __name__=="__main__":
    
    # number of agents
    n = 25
    
    # generate a connected graph
    k = 3
    p = 0.25
    G = nx.connected_watts_strogatz_graph(n,k,p)
    diam = nx.diameter(G)

    # initialize agents
    agents = make_agents(n,diam)

    # Print max_id of all agents
    print('INITIAL MAX ID: ')
    print_max_id(agents)

    # run the simulation
    # run_sim(agents,G,diam) # uncomment to run sim without returning agents
    agents = run_sim(agents,G,diam,return_agents=True) # uncomment to run sim and return the agents

    # Print max_id of all agents
    print('FINAL MAX ID: ')
    print_max_id(agents)