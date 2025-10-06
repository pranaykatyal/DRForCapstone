import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from network_agent import NetworkAgent, DynamicAgent
from math import sin, cos

from utils import plot_agents, plot_agents_circle, dist_ccw, dist_cw

def run_sim(agents: list[NetworkAgent], G: nx.graph, max_iter: int, return_agents: bool = False, ctl: bool = False, plotYN: bool = False, env_lims: list[int] = [0, 100, 0, 100]):
    
    n = len(agents)

    if plotYN:
        plt.ion()
        n = len(agents)
        fig, axes = plot_agents(agents)
        centroid = sum([agent.val for agent in agents])/len(agents)
        circ = plt.Circle((centroid[0],centroid[1]),0.5,fill=False)
        axes.set_aspect( 1 )
        axes.add_artist( circ )
        plt.show()

    # number of time steps is equal to diameter of the graph
    for l in range(max_iter):
        for ii,jj in product(range(n),range(n)):
            if (ii,jj) in G.edges and ii!=jj:
                agents[ii].add_msg(agents[jj].msg())

        for ii in range(n):
            agents[ii].stf()

        # simulate dynamics if they are included
        if ctl == True:

            for ii in range(n):
                agents[ii].ctl()

            for ii in range(n):
                agents[ii].step()


        # clear messages for next round
        for ii in range(n):
            agents[ii].clear_msgs()

        if plotYN:
            x = [agent.val[0] for agent in agents]
            y = [agent.val[1] for agent in agents]
            axes.clear()  # clearing the axes
            axes.scatter(x,y)  # creating new scatter chart with updated data    
            # axes.set_aspect( 1 )        
            centroid = sum([agent.val for agent in agents])/len(agents)
            circ = plt.Circle((centroid[0],centroid[1]),1,fill=True)
            axes.set_aspect( 1 )
            axes.add_artist( circ )
            axes.set_xlim(env_lims[0],env_lims[1])
            axes.set_ylim(env_lims[2],env_lims[3])
            centroid = sum([agent.val for agent in agents])/len(agents)
            fig.canvas.draw()  # forcing the artist to redraw itself
            plt.pause(0.001)

    if return_agents:
        return agents
    

def run_ctl_sim_circle(agents: list[DynamicAgent], max_iter: int, r_comm = 1, return_agents: bool = False):
    plt.ion()
    n = len(agents)
    fig, axes = plot_agents_circle(agents)
    plt.show()
    # number of time steps is equal to diameter of the graph
    for l in range(max_iter):
        for ii,jj in product(range(n),range(n)):
            if (dist_ccw(agents[ii].theta,agents[jj].theta)<=r_comm or dist_cw(agents[ii].theta,agents[jj].theta)<=r_comm) and ii!=jj:
                agents[ii].add_msg(agents[jj].msg())

        for ii in range(n):
            agents[ii].stf()

        for ii in range(n):
            agents[ii].ctl()

        for ii in range(n):
            agents[ii].step()
            
        # clear messages for next round
        for ii in range(n):
            agents[ii].clear_msgs()

        # fig = plot_agents(agents)
        x = [cos(agent.theta) for agent in agents]
        y = [sin(agent.theta) for agent in agents]
        axes.clear()  # clearing the axes
        axes.scatter(x,y)  # creating new scatter chart with updated data    
        circ = plt.Circle((0,0),1,fill=False)
        axes.set_aspect( 1 )
        axes.add_artist( circ )
        axes.set_xlim(-1.5,1.5)
        axes.set_ylim(-1.5,1.5)
        fig.canvas.draw()  # forcing the artist to redraw itself
        plt.pause(0.001)
    plt.pause(5)

    if return_agents:
        return agents