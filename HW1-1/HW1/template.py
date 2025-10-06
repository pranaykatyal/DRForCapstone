import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from network_agent import NetworkAgent
from distributed_algorithm import run_sim, run_sim_tVar
np.random.seed(42)

# Define your agent as a subclass of Network Agent
# Include all of the necessary methods
class ConsensusAgent(NetworkAgent):

    def __init__(self, id = 0, x = 0.0, alpha = 0.5):
        self.ID = id
        self.x = x
        self.msgs = []
        self.alpha = alpha
        self.history = [x]
            
    def msg(self):
        return self.x
    
    def add_msg(self,msg):
        self.msgs.append(msg)
    
    def stf(self):
        total_diff  = 0
        for msg in self.msgs:
            total_diff += (msg-self.x)
        self.x += self.alpha * total_diff
        self.history.append(self.x)
        
    
    def clear_msgs(self):
        self.msgs = []
    
# Define any helper functions you would like to use
def extract_animation_data(agents):
    frames = []
    # How many time steps do we have? (length of any agent's history)
    num_steps = len(agents[0].history)
    
    # For each time step
    for step in range(num_steps):
        frame = []
        # For each agent, get their value at this time step
        for agent in agents:
            frame.append(agent.history[step])
        frames.append(frame)
    
    return frames

def generate_time_varying_graphs(G, max_iter):
    graphs = []
    original_edges = len(G.edges)  # Make sure this line is here
    
    for iteration in range(max_iter):
        new_graph = G.copy()
        edge_list = list(new_graph.edges)
        
        if len(edge_list) > 0:
            num_to_remove = np.random.randint(int(0.5*len(edge_list)), int(0.9*len(edge_list)))  # Remove 80-95% of edges
            if num_to_remove > 0:
                edges_to_remove = np.random.choice(len(edge_list), size=num_to_remove, replace=False)
                edges_to_remove = [edge_list[i] for i in edges_to_remove]
                new_graph.remove_edges_from(edges_to_remove)
        
        # Print graph statistics
        current_edges = len(new_graph.edges)
        is_connected = nx.is_connected(new_graph)
        print(f"Iteration {iteration}: {current_edges}/{original_edges} edges, Connected: {is_connected}")
        
        graphs.append(new_graph)
    
    return graphs
def compute_lyapunov_function(agent_states, L):
    """Compute V_G(x) = (1/2) * x^T * L * x"""
    x = np.array(agent_states)
    return 0.5 * x.T @ L @ x


# Main loop
if __name__=="__main__":

    # Create a graph with the appropriate number of nodes
    n = 5
    # G = nx.complete_graph(n)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4,2)])  # Simple directed cycle
    lyapunov_values = []
    
    # Get the Laplacian matrix
    L = nx.laplacian_matrix(G).toarray()
    print("Laplacian Matrix:")
    print(L)

    # For directed graphs, also check the adjacency and degree matrices
    A = nx.adjacency_matrix(G).toarray()
    D = np.diag([G.out_degree(i) for i in G.nodes()])
    print("\nAdjacency Matrix:")
    print(A)
    print("\nDegree Matrix:")
    print(D)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(L)
    print("\nEigenvalues:", eigenvalues)
    
    # Step 1: Analyze complete graph
    G_complete = nx.complete_graph(5)
    L_complete = nx.laplacian_matrix(G_complete).toarray()
    eigs_complete = np.linalg.eigvals(L_complete)
    max_eig_complete = np.max(np.real(eigs_complete))

    print(f"Complete graph largest eigenvalue: {max_eig_complete}")

    # Step 2: Use your directed graph eigenvalues
    max_eig_directed = np.max(np.real(eigenvalues))  # You already have this
    print(f"Directed graph largest eigenvalue: {max_eig_directed}")

    # Step 3: Predict the breaking point
    predicted_alpha_limit = 0.40 * (max_eig_complete / max_eig_directed)
    print(f"Predicted α limit for directed graph: {predicted_alpha_limit}")
    
    
    # Initialize the agents with appropriate initial values
    val = np.random.randint(0, 10, size=n)
    agents = [None] * n
    alpha = 0.2
    
    for i in range(n):
        agents[i] = ConsensusAgent(i,val[i],alpha)
    print(f"Initial values: {[agent.x for agent in agents]}")
    
    # Generate time-varying graphs and run simulation
    max_iter = 15
    # graphs = generate_time_varying_graphs(G, max_iter)
    # agents = run_sim_tVar(agents, graphs, max_iter, return_agents=True, ctl=False)
    # agents = run_sim(agents, G, max_iter, return_agents=True, ctl=False)
    
    # Manual simulation to track Lyapunov function
    # max_iter = 15
    # lyapunov_values = []

    for iteration in range(max_iter):
        # Get current agent states
        current_states = [agent.x for agent in agents]
        
        # Compute Lyapunov function
        V = compute_lyapunov_function(current_states, L)
        lyapunov_values.append(V)
        
        # Manual consensus step (replicate what run_sim does)
        # Send messages
        for i in range(len(agents)):
            for j in range(len(agents)):
                if G.has_edge(i, j) and i != j:
                    agents[i].add_msg(agents[j].msg())
        
        # Update states
        for agent in agents:
            agent.stf()
        
        # Clear messages
        for agent in agents:
            agent.clear_msgs()

    print(f"Final values: {[agent.x for agent in agents]}")

    # Plot Lyapunov function
    plt.figure()
    plt.plot(lyapunov_values)
    plt.xlabel('Iteration')
    plt.ylabel('Lyapunov Function V_G(x)')
    plt.title('Energy Function - Should Always Decrease')
    plt.grid()
    plt.show()
    # print(f"Final values: {[agent.x for agent in agents]}")
    
    # Animation with correct graph visualization
    frames = extract_animation_data(agents)
    print(np.shape(frames))
    pos = nx.spring_layout(G)  # Use same positions for all graphs
    plt.figure(figsize=(12, 5))

    for i in range(len(frames)):
        plt.clf()  # Clear everything
        
        # Left subplot - Graph with node values (SHOWING ACTUAL GRAPH USED)
        plt.subplot(1, 2, 1)
        frame_val = frames[i]
        
        labels = {}
        for j, value in enumerate(frame_val):
            labels[j] = f"{value:.4f}"
        
        # Draw the actual graph used at this iteration, not the original complete graph
        # current_graph = graphs[i-1]
        # nx.draw(current_graph, pos=pos, with_labels=False)
        # nx.draw_networkx_labels(current_graph, pos=pos, labels=labels)

        nx.draw(G, pos=pos, with_labels=False)
        nx.draw_networkx_labels(G, pos=pos, labels=labels)
        # Show connectivity status in title
        # is_connected = nx.is_connected(current_graph)
        # edge_count = len(current_graph.edges)
        # plt.title(f"Network - Frame {i}\n{edge_count} edges, Connected: {is_connected}")
        is_connected = nx.is_strongly_connected(G)
        edge_count = len(G.edges)
        plt.title(f"Network - Frame {i}\n{edge_count} edges, Connected: {is_connected}")
        
        # Right subplot - Convergence plot
        plt.subplot(1, 2, 2)
        for agent_idx, agent in enumerate(agents):
            plt.plot(agent.history[:i+1], label=f'Agent {agent_idx}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Convergence - Iteration {i}')
        plt.legend()
        plt.grid()
        plt.ylim(0, 10)  # Adjust range as needed
        
        plt.tight_layout()  # Prevents overlap
        plt.pause(0.5)

    plt.show()