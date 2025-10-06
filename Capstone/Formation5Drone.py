# Formation5Drone.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from network_agent import DynamicAgent
from distributed_algorithm import run_sim

class DroneAgent(DynamicAgent):
    """
    Individual drone agent for formation control.
    
    Attributes:
        id (int): Unique agent identifier
        state (np.ndarray): Current position [x, y] in R^2
        role (str): 'leader' or 'follower'
        formation_offset (np.ndarray): Desired [dx, dy] offset from leader
        alpha (float): Control gain for convergence speed
        msgs (list): Buffer for incoming messages, each msg is tuple (id, state, role)
        state_hist (list): History of states for plotting
        val (np.ndarray): Alias for state (for compatibility with plot_agents)
    """
    
    def __init__(self, id, state, role='follower', formation_offset=None, alpha=0.5):
        """
        Initialize drone agent.
        
        Args:
            id (int): Agent ID
            state (list or np.ndarray): Initial position [x, y]
            role (str): 'leader' or 'follower'
            formation_offset (np.ndarray or None): Desired offset from leader [dx, dy]
            alpha (float): Control gain (0 < alpha < 1 for stability)
        """
        self.id = id
        self.state = np.array(state, dtype=float)  # np.ndarray shape (2,)
        self.val = self.state  # Alias for compatibility with plotting
        self.role = role  # str: 'leader' or 'follower'
        self.formation_offset = formation_offset  # np.ndarray shape (2,) or None
        self.alpha = alpha  # float
        self.msgs = []  # list of tuples: [(id, state, role), ...]
        self.state_hist = [self.state.copy()]  # list of np.ndarray
        
    def msg(self):
        """
        Create message to broadcast to neighbors.
        
        Returns:
            tuple: (id, state, role) where
                   id (int): this agent's ID
                   state (np.ndarray): current position [x, y]
                   role (str): 'leader' or 'follower'
        """
        return (self.id, self.state.copy(), self.role)
    
    def stf(self):
        """
        State Transition Function: Update state based on control law.
        This runs after all messages are received.
        """
        # Compute control input (desired velocity direction)
        desired_velocity = self.compute_control()  # np.ndarray shape (2,)
        
        # Update state using simple integration
        self.state = self.state + self.alpha * desired_velocity
        self.val = self.state  # Keep val synchronized
        self.state_hist.append(self.state.copy())
        
    def compute_control(self):
        """
        Compute control law (formation tracking).
        
        Returns:
            np.ndarray: Control input (velocity direction) shape (2,)
        """
        if self.role == 'leader':
            # Leader stays stationary for now
            return np.zeros(2)  # np.ndarray shape (2,)
        
        # Follower: search for leader's state in messages
        leader_state = None  # Will be np.ndarray shape (2,) when found
        for msg_id, msg_state, msg_role in self.msgs:
            if msg_role == 'leader':
                leader_state = msg_state  # np.ndarray shape (2,)
                break
        
        if leader_state is None:
            # No leader found, don't move
            return np.zeros(2)
        
        # Desired position = leader's position + formation offset
        desired_pos = leader_state + self.formation_offset  # np.ndarray shape (2,)
        
        # Proportional control: move toward desired position
        error = desired_pos - self.state  # np.ndarray shape (2,)
        
        return error  # Control input proportional to error
    
    def clear_msgs(self):
        """Clear message buffer for next iteration."""
        self.msgs = []
    
    def add_msg(self, msg):
        """
        Add received message to buffer.
        
        Args:
            msg (tuple): (id, state, role) from neighboring agent
        """
        self.msgs.append(msg)
    
    def ctl(self):
        """
        Control output (required by DynamicAgent interface).
        For now, just returns current state.
        
        Returns:
            np.ndarray: Current state [x, y]
        """
        return self.state
    
    def step(self):
        """
        Dynamics integration step (required by DynamicAgent interface).
        Currently not used - state update happens in stf().
        """
        pass


if __name__ == "__main__":
    # Simulation parameters
    n = 5  # int: Total number of agents (1 leader + 4 followers)
    max_iter = 200  # int: Number of simulation iterations
    alpha = 0.1  # float: Control gain (smaller = smoother, slower convergence)
    
    # Formation geometry: square pattern around leader
    # Each follower has a desired offset from the leader's position
    formation_offsets = {
        # Agent ID: offset vector [dx, dy]
        1: np.array([3, 0]),    # Right of leader
        2: np.array([0, 3]),    # Above leader
        3: np.array([-3, 0]),   # Left of leader
        4: np.array([0, -3])    # Below leader
    }
    
    # Initialize agents
    agents = []  # list of DroneAgent objects
    
    # Create leader at center of environment
    agents.append(DroneAgent(
        id=0, 
        state=[50, 50],  # Center of 100x100 environment
        role='leader'
    ))
    
    # Create followers at random initial positions
    for i in range(1, n):
        init_pos = np.random.uniform(40, 60, 2)  # Random position near leader
        agents.append(DroneAgent(
            id=i, 
            state=init_pos, 
            role='follower',
            formation_offset=formation_offsets[i],
            alpha=alpha
        ))
    
    # Communication graph: fully connected (all agents can communicate)
    G = nx.complete_graph(n)  # networkx.Graph object
    
    # Environment limits for plotting [x_min, x_max, y_min, y_max]
    env_lims = [0, 100, 0, 100]
    
    # Run simulation with visualization
    agents = run_sim(
        agents,  # list of DroneAgent
        G,  # networkx.Graph
        max_iter,  # int
        return_agents=True,  # bool: return updated agents
        plotYN=True,  # bool: enable real-time plotting
        env_lims=env_lims  # list: environment boundaries
    )
    
    print("Formation achieved!")
    print(f"Final positions:")
    for agent in agents:
        print(f"  Agent {agent.id} ({agent.role}): {agent.state}")