# Formation5Drone.py
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from network_agent import DynamicAgent
from distributed_algorithm import run_sim

class DroneAgent(DynamicAgent):
    """
    Drone agent for distributed formation control around a target.
    
    Attributes:
        id (int): Unique agent identifier (0-4)
        state (np.ndarray): Current position [x, y] in R^2
        target_pos (np.ndarray): Shared target position [x, y]
        formation_radius (float): Distance from target for formation
        alpha (float): Control gain for convergence speed
        msgs (list): Buffer for messages from neighbors
        val (np.ndarray): Alias for state (plotting compatibility)
    """
    
    def __init__(self, id, state, target_pos, formation_radius=5.0, alpha=0.1):
        """
        Initialize drone agent.
        
        Args:
            id (int): Agent ID (0-4), determines position in formation
            state (list/np.ndarray): Initial position [x, y]
            target_pos (np.ndarray): Target location [x, y]
            formation_radius (float): Radius of formation circle around target
            alpha (float): Control gain (0 < alpha <= 1)
        """
        self.id = id  # int: 0-4
        self.state = np.array(state, dtype=float)  # np.ndarray shape (2,)
        self.val = self.state  # Alias for plotting
        self.target_pos = np.array(target_pos, dtype=float)  # np.ndarray shape (2,)
        self.formation_radius = formation_radius  # float
        self.alpha = alpha  # float
        self.msgs = []  # list of tuples: [(id, state), ...]
        self.state_hist = [self.state.copy()]
        
    def msg(self):
        """
        Broadcast current position to neighbors.
        
        Returns:
            tuple: (id, state) where state is np.ndarray [x, y]
        """
        return (self.id, self.state.copy())
    
    def stf(self):
        """
        State Transition Function: Update position based on formation control.
        Runs after receiving all neighbor messages.
        """
        # Compute control based on formation objective
        control_input = self.compute_formation_control()  # np.ndarray shape (2,)
        
        # Update state
        self.state = self.state + self.alpha * control_input
        self.val = self.state  # Sync alias
        self.state_hist.append(self.state.copy())
        
    def compute_formation_control(self):
        """
        Compute control to drive agent to assigned formation position.
        
        Formation: Pentagon around target, agent i at angle (2πi/5)
        
        Returns:
            np.ndarray: Control velocity [vx, vy]
        """
        # Compute assigned angle based on ID
        # 5 agents form a pentagon: angles at 0°, 72°, 144°, 216°, 288°
        n_agents = 5  # Total number of agents
        assigned_angle = (2 * np.pi * self.id) / n_agents  # float: radians
        
        # Desired position = target + radius * [cos(θ), sin(θ)]
        desired_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle)
        ])  # np.ndarray shape (2,)
        
        # Proportional control: move toward desired position
        error = desired_pos - self.state  # np.ndarray shape (2,)
        
        return error  # Control input
    
    def clear_msgs(self):
        """Clear message buffer for next iteration."""
        self.msgs = []
    
    def add_msg(self, msg):
        """
        Receive message from neighbor.
        
        Args:
            msg (tuple): (id, state) from another agent
        """
        self.msgs.append(msg)
    
    def ctl(self):
        """Required by DynamicAgent interface. Returns current state."""
        return self.state
    
    def step(self):
        """Required by DynamicAgent interface. State update in stf()."""
        pass


if __name__ == "__main__":
    # Simulation parameters
    n = 5  # int: Number of drones
    max_iter = 300  # int: Simulation steps
    alpha = 0.15  # float: Control gain (higher = faster convergence)
    formation_radius = 8.0  # float: Formation size
    
    # Target position (stationary for now)
    target_pos = np.array([50.0, 50.0])  # np.ndarray shape (2,)
    
    # Environment boundaries
    env_lims = [0, 100, 0, 100]  # list: [x_min, x_max, y_min, y_max]
    
    # Initialize agents at random positions
    agents = []  # list of DroneAgent
    np.random.seed(42)  # For reproducibility
    
    for i in range(n):
        # Random initial position in environment
        init_pos = np.random.uniform(20, 80, 2)  # np.ndarray shape (2,)
        
        agents.append(DroneAgent(
            id=i,
            state=init_pos,
            target_pos=target_pos,
            formation_radius=formation_radius,
            alpha=alpha
        ))
    
    # Communication graph: fully connected (all agents communicate)
    G = nx.complete_graph(n)  # networkx.Graph
    
    # Run simulation
    print("Starting formation control simulation...")
    print(f"Target at: {target_pos}")
    print(f"Formation radius: {formation_radius}")
    print(f"Initial positions:")
    for agent in agents:
        print(f"  Agent {agent.id}: {agent.state}")
    
    agents = run_sim(
        agents,
        G,
        max_iter,
        return_agents=True,
        plotYN=True,
        env_lims=env_lims
    )
    
    print("\nFormation achieved!")
    print("Final positions:")
    for agent in agents:
        # Compute distance to assigned position
        angle = (2 * np.pi * agent.id) / n
        desired = target_pos + formation_radius * np.array([np.cos(angle), np.sin(angle)])
        error = np.linalg.norm(agent.state - desired)
        print(f"  Agent {agent.id}: {agent.state}, error: {error:.3f}")