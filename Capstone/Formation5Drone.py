# Formation5Drone.py - UPDATED VERSION
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from network_agent import DynamicAgent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


# Visualization toggles
# -----------------------------------------------------------
# VIS_GRAPH: Shows static graph snapshots (nodes = drones, edges = neighbors)
# VIS_BARRIERS:Plots the evolution of pairwise barrier functions h_ij(t)
# VIS_SPHERES: Shows 3D animation with safety spheres (radius = d_safe)
# -----------------------------------------------------------
VIS_GRAPH = True
VIS_BARRIERS = True
VIS_SPHERES = True

class DroneAgent(DynamicAgent):
    """
    Enhanced 3D drone agent with velocity tracking for CBF integration.
    
    Attributes:
        id (int): Unique agent identifier (0-4)
        position (np.ndarray): Current position [x, y, z] in R^3
        velocity (np.ndarray): Current velocity [vx, vy, vz] in R^3
        acceleration (np.ndarray): Current acceleration [ax, ay, az] in R^3
        target_pos (np.ndarray): Shared target position [x, y, z]
        formation_radius (float): Distance from target for formation
        dt (float): Simulation timestep
        msgs (list): Buffer for messages from neighbors
    """
    
    def __init__(self, id, state_3d, target_pos_3d, formation_radius=5.0, 
                 Kp=1.0, Kd=0.5, dt=0.1):
        """
        Initialize 3D drone agent with velocity tracking.
        
        Args:
            id (int): Agent ID (0-4), determines position in formation
            state_3d (list/np.ndarray): Initial position [x, y, z]
            target_pos_3d (np.ndarray): Target location [x, y, z]
            formation_radius (float): Radius of formation circle around target
            Kp (float): Proportional gain for position control
            Kd (float): Derivative gain for velocity damping
            dt (float): Simulation timestep
        """
        self.id = id  # int: 0-4
        self.position = np.array(state_3d, dtype=float)  # np.ndarray shape (3,)
        self.velocity = np.zeros(3, dtype=float)  # np.ndarray shape (3,)
        self.acceleration = np.zeros(3, dtype=float)  # np.ndarray shape (3,)
        
        self.target_pos = np.array(target_pos_3d, dtype=float)  # np.ndarray shape (3,)
        self.formation_radius = formation_radius  # float
        self.Kp = Kp  # Proportional gain
        self.Kd = Kd  # Derivative gain
        self.dt = dt  # Simulation timestep
        
        self.msgs = []  # list of tuples: [(id, position, velocity), ...]
        
        # History for plotting and analysis
        self.position_hist = [self.position.copy()]
        self.velocity_hist = [self.velocity.copy()]
        self.acceleration_hist = [self.acceleration.copy()]
        
        # For compatibility with old plotting code
        self.state = self.position[:2]  # [x, y] for 2D plots
        self.val = self.state
        
    def msg(self):
        """
        Broadcast current state to neighbors.
        
        Returns:
            tuple: (id, position, velocity) for CBF communication
        """
        return (self.id, self.position.copy(), self.velocity.copy())
    
    def compute_desired_acceleration(self):
        """
        Compute nominal acceleration from formation control law.
        This is what the formation controller wants to do (may be unsafe).
        
        Formation: Pentagon around target in XY plane
        Control Law: PD control to desired position
        
        Returns:
            np.ndarray: Desired acceleration [ax, ay, az]
        """
        # Compute assigned position in formation (pentagon)
        n_agents = 5
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        # Formation in XY plane at target Z height
        desired_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0  # Formation stays in horizontal plane
        ])
        
        # PD control: acc = Kp*(pos_error) + Kd*(vel_error)
        pos_error = desired_pos - self.position
        vel_error = -self.velocity  # Desired velocity = 0 (hover in formation)
        
        acc_desired = self.Kp * pos_error + self.Kd * vel_error
        
        return acc_desired
    
    def update_dynamics(self, acc_safe):
        """
        Update position and velocity using safe acceleration from CBF.
        Uses simple Euler integration.
        
        Args:
            acc_safe (np.ndarray): Safe acceleration from CBF filter [ax, ay, az]
        """
        # Store current acceleration
        self.acceleration = acc_safe.copy()
        
        # Integrate acceleration → velocity
        self.velocity = self.velocity + self.acceleration * self.dt
        
        # Integrate velocity → position
        self.position = self.position + self.velocity * self.dt
        
        # Update compatibility fields
        self.state = self.position[:2]
        self.val = self.state
        
        # Store history
        self.position_hist.append(self.position.copy())
        self.velocity_hist.append(self.velocity.copy())
        self.acceleration_hist.append(self.acceleration.copy())
    
    # Legacy methods for compatibility with distributed_algorithm.py
    def stf(self):
        """Legacy: State Transition Function (for old simulation code)."""
        # Compute desired acceleration
        acc_desired = self.compute_desired_acceleration()
        
        # In legacy mode, just apply desired acceleration directly (no CBF)
        self.update_dynamics(acc_desired)
    
    def clear_msgs(self):
        """Clear message buffer for next iteration."""
        self.msgs = []
    
    def add_msg(self, msg):
        """
        Receive message from neighbor.
        
        Args:
            msg (tuple): (id, position, velocity) from another agent
        """
        self.msgs.append(msg)
    
    def ctl(self):
        """Required by DynamicAgent interface. Returns current state."""
        return self.position
    
    def step(self):
        """Required by DynamicAgent interface. State update happens in stf()."""
        pass
    
    def get_formation_error(self):
        """
        Compute distance from ideal formation position.
        
        Returns:
            float: Distance in meters
        """
        n_agents = 5
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        ideal_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0
        ])
        
        return np.linalg.norm(self.position - ideal_pos)

# =============== Utility Animation Functions ===============

def animate_safety_spheres(agents, cbf_filter):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Safety Spheres (d_safe zones)")
    n = len(agents)
    scatters = [ax.plot([], [], [], 'o', label=f'Agent {i}')[0] for i in range(n)]
    T = len(agents[0].position_hist)

    # Draw safety zones as spheres
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_s = cbf_filter.d_safe*np.cos(u)*np.sin(v)
    y_s = cbf_filter.d_safe*np.sin(u)*np.sin(v)
    z_s = cbf_filter.d_safe*np.cos(v)
    spheres = [ax.plot_surface(x_s, y_s, z_s, alpha=0.05, color='r') for _ in range(n)]

    def update(frame):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 3)
        ax.set_title(f"Safety Spheres at t={frame * agents[0].dt:.2f}s")
        for i, agent in enumerate(agents):
            p = agent.position_hist[frame]
            ax.scatter(*p, label=f"A{i}")
            # draw sphere
            ax.plot_surface(x_s + p[0], y_s + p[1], z_s + p[2], alpha=0.1, color='r')
        return scatters

    anim = FuncAnimation(fig, update, frames=range(0, T, 5), interval=100)
    plt.show()


def plot_barrier_functions(agents, cbf_filter):
    import matplotlib.pyplot as plt
    n = len(agents)
    T = len(agents[0].position_hist)
    time = np.arange(T) * agents[0].dt

    plt.figure(figsize=(10,6))
    for i in range(n):
        for j in range(i+1, n):
            h_vals = [
                np.linalg.norm(agents[i].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                for k in range(T)
            ]
            plt.plot(time, h_vals, label=f"h_{i}{j}")
    plt.axhline(0, color='k', linestyle='--', label='Safety boundary')
    plt.xlabel('Time (s)')
    plt.ylabel('Barrier Function $h_{ij}(t)$')
    plt.title('Pairwise GCBF Safety Functions Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_graph_snapshots(agents, cbf_filter, snapshots=[0, -1]):
    import networkx as nx
    import matplotlib.pyplot as plt
    n = len(agents)
    dt = agents[0].dt
    for idx in snapshots:
        pos = np.array([a.position_hist[idx] for a in agents])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < cbf_filter.R_sense:
                    color = 'r' if dist < cbf_filter.d_safe * 1.1 else 'gray'
                    G.add_edge(i, j, color=color)
        plt.figure(figsize=(6,6))
        colors = [G[u][v]['color'] for u,v in G.edges()]
        nx.draw(G, pos={i: pos[i,:2] for i in range(n)}, node_color='skyblue',
                with_labels=True, edge_color=colors, node_size=500)
        plt.title(f"Agent Graph at t = {idx * dt:.2f}s")
        plt.axis('equal')
        plt.show()





# ==================== SIMULATION FUNCTIONS ====================

def run_formation_with_cbf(n_agents=5, max_iter=500, dt=0.1, use_cbf=True, animate=True):
    """
    Run formation control simulation with optional CBF safety filter.
    
    Args:
        n_agents: Number of drones (default 5 for pentagon)
        max_iter: Number of simulation steps
        dt: Timestep in seconds
        use_cbf: Whether to use CBF safety filter
        animate: Whether to animate the simulation or not.
    
    Returns:
        agents: List of DroneAgent objects with trajectory history
        cbf_stats: CBF statistics (if use_cbf=True), else None
    """
    # Import CBF filter if needed
    if use_cbf:
        try:
            from cbf_safety import GraphCBFSafetyFilter
        except ImportError:
            print("WARNING: Could not import GraphCBFSafetyFilter")
            print("Place cbf_safety.py in the same directory")
            use_cbf = False
    
    # Initialize CBF filter
    cbf_filter = None
    if use_cbf:
        cbf_filter = GraphCBFSafetyFilter(
            n_drones=n_agents,
            safety_distance=1.5,      # Minimum 1.5m between drones
            sensing_radius=8.0,       # Detect neighbors within 8m
            alpha1=2.0,               # CBF responsiveness
            alpha2=1.0,               # CBF convergence rate
            max_acceleration=5.0      # Physical limit
        )
        
        # Check minimum required sensing radius
        v_max = 3.0  # Expected max velocity
        R_min = cbf_filter.compute_minimum_sensing_radius(
            gamma=cbf_filter.alpha2, 
            v_max=v_max
        )
        print(f"Minimum sensing radius for safety: {R_min:.2f}m")
        if cbf_filter.R_sense < R_min:
            print(f"⚠️  WARNING: Using R={cbf_filter.R_sense}m < {R_min:.2f}m")
            print("   Safety not guaranteed! Increase sensing_radius.")
    
    # Initialize agents
    target_pos = np.array([0.0, 0.0, 2.0])  # Formation center
    formation_radius = 5.0
    
    agents = []
    np.random.seed(42)  # Reproducibility
    
    print(f"\n{'='*60}")
    print(f"Formation Control Simulation")
    print(f"{'='*60}")
    print(f"Agents: {n_agents}")
    print(f"Target: {target_pos}")
    print(f"Formation radius: {formation_radius}m")
    print(f"CBF enabled: {use_cbf}")
    print(f"Timestep: {dt}s, Duration: {max_iter*dt:.1f}s")
    print(f"{'='*60}\n")
    
    for i in range(n_agents):
        # Random initial position
        init_pos = np.random.uniform(-10, 10, 3)
        init_pos[2] = np.random.uniform(1, 3)  # Reasonable altitude
        
        agents.append(DroneAgent(
            id=i,
            state_3d=init_pos,
            target_pos_3d=target_pos,
            formation_radius=formation_radius,
            Kp=1.0,
            Kd=0.5,
            dt=dt
        ))
        print(f"Agent {i} initialized at: {init_pos}")
    
    
    # ==================== ANIMATION SETUP ====================
    if animate:
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        
        
        # Generate colors for each drone
        colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
        
        # Initialize scatter plot for drones (empty at first)
        drone_scatter = ax.scatter([], [], [], s=200, c='blue', marker='o', 
                                  edgecolors='black', linewidths=2)
        
        # Plot target position
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                  c='red', marker='X', s=500, edgecolors='black', linewidths=2,
                  label='Target')
        
        # Plot ideal formation positions (green X's)
        ideal_positions = []
        for i in range(n_agents):
            angle = (2 * np.pi * i) / n_agents
            ideal_pos = target_pos + formation_radius * np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            ideal_positions.append(ideal_pos)
            ax.scatter(ideal_pos[0], ideal_pos[1], ideal_pos[2],
                      c='green', marker='x', s=200, alpha=0.5,
                      edgecolors='darkgreen', linewidths=2)
        
        # Draw formation circle
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = target_pos[0] + formation_radius * np.cos(theta)
        circle_y = target_pos[1] + formation_radius * np.sin(theta)
        circle_z = np.ones_like(theta) * target_pos[2]
        ax.plot(circle_x, circle_y, circle_z, 'g--', alpha=0.3, linewidth=2)
        
        # Initialize trajectory lines for each drone (empty)
        trajectory_lines = []
        for i in range(n_agents):
            line, = ax.plot([], [], [], color=colors[i], linewidth=1.5, 
                          alpha=0.6, label=f'Drone {i}')
            trajectory_lines.append(line)
        
        # Initialize velocity vectors (will be updated in loop)
        quiver = None
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('Real-time Formation Control', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        # Set initial axis limits
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(0, 5)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for live statistics
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # ==================== END ANIMATION SETUP ====================
    
    # Main simulation loop
    print(f"\nStarting simulation...")
    
    for iteration in range(max_iter):
        # Step 1: Each agent computes DESIRED acceleration
        acc_desired = np.zeros((n_agents, 3))
        for i, agent in enumerate(agents):
            acc_desired[i] = agent.compute_desired_acceleration()
        
        # Step 2: Collect current states
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])
        
        # Step 3: Filter through CBF (if enabled)
        if use_cbf:
            acc_safe = cbf_filter.filter_accelerations(
                positions, 
                velocities, 
                acc_desired,
                obstacles=None  # Add obstacles here if needed
            )
            
            # Safety check
            is_safe, violations = cbf_filter.check_safety(positions)
            if not is_safe:
                print(f"\n⚠️  SAFETY VIOLATION at iteration {iteration}:")
                for v in violations:
                    print(f"   {v}")
        else:
            # No CBF: use desired acceleration directly
            acc_safe = acc_desired
        
        # Step 4: Update each agent with safe acceleration
        for i, agent in enumerate(agents):
            agent.update_dynamics(acc_safe[i])
        
        # ==================== ANIMATION UPDATE ====================
        # Step 4.5: Update animation (every 2 iterations for smooth display)
        if animate and iteration % 2 == 0:
            # Update drone positions
            current_positions = np.array([agent.position for agent in agents])
            drone_scatter._offsets3d = (current_positions[:, 0], 
                                        current_positions[:, 1], 
                                        current_positions[:, 2])
            
            # Update trajectories
            for i, agent in enumerate(agents):
                traj = np.array(agent.position_hist)
                trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                trajectory_lines[i].set_3d_properties(traj[:, 2])
            
            # Update velocity vectors
            if quiver is not None:
                quiver.remove()
            
            # Draw velocity vectors for each drone
            current_velocities = np.array([agent.velocity for agent in agents])
            # Scale velocities for visibility
            vel_scale = 0.5
            quiver = ax.quiver(current_positions[:, 0], current_positions[:, 1], current_positions[:, 2],
                              current_velocities[:, 0]*vel_scale, 
                              current_velocities[:, 1]*vel_scale, 
                              current_velocities[:, 2]*vel_scale,
                              color='blue', alpha=0.6, arrow_length_ratio=0.3, linewidths=1.5)
            
            # Update info text
            formation_errors = [agent.get_formation_error() for agent in agents]
            avg_error = np.mean(formation_errors)
            max_error = np.max(formation_errors)
            avg_vel = np.mean([np.linalg.norm(a.velocity) for a in agents])
            
            # Check minimum distance between drones
            min_dist = float('inf')
            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    dist = np.linalg.norm(agents[i].position - agents[j].position)
                    min_dist = min(min_dist, dist)
            
            info_str = (f"Iteration: {iteration}/{max_iter}\n"
                       f"Time: {iteration*dt:.1f}s / {max_iter*dt:.1f}s\n"
                       f"Formation Error:\n"
                       f"  Avg: {avg_error:.3f}m\n"
                       f"  Max: {max_error:.3f}m\n"
                       f"Avg Velocity: {avg_vel:.3f}m/s\n"
                       f"Min Distance: {min_dist:.3f}m")
            
            if use_cbf:
                stats = cbf_filter.get_statistics()
                if stats.get('total_solves', 0) > 0:
                    activation_rate = stats.get('activation_rate', 0)
                    info_str += f"\nCBF Active: {activation_rate*100:.1f}%"
            
            info_text.set_text(info_str)
            
            # Dynamically adjust view limits
            all_x = current_positions[:, 0]
            all_y = current_positions[:, 1]
            all_z = current_positions[:, 2]
            
            margin = 3.0
            ax.set_xlim(min(all_x.min()-margin, -15), max(all_x.max()+margin, 15))
            ax.set_ylim(min(all_y.min()-margin, -15), max(all_y.max()+margin, 15))
            ax.set_zlim(max(0, all_z.min()-margin), all_z.max()+margin)
            
            # Update display
            plt.draw()
            plt.pause(0.01)
        # ==================== END ANIMATION UPDATE ====================
        
        
        # Step 5: Progress reporting
        if iteration % 50 == 0:
            formation_errors = [agent.get_formation_error() for agent in agents]
            avg_error = np.mean(formation_errors)
            max_error = np.max(formation_errors)
            
            avg_vel = np.mean([np.linalg.norm(a.velocity) for a in agents])
            
            print(f"Iter {iteration:3d}: "
                  f"Formation error: avg={avg_error:.3f}m, max={max_error:.3f}m, "
                  f"Avg velocity: {avg_vel:.3f}m/s")
    
    
    # Optional visualization calls (only if CBF filter is active)
    if use_cbf and cbf_filter is not None:
        if VIS_GRAPH:
            visualize_graph_snapshots(agents, cbf_filter, [0, max_iter // 2, -1])
        if VIS_BARRIERS:
            plot_barrier_functions(agents, cbf_filter)
        if VIS_SPHERES:
            animate_safety_spheres(agents, cbf_filter)
    else:
        print("\n[INFO] Skipping CBF visualizations — CBF filter disabled.")


    
    # Print final statistics
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    

    
    # Close animation and keep final frame
    if animate:
        plt.ioff()  # Turn off interactive mode
        plt.show(block=False)  # Keep the final frame visible
        print("\nAnimation window showing final state. Close to continue...")
    
    if use_cbf:
        stats = cbf_filter.get_statistics()
        print("\nCBF Statistics:")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
    
    final_errors = [agent.get_formation_error() for agent in agents]
    print(f"\nFinal Formation Errors:")
    for i, error in enumerate(final_errors):
        print(f"  Agent {i}: {error:.3f}m")
    print(f"  Average: {np.mean(final_errors):.3f}m")
    print(f"  Maximum: {np.max(final_errors):.3f}m")
    
    cbf_stats = cbf_filter.get_statistics() if use_cbf else None
    return agents, cbf_stats


def plot_results_3d(agents):
    """
    Visualize 3D trajectories and final formation.
    
    Args:
        agents: List of DroneAgent objects with history
    """
    n_agents = len(agents)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: 3D Trajectories
    ax1 = fig.add_subplot(131, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
    
    for i, agent in enumerate(agents):
        traj = np.array(agent.position_hist)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], linewidth=2, label=f'Drone {agent.id}')
        
        # Mark start and end
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                   color=colors[i], marker='o', s=100, edgecolors='black')
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   color=colors[i], marker='*', s=200, edgecolors='black')
    
    # Mark target
    target = agents[0].target_pos
    ax1.scatter(target[0], target[1], target[2], 
               color='red', marker='x', s=300, linewidths=3, label='Target')
    
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title('3D Drone Trajectories', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Formation (Top View)
    ax2 = fig.add_subplot(132)
    
    for i, agent in enumerate(agents):
        # Final position
        ax2.plot(agent.position[0], agent.position[1], 'o', 
                markersize=12, color=colors[i], label=f'Drone {agent.id}')
        
        # Ideal position
        angle = (2 * np.pi * agent.id) / n_agents
        ideal_x = agent.target_pos[0] + agent.formation_radius * np.cos(angle)
        ideal_y = agent.target_pos[1] + agent.formation_radius * np.sin(angle)
        ax2.plot(ideal_x, ideal_y, 'x', markersize=12, 
                color=colors[i], markeredgewidth=3)
        
        # Draw error line
        ax2.plot([agent.position[0], ideal_x], 
                [agent.position[1], ideal_y], 
                '--', color=colors[i], alpha=0.5, linewidth=1)
    
    # Draw formation circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = target[0] + agent.formation_radius * np.cos(theta)
    circle_y = target[1] + agent.formation_radius * np.sin(theta)
    ax2.plot(circle_x, circle_y, 'k--', alpha=0.3, linewidth=2)
    
    # Mark target
    ax2.plot(target[0], target[1], 'r+', markersize=20, markeredgewidth=3)
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('Final Formation (Top View)\no = actual, x = ideal', 
                 fontsize=12, fontweight='bold')
    ax2.axis('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Formation Error Over Time
    ax3 = fig.add_subplot(133)
    
    max_len = max(len(agent.position_hist) for agent in agents)
    time = np.arange(max_len) * agents[0].dt
    
    for i, agent in enumerate(agents):
        errors = [np.linalg.norm(np.array(agent.position_hist[j]) - 
                                 (agent.target_pos + agent.formation_radius * np.array([
                                     np.cos(2*np.pi*agent.id/n_agents),
                                     np.sin(2*np.pi*agent.id/n_agents),
                                     0.0
                                 ])))
                 for j in range(len(agent.position_hist))]
        ax3.plot(time[:len(errors)], errors, color=colors[i], 
                linewidth=2, label=f'Drone {agent.id}')
    
    # Average error
    avg_errors = []
    for t in range(max_len):
        errors_at_t = []
        for agent in agents:
            if t < len(agent.position_hist):
                ideal = agent.target_pos + agent.formation_radius * np.array([
                    np.cos(2*np.pi*agent.id/n_agents),
                    np.sin(2*np.pi*agent.id/n_agents),
                    0.0
                ])
                error = np.linalg.norm(agent.position_hist[t] - ideal)
                errors_at_t.append(error)
        if errors_at_t:
            avg_errors.append(np.mean(errors_at_t))
    
    ax3.plot(time[:len(avg_errors)], avg_errors, 'k--', 
            linewidth=3, label='Average', alpha=0.7)
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Formation Error (m)', fontsize=10)
    ax3.set_title('Formation Error Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_with_without_cbf():
    """
    Run two simulations side-by-side to compare CBF vs no CBF.
    """
    print("\n" + "="*60)
    print("COMPARISON: WITH vs WITHOUT CBF")
    print("="*60)
    
    # Run without CBF
    print("\n--- Running WITHOUT CBF ---")
    agents_no_cbf, _ = run_formation_with_cbf(
        n_agents=5, max_iter=300, dt=0.1, use_cbf=False
    )
    
    # Run with CBF
    print("\n--- Running WITH CBF ---")
    agents_with_cbf, cbf_stats = run_formation_with_cbf(
        n_agents=5, max_iter=300, dt=0.1, use_cbf=True
    )
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Formation errors
    errors_no_cbf = [agent.get_formation_error() for agent in agents_no_cbf]
    errors_with_cbf = [agent.get_formation_error() for agent in agents_with_cbf]
    
    print("\nFinal Formation Error:")
    print(f"  Without CBF: avg={np.mean(errors_no_cbf):.3f}m, "
          f"max={np.max(errors_no_cbf):.3f}m")
    print(f"  With CBF:    avg={np.mean(errors_with_cbf):.3f}m, "
          f"max={np.max(errors_with_cbf):.3f}m")
    
    # Check for collisions
    print("\nSafety Check (minimum inter-drone distance):")
    
    for name, agents in [("Without CBF", agents_no_cbf), 
                         ("With CBF", agents_with_cbf)]:
        min_dist = float('inf')
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                dist = np.linalg.norm(agents[i].position - agents[j].position)
                min_dist = min(min_dist, dist)
        print(f"  {name}: {min_dist:.3f}m")
    
    return agents_no_cbf, agents_with_cbf




# ==================== MAIN ====================

if __name__ == "__main__":
    # Option 1: Run with CBF
    agents, cbf_stats = run_formation_with_cbf(
        n_agents=5,
        max_iter=500,
        dt=0.05,
        use_cbf=True,  # Set to False to test without CBF
        animate=True
    )
    
    # Visualize results
    plot_results_3d(agents)
    
    # Option 2: Compare with and without CBF
    # agents_no_cbf, agents_with_cbf = compare_with_without_cbf()
    # plot_results_3d(agents_with_cbf)