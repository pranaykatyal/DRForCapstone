# Formation5Drone.py - COMPLETE VERSION WITH ALL FEATURES
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from network_agent import DynamicAgent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import logging
logging.getLogger('matplotlib.axes._base').setLevel(logging.ERROR)

# Visualization toggles
# -----------------------------------------------------------
# VIS_GRAPH_OVERLAY: Shows graph connectivity overlaid on main animation
# VIS_BARRIERS: Plots the evolution of pairwise barrier functions h_ij(t)
# VIS_2D_TOPVIEW: Shows real-time 2D top-down view alongside 3D view
# FOCUS_AGENT: Which agent to focus on for GCBF visualization (-1 for all, 0-4 for specific)
# CONVERGENCE_THRESHOLD: Formation error threshold to stop simulation (meters)
# -----------------------------------------------------------
VIS_GRAPH_OVERLAY = True
VIS_BARRIERS = True
VIS_2D_TOPVIEW = True
FOCUS_AGENT = 1  # Default to Agent 0 for GCBF visualization
CONVERGENCE_THRESHOLD = 0.05  # Stop when all agents within 5cm of ideal positions
CONVERGENCE_VELOCITY = 0.01  # And velocity below 1cm/s

class DroneAgent(DynamicAgent):
    """
    Enhanced 3D drone agent with velocity tracking for CBF integration.
    """
    
    def __init__(self, id, state_3d, target_pos_3d, formation_radius=5.0, 
                 Kp=0.5, Kd=1.2, dt=0.1, max_velocity=3.0, max_acceleration=2.0):
        """
        Initialize 3D drone agent with velocity tracking.
        """
        self.id = id  # int: 0-4
        self.position = np.array(state_3d, dtype=float)  # np.ndarray shape (3,)
        self.velocity = np.zeros(3, dtype=float)  # np.ndarray shape (3,) - starts at rest
        self.acceleration = np.zeros(3, dtype=float)  # np.ndarray shape (3,)
        
        self.target_pos = np.array(target_pos_3d, dtype=float)  # np.ndarray shape (3,)
        self.formation_radius = formation_radius  # float
        self.Kp = Kp  # Proportional gain (reduced from 1.0 to 0.5)
        self.Kd = Kd  # Derivative gain (increased from 0.5 to 1.2)
        self.dt = dt  # Simulation timestep
        self.max_velocity = max_velocity  # Velocity saturation
        self.max_acceleration = max_acceleration  # Acceleration saturation
        
        self.msgs = []  # list of tuples: [(id, position, velocity), ...]
        
        # History for plotting and analysis
        self.position_hist = [self.position.copy()]
        self.velocity_hist = [self.velocity.copy()]
        self.acceleration_hist = [self.acceleration.copy()]
        
        # For compatibility with old plotting code
        self.state = self.position[:2]  # [x, y] for 2D plots
        self.val = self.state
        
    def msg(self):
        """Broadcast current state to neighbors."""
        return (self.id, self.position.copy(), self.velocity.copy())
    
    def compute_desired_acceleration(self):
        """
        Compute nominal acceleration from formation control law.
        Control Law: PD control to desired position with saturation
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
        
        # Compute PD control law
        acc_desired = self.Kp * pos_error + self.Kd * vel_error
        
        # Saturate acceleration to physical limits (even without CBF)
        acc_norm = np.linalg.norm(acc_desired)
        if acc_norm > self.max_acceleration:
            acc_desired = (acc_desired / acc_norm) * self.max_acceleration
        
        return acc_desired
    
    def update_dynamics(self, acc_safe):
        """Update position and velocity using safe acceleration from CBF."""
        # Store current acceleration
        self.acceleration = acc_safe.copy()
        
        # Integrate acceleration → velocity
        self.velocity = self.velocity + self.acceleration * self.dt
        
        # Saturate velocity to physical limits
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > self.max_velocity:
            self.velocity = (self.velocity / vel_norm) * self.max_velocity
        
        # Integrate velocity → position
        self.position = self.position + self.velocity * self.dt
        
        # Update compatibility fields
        self.state = self.position[:2]
        self.val = self.state
        
        # Store history
        self.position_hist.append(self.position.copy())
        self.velocity_hist.append(self.velocity.copy())
        self.acceleration_hist.append(self.acceleration.copy())
    
    def get_formation_error(self):
        """Compute distance from ideal formation position."""
        n_agents = 5
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        ideal_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0
        ])
        
        return np.linalg.norm(self.position - ideal_pos)
    
    # Legacy methods for compatibility
    def stf(self):
        """Legacy: State Transition Function."""
        acc_desired = self.compute_desired_acceleration()
        self.update_dynamics(acc_desired)
    
    def clear_msgs(self):
        """Clear message buffer for next iteration."""
        self.msgs = []
    
    def add_msg(self, msg):
        """Receive message from neighbor."""
        self.msgs.append(msg)
    
    def ctl(self):
        """Required by DynamicAgent interface."""
        return self.position
    
    def step(self):
        """Required by DynamicAgent interface."""
        pass

# =============== Utility Visualization Functions ===============

def plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT):
    """Plot the evolution of pairwise barrier functions over time for a specific agent."""
    import matplotlib.pyplot as plt
    n = len(agents)
    T = len(agents[0].position_hist)
    time = np.arange(T) * agents[0].dt

    plt.figure(figsize=(10,6))
    
    if focus_agent >= 0 and focus_agent < n:
        # Show only barriers for the focused agent
        for j in range(n):
            if j != focus_agent:
                h_vals = [
                    np.linalg.norm(agents[focus_agent].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{focus_agent},{j}", linewidth=2)
        plt.title(f'GCBF Safety Functions for Agent {focus_agent} Over Time', fontsize=14)
    else:
        # Show all barriers
        for i in range(n):
            for j in range(i+1, n):
                h_vals = [
                    np.linalg.norm(agents[i].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{i},{j}")
        plt.title('All Pairwise GCBF Safety Functions Over Time', fontsize=14)
    
    plt.axhline(0, color='k', linestyle='--', label='Safety boundary', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Barrier Function $h_{ij}(t)$', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==================== SIMULATION FUNCTION ====================

def run_formation_with_cbf(n_agents=5, max_iter=500, dt=0.1, use_cbf=True, animate=True):
    """
    Run formation control simulation with optional CBF safety filter.
    
    Args:
        n_agents: Number of drones (default 5 for pentagon)
        max_iter: Maximum number of simulation steps
        dt: Timestep in seconds
        use_cbf: Whether to use CBF safety filter
        animate: Whether to animate the simulation
    
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
    print(f"Timestep: {dt}s, Max Duration: {max_iter*dt:.1f}s")
    print(f"Controller: PD with Kp=0.5, Kd=1.2 (tuned to reduce overshoot)")
    print(f"Saturation: max_vel=3.0 m/s, max_acc=2.0 m/s²")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD:.2f}m")
    print(f"Focus Agent for GCBF: {FOCUS_AGENT if FOCUS_AGENT >= 0 else 'All'}")
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
            Kp=0.5,  # Reduced from 1.0 to prevent overshoot
            Kd=1.2,  # Increased from 0.5 for better damping
            dt=dt,
            max_velocity=3.0,  # Add velocity saturation
            max_acceleration=2.0  # Add acceleration saturation
        ))
        print(f"Agent {i} initialized at: {init_pos} with zero initial velocity")
    
    # ==================== ANIMATION SETUP ====================
    if animate:
        plt.ion()  # Turn on interactive mode
        
        if VIS_2D_TOPVIEW:
            # Create figure with two subplots: 3D view and 2D top view
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(121, projection='3d')
            ax2d = fig.add_subplot(122)  # 2D top-down view
        else:
            # Single 3D view
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax2d = None
        
        # Generate colors for each drone
        colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
        
        # ========== 3D VIEW SETUP ==========
        # Initialize scatter plot for drones
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
                      edgecolors='orange', linewidths=2)
        
        # Draw formation circle
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = target_pos[0] + formation_radius * np.cos(theta)
        circle_y = target_pos[1] + formation_radius * np.sin(theta)
        circle_z = np.ones_like(theta) * target_pos[2]
        ax.plot(circle_x, circle_y, circle_z, 'g--', alpha=0.3, linewidth=2)
        
        # Initialize trajectory lines for each drone
        trajectory_lines = []
        for i in range(n_agents):
            line, = ax.plot([], [], [], color=colors[i], linewidth=1.5, 
                          alpha=0.6, label=f'Drone {i}')
            trajectory_lines.append(line)
        
        # Initialize velocity vectors and graph edges
        quiver = None
        graph_lines = []
        
        # Set 3D labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D Formation Control', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(0, 5)
        ax.grid(True, alpha=0.3)
        
        # ========== 2D TOP VIEW SETUP ==========
        if VIS_2D_TOPVIEW and ax2d is not None:
            # Initialize 2D drone positions
            # Initialize with agents' starting positions
            x_init = [a.position[0] for a in agents]
            y_init = [a.position[1] for a in agents]

            drone_scatter_2d = ax2d.scatter(
                x_init, y_init, s=400, c=colors, marker='o',
                edgecolor='black', linewidths=1.5
            )
                        
            # Plot target in 2D
            ax2d.scatter(target_pos[0], target_pos[1], 
                       c='red', marker='X', s=500, edgecolors='black', linewidths=2, zorder=5)
            
            # Plot ideal positions in 2D with agent labels
            for i in range(n_agents):
                angle = (2 * np.pi * i) / n_agents
                ideal_x = target_pos[0] + formation_radius * np.cos(angle)
                ideal_y = target_pos[1] + formation_radius * np.sin(angle)
                ax2d.scatter(ideal_x, ideal_y, c='green', marker='x', s=300, 
                           alpha=0.5, edgecolors='orange', linewidths=3, zorder=4)
                ax2d.text(ideal_x, ideal_y + 0.7, f'Pos{i}', fontsize=9, 
                        ha='center', color='orange', weight='bold')
            
            # Draw formation circle in 2D
            ax2d.plot(circle_x, circle_y, 'g--', alpha=0.3, linewidth=2, zorder=1)
            
            # Initialize sensing and safety circles (will be updated in loop)
            sensing_circle = None
            safety_circle = None
            safety_circle_edge = None
            
            # Initialize 2D trajectory lines
            trajectory_lines_2d = []
            for i in range(n_agents):
                line, = ax2d.plot([], [], color=colors[i], linewidth=2, 
                                alpha=0.4, zorder=2)
                trajectory_lines_2d.append(line)
            
            # Initialize 2D graph edges
            graph_lines_2d = []
            
            # Add drone ID labels (will be updated with positions)
            drone_labels_2d = []
            for i in range(n_agents):
                label = ax2d.text(0, 0, f'D{i}', fontsize=10, ha='center', va='center',
                                 color='white', weight='bold', zorder=15)
                drone_labels_2d.append(label)
            
            ax2d.set_xlabel('X (m)', fontsize=12)
            ax2d.set_ylabel('Y (m)', fontsize=12)
            ax2d.set_title(f'2D Top View (GCBF Focus: Agent {FOCUS_AGENT if FOCUS_AGENT >= 0 else "All"})', 
                         fontsize=14, fontweight='bold')
            ax2d.axis('equal')
            ax2d.grid(True, alpha=0.3)
            ax2d.set_xlim(-15, 15)
            ax2d.set_ylim(-15, 15)
        
        # Add text annotation for live statistics
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # ==================== END ANIMATION SETUP ====================
    
    # Main simulation loop
    print(f"\nStarting simulation...")
    converged = False
    convergence_count = 0
    convergence_patience = 20  # Must be converged for this many iterations
    
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
                obstacles=None
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
        
        # ==================== CONVERGENCE CHECK ====================
        formation_errors = [agent.get_formation_error() for agent in agents]
        avg_error = np.mean(formation_errors)
        max_error = np.max(formation_errors)
        avg_vel = np.mean([np.linalg.norm(a.velocity) for a in agents])
        max_vel = np.max([np.linalg.norm(a.velocity) for a in agents])
        
        # Check if converged
        if max_error < CONVERGENCE_THRESHOLD and max_vel < CONVERGENCE_VELOCITY:
            convergence_count += 1
            if convergence_count >= convergence_patience:
                converged = True
                print(f"\n✅ FORMATION CONVERGED at iteration {iteration}")
                print(f"   Max error: {max_error:.3f}m < {CONVERGENCE_THRESHOLD}m")
                print(f"   Max velocity: {max_vel:.3f}m/s < {CONVERGENCE_VELOCITY}m/s")
                if not animate:
                    break  # Exit if not animating
        else:
            convergence_count = 0  # Reset counter if not converged
        
        # ==================== ANIMATION UPDATE ====================
        if animate and iteration % 2 == 0:
            # Update drone positions
            current_positions = np.array([agent.position for agent in agents])
            drone_scatter._offsets3d = (current_positions[:, 0], 
                                        current_positions[:, 1], 
                                        current_positions[:, 2])
            
            # Color drones by their ID
            drone_scatter.set_color(colors)
            
            # Update trajectories
            for i, agent in enumerate(agents):
                traj = np.array(agent.position_hist)
                trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                trajectory_lines[i].set_3d_properties(traj[:, 2])
            
            # Update velocity vectors
            if quiver is not None:
                quiver.remove()
            
            current_velocities = np.array([agent.velocity for agent in agents])
            vel_scale = 0.5
            quiver = ax.quiver(current_positions[:, 0], current_positions[:, 1], current_positions[:, 2],
                              current_velocities[:, 0]*vel_scale, 
                              current_velocities[:, 1]*vel_scale, 
                              current_velocities[:, 2]*vel_scale,
                              color='blue', alpha=0.6, arrow_length_ratio=0.3, linewidths=1.5)
            
            # ============ 3D GRAPH OVERLAY VISUALIZATION ============
            if VIS_GRAPH_OVERLAY and use_cbf:
                # Remove old graph lines
                for line in graph_lines:
                    line.remove()
                graph_lines.clear()
                
                # Draw edges based on sensing radius
                if FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                    # Show only edges connected to focused agent
                    for j in range(n_agents):
                        if j != FOCUS_AGENT:
                            dist = np.linalg.norm(current_positions[FOCUS_AGENT] - current_positions[j])
                            if dist < cbf_filter.R_sense:
                                # Color based on safety status
                                if dist < cbf_filter.d_safe * 1.1:
                                    color = 'red'  # Too close!
                                    linewidth = 3.0
                                    alpha = 0.9
                                elif dist < cbf_filter.d_safe * 1.5:
                                    color = 'orange'  # Getting close
                                    linewidth = 2.5
                                    alpha = 0.7
                                else:
                                    color = 'cyan'  # Normal neighbor for focused agent
                                    linewidth = 2.0
                                    alpha = 0.5
                                
                                # Draw edge in 3D
                                line = ax.plot([current_positions[FOCUS_AGENT,0], current_positions[j,0]],
                                             [current_positions[FOCUS_AGENT,1], current_positions[j,1]],
                                             [current_positions[FOCUS_AGENT,2], current_positions[j,2]],
                                             color=color, linewidth=linewidth, alpha=alpha)[0]
                                graph_lines.append(line)
                else:
                    # Show all edges
                    for i in range(n_agents):
                        for j in range(i+1, n_agents):
                            dist = np.linalg.norm(current_positions[i] - current_positions[j])
                            if dist < cbf_filter.R_sense:
                                if dist < cbf_filter.d_safe * 1.1:
                                    color = 'red'
                                    linewidth = 2.5
                                    alpha = 0.8
                                elif dist < cbf_filter.d_safe * 1.5:
                                    color = 'orange'
                                    linewidth = 2.0
                                    alpha = 0.6
                                else:
                                    color = 'gray'
                                    linewidth = 1.0
                                    alpha = 0.3
                                
                                line = ax.plot([current_positions[i,0], current_positions[j,0]],
                                             [current_positions[i,1], current_positions[j,1]],
                                             [current_positions[i,2], current_positions[j,2]],
                                             color=color, linewidth=linewidth, alpha=alpha)[0]
                                graph_lines.append(line)
            
            # ============ 2D TOP VIEW UPDATE ============
            if VIS_2D_TOPVIEW and ax2d is not None:
                # Update 2D drone positions
                drone_scatter_2d.set_offsets(current_positions[:, :2])
                
                # Update drone labels
                for i in range(n_agents):
                    drone_labels_2d[i].set_position((current_positions[i, 0], current_positions[i, 1]))
                
                # Update 2D trajectories
                for i, agent in enumerate(agents):
                    traj = np.array(agent.position_hist)
                    trajectory_lines_2d[i].set_data(traj[:, 0], traj[:, 1])
                
                # Update sensing/safety circles for focused agent
                if use_cbf and FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                    # Remove old circles
                    for patch in list(ax2d.patches):
                        if isinstance(patch, plt.Circle):
                            patch.remove()
                    
                    agent_pos = current_positions[FOCUS_AGENT]
                    
                    # Add sensing circle (blue dotted)
                    sensing_circle = plt.Circle((agent_pos[0], agent_pos[1]), 
                                              cbf_filter.R_sense, 
                                              fill=False, color='blue', 
                                              linestyle=':', linewidth=2, alpha=0.4,
                                              label='Sensing Range', zorder=3)
                    ax2d.add_patch(sensing_circle)
                    
                    # Add safety zone (red filled)
                    safety_circle = plt.Circle((agent_pos[0], agent_pos[1]), 
                                             cbf_filter.d_safe, 
                                             fill=True, color='red', 
                                             alpha=0.15, zorder=2)
                    ax2d.add_patch(safety_circle)
                    
                    # Add safety zone edge (red dashed)
                    safety_circle_edge = plt.Circle((agent_pos[0], agent_pos[1]), 
                                                   cbf_filter.d_safe, 
                                                   fill=False, color='red', 
                                                   linestyle='--', linewidth=2.5, alpha=0.7,
                                                   label='Safety Zone', zorder=3)
                    ax2d.add_patch(safety_circle_edge)
                
                # Draw graph edges in 2D
                for line in graph_lines_2d:
                    line.remove()
                graph_lines_2d.clear()
                
                if VIS_GRAPH_OVERLAY and use_cbf:
                    if FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                        # Show only edges for focused agent
                        for j in range(n_agents):
                            if j != FOCUS_AGENT:
                                dist = np.linalg.norm(current_positions[FOCUS_AGENT] - current_positions[j])
                                if dist < cbf_filter.R_sense:
                                    if dist < cbf_filter.d_safe * 1.1:
                                        color = 'red'
                                        linewidth = 3.0
                                        alpha = 0.9
                                    elif dist < cbf_filter.d_safe * 1.5:
                                        color = 'orange'
                                        linewidth = 2.5
                                        alpha = 0.7
                                    else:
                                        color = 'cyan'
                                        linewidth = 2.0
                                        alpha = 0.5
                                    
                                    line, = ax2d.plot([current_positions[FOCUS_AGENT,0], current_positions[j,0]],
                                                    [current_positions[FOCUS_AGENT,1], current_positions[j,1]],
                                                    color=color, linewidth=linewidth, alpha=alpha, zorder=6)
                                    graph_lines_2d.append(line)
                    else:
                        # Show all edges
                        for i in range(n_agents):
                            for j in range(i+1, n_agents):
                                dist = np.linalg.norm(current_positions[i] - current_positions[j])
                                if dist < cbf_filter.R_sense:
                                    if dist < cbf_filter.d_safe * 1.1:
                                        color = 'red'
                                        linewidth = 2.5
                                        alpha = 0.8
                                    elif dist < cbf_filter.d_safe * 1.5:
                                        color = 'orange'
                                        linewidth = 2.0
                                        alpha = 0.6
                                    else:
                                        color = 'gray'
                                        linewidth = 1.0
                                        alpha = 0.3
                                    
                                    line, = ax2d.plot([current_positions[i,0], current_positions[j,0]],
                                                    [current_positions[i,1], current_positions[j,1]],
                                                    color=color, linewidth=linewidth, alpha=alpha, zorder=6)
                                    graph_lines_2d.append(line)
                
                # Update 2D view limits dynamically
                all_x = current_positions[:, 0]
                all_y = current_positions[:, 1]
                margin = 5.0
                ax2d.set_xlim(min(all_x.min()-margin, -15), max(all_x.max()+margin, 15))
                ax2d.set_ylim(min(all_y.min()-margin, -15), max(all_y.max()+margin, 15))
            
            # Update info text
            avg_acc = np.mean([np.linalg.norm(a.acceleration) for a in agents])
            info_text.set_text(
                f"Iteration: {iteration:04d}\n"
                f"Avg error: {avg_error:.3f} m\n"
                f"Max error: {max_error:.3f} m\n"
                f"Avg vel: {avg_vel:.3f} m/s\n"
                f"Max vel: {max_vel:.3f} m/s\n"
                f"Avg acc: {avg_acc:.3f} m/s²"
            )
            
            # Pause briefly for animation smoothness
            plt.pause(0.001)
            
            # Check minimum distance between drones
            min_dist = float('inf')
            for i in range(n_agents):
                for j in range(i+1, n_agents):
                    dist = np.linalg.norm(agents[i].position - agents[j].position)
                    min_dist = min(min_dist, dist)
            
            status = " CONVERGED" if converged else " CONVERGING..."
            info_str = (f"{status}\n"
                       f"Iteration: {iteration}/{max_iter}\n"
                       f"Time: {iteration*dt:.1f}s / {max_iter*dt:.1f}s\n"
                       f"Formation Error:\n"
                       f"  Avg: {avg_error:.3f}m\n"
                       f"  Max: {max_error:.3f}m (goal < {CONVERGENCE_THRESHOLD}m)\n"
                       f"Velocity:\n"
                       f"  Avg: {avg_vel:.3f}m/s\n"
                       f"  Max: {max_vel:.3f}m/s (goal < {CONVERGENCE_VELOCITY}m/s)\n"
                       f"Avg Acceleration: {avg_acc:.3f}m/s²\n"
                       f"Min Distance: {min_dist:.3f}m")
            
            if use_cbf:
                stats = cbf_filter.get_statistics()
                if stats.get('total_solves', 0) > 0:
                    activation_rate = stats.get('activation_rate', 0)
                    info_str += f"\nCBF Active: {activation_rate*100:.1f}%"
                    if cbf_filter.d_safe:
                        info_str += f"\nSafety Distance: {cbf_filter.d_safe:.2f}m"
            
            # Add graph connectivity info
            if VIS_GRAPH_OVERLAY and use_cbf:
                n_edges = len(graph_lines)
                info_str += f"\nGraph Edges: {n_edges}"
            
            # info_text.set_text(info_str)
            
            # Dynamically adjust 3D view limits
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
            
            # Stop animation if converged
            if converged:
                plt.pause(1.0)  # Pause to show converged state
                break
        # ==================== END ANIMATION UPDATE ====================
        
        # Step 5: Progress reporting
        if iteration % 50 == 0:
            print(f"Iter {iteration:3d}: "
                  f"Formation error: avg={avg_error:.3f}m, max={max_error:.3f}m, "
                  f"Avg vel: {avg_vel:.3f}m/s, Avg acc: {avg_acc:.3f}m/s²")
        
        # Exit loop if converged (for non-animated case)
        if converged and not animate:
            break
    
    # Optional barrier function visualization (only if CBF filter is active)
    if use_cbf and cbf_filter is not None and VIS_BARRIERS:
        plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT)
    
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
    
    # Check final velocities
    final_velocities = [np.linalg.norm(agent.velocity) for agent in agents]
    print(f"\nFinal Velocities:")
    for i, vel in enumerate(final_velocities):
        print(f"  Agent {i}: {vel:.3f}m/s")
    print(f"  Average: {np.mean(final_velocities):.3f}m/s")
    
    # Report convergence
    if converged:
        actual_iter = len(agents[0].position_hist) - 1
        print(f"\n✅ Formation converged after {actual_iter} iterations ({actual_iter*dt:.1f}s)")
    else:
        print(f"\n⚠️  Formation did not converge within {max_iter} iterations")
    
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
    
    # Add convergence threshold line
    ax3.axhline(y=CONVERGENCE_THRESHOLD, color='r', linestyle=':', 
               linewidth=2, label=f'Convergence Threshold ({CONVERGENCE_THRESHOLD}m)')
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Formation Error (m)', fontsize=10)
    ax3.set_title('Formation Error Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_with_without_cbf():
    """Run two simulations side-by-side to compare CBF vs no CBF."""
    print("\n" + "="*60)
    print("COMPARISON: WITH vs WITHOUT CBF")
    print("="*60)
    
    # Run without CBF
    print("\n--- Running WITHOUT CBF ---")
    agents_no_cbf, _ = run_formation_with_cbf(
        n_agents=5, max_iter=1000, dt=0.1, use_cbf=False, animate=False
    )
    
    # Run with CBF
    print("\n--- Running WITH CBF ---")
    agents_with_cbf, cbf_stats = run_formation_with_cbf(
        n_agents=5, max_iter=1000, dt=0.1, use_cbf=True, animate=False
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
    
    # Convergence time
    print("\nConvergence Time:")
    for name, agents in [("Without CBF", agents_no_cbf), 
                         ("With CBF", agents_with_cbf)]:
        converged_iter = len(agents[0].position_hist) - 1
        print(f"  {name}: {converged_iter * agents[0].dt:.1f}s ({converged_iter} iterations)")
    
    return agents_no_cbf, agents_with_cbf


# ==================== MAIN ====================

if __name__ == "__main__":
    # Option 1: Run with CBF and all visualizations
    agents, cbf_stats = run_formation_with_cbf(
        n_agents=5,
        max_iter=2000,  # Increased to allow convergence detection
        dt=0.05,
        use_cbf=True,  
        animate=True
    )
    
    # Visualize results
    plot_results_3d(agents)
    
    # Option 2: Compare with and without CBF
    # agents_no_cbf, agents_with_cbf = compare_with_without_cbf()
    # plot_results_3d(agents_with_cbf)