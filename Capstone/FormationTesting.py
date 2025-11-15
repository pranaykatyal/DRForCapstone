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
FOCUS_AGENT = 0  # Default to Agent 0 for GCBF visualization
CONVERGENCE_THRESHOLD = 0.05  # Stop when all agents within 5cm of ideal positions
CONVERGENCE_VELOCITY = 0.01  # And velocity below 1cm/s
NUM_AGENTS = 5 # min 3 
N_OBSTACLES = 3 
# Target and Formation Parameters
TARGET_TIME = 15.0  # Time for target to complete trajectory (seconds)
FORMATION_RADIUS = 5.0  # Formation radius in meters

# =============== Moving Target Class ===============

class MovingTarget:
    """Moving target that drones track while maintaining formation."""
    
    def __init__(self, start_pos, end_pos, duration, dt=0.1):
        """
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            duration: Time to complete trajectory (seconds)
            dt: Timestep
        """
        self.position = np.array(start_pos, dtype=float)
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.duration = duration
        self.dt = dt
        self.time = 0.0
        
        # Constant velocity motion
        self.velocity = (self.end_pos - self.start_pos) / self.duration
        
        # History
        self.position_hist = [self.position.copy()]
        self.velocity_hist = [self.velocity.copy()]
    
    def update(self):
        """Update target position."""
        self.time += self.dt
        
        if self.time * np.linalg.norm(self.velocity) < np.linalg.norm(self.end_pos - self.start_pos):
            self.position = self.position + self.velocity * self.dt
        else:
            # Reached end, stop moving
            self.velocity = np.zeros(3)
            self.position = self.end_pos.copy()
        
        self.position_hist.append(self.position.copy())
        self.velocity_hist.append(self.velocity.copy())

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
        Control Law: Track target while maintaining safe formation spacing
        """
        # Compute assigned position in formation (pentagon)
        n_agents = NUM_AGENTS
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        # Formation in XY plane at target Z height
        desired_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0  # Formation stays in horizontal plane
        ])
        
        # Adaptive weighting based on distance to center
        formation_center = self.target_pos
        dist_to_center = np.linalg.norm(self.position - formation_center)
        
        # Scale thresholds relative to formation_radius
        far_threshold = 3.0 * self.formation_radius      # 15m for 5m radius
        approaching_threshold = 2.0 * self.formation_radius  # 10m
        near_threshold = 1.2 * self.formation_radius     # 6m (just outside formation)
        
        if dist_to_center > far_threshold:
            # Very far: chase aggressively
            center_weight = 0.9
            formation_weight = 0.1
        elif dist_to_center > approaching_threshold:
            # Far: mostly chase
            center_weight = 0.7
            formation_weight = 0.3
        elif dist_to_center > near_threshold:
            # Approaching: balanced
            center_weight = 0.5
            formation_weight = 0.5
        else:
            # **FIXED**: When close, ONLY track formation position
            center_weight = 0.0
            formation_weight = 1.0
        
        center_error = formation_center - self.position
        formation_error = desired_pos - self.position
        
        combined_error = center_weight * center_error + formation_weight * formation_error
        
        # PD control: acc = Kp*(pos_error) + Kd*(vel_error)
        vel_error = -self.velocity  # Desired velocity = 0 (hover in formation)
        
        # Compute PD control law
        acc_desired = self.Kp * combined_error + self.Kd * vel_error
        
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
        n_agents = NUM_AGENTS
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
# ==================== OBSTACLE GENERATION ====================

def generate_random_spherical_obstacles(
        n_obstacles=5,
        target_start=np.array([0.0, 0.0, 0.0]),
        target_end=np.array([50.0, 50.0, 10.0]),
        min_radius=2.0,
        max_radius=6.0):
    """
    Generate random spherical obstacles along the target's path.

    Obstacles appear near the line segment from start→end
    so that drones MUST detour around them.

    Returns:
        List of dicts: {"center": np.array([x,y,z]), "radius": r}
    """
    obstacles = []

    for _ in range(n_obstacles):
        # Random interpolation along target path
        t = np.random.uniform(0.1, 0.9)
        base = target_start + t * (target_end - target_start)

        # Random sideways offset (perpendicular plane)
        offset = np.random.uniform(-10, 10, 3)
        offset[2] = np.random.uniform(-2, 2)    # vertical spread

        center = base + offset
        radius = np.random.uniform(min_radius, max_radius)

        obstacles.append({
            "center": center,
            "radius": radius
        })

    return obstacles

# ==================== SIMULATION FUNCTION ====================

def run_formation_with_cbf(n_agents=NUM_AGENTS, max_iter=500, dt=0.1, use_cbf=True, animate=True,
                           moving_target=True):
    """
    Run formation control simulation with optional CBF safety filter.
    
    Args:
        n_agents: Number of drones (default 5 for pentagon)
        max_iter: Maximum number of simulation steps
        dt: Timestep in seconds
        use_cbf: Whether to use CBF safety filter
        animate: Whether to animate the simulation
        moving_target: Whether target should move
    
    Returns:
        agents: List of DroneAgent objects with trajectory history
        target: MovingTarget object (or None if static)
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
            safety_distance=2.0,      # Minimum 1.5m between drones
            sensing_radius=8.0,       # Detect neighbors within 8m
            alpha1=3.0,               # CBF responsiveness
            alpha2=2.5,               # CBF convergence rate
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
    
    # Initialize target
    formation_radius = FORMATION_RADIUS  # Use global parameter
    
    if moving_target:
        target = MovingTarget(
            start_pos=np.array([0.0, 0.0, 0.0]), # Start position
            end_pos=np.array([25.0, 25.0, 5.0]), # Goal position
            duration=TARGET_TIME,  # Use global parameter
            dt=dt
        )
        print(f"Moving target: {target.start_pos} → {target.end_pos} over {target.duration}s")
    else:
        target = None
        target_pos = np.array([0.0, 0.0, 2.0])
    
    agents = []
    np.random.seed(42)  # Reproducibility
    
    print(f"\n{'='*60}")
    print(f"Formation Control Simulation")
    print(f"{'='*60}")
    print(f"Agents: {n_agents}")
    print(f"Target: {'MOVING' if moving_target else 'STATIC'}")
    print(f"Formation radius: {formation_radius}m")
    print(f"CBF enabled: {use_cbf}")
    print(f"Timestep: {dt}s, Max Duration: {max_iter*dt:.1f}s")
    print(f"Controller: PD with Kp=0.5, Kd=1.2 (tuned to reduce overshoot)")
    print(f"Saturation: max_vel=5.0 m/s, max_acc=4.0 m/s²")
    print(f"Target Sensing: 15.0m direct, consensus with neighbors beyond")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD:.2f}m")
    print(f"Focus Agent for GCBF: {FOCUS_AGENT if FOCUS_AGENT >= 0 else 'All'}")
    print(f"{'='*60}\n")
    
    for i in range(n_agents):
        # Random initial position
        init_pos = np.random.uniform(-10, 10, 3)
        init_pos[2] = np.random.uniform(1, 3)  # Reasonable altitude
        
        current_target_pos = target.position if moving_target else target_pos
        
        agents.append(DroneAgent(
            id=i,
            state_3d=init_pos,
            target_pos_3d=current_target_pos,
            formation_radius=formation_radius,
            Kp=0.5,
            Kd=1.2,
            dt=dt,
            max_velocity=5.0,  # Increased from 3.0 to 5.0 m/s
            max_acceleration=4.0  # Increased from 2.0 to 4.0 m/s²
        ))
        print(f"Agent {i} initialized at: {init_pos} with zero initial velocity")
    
    if moving_target:
        obstacles = generate_random_spherical_obstacles(
            n_obstacles=N_OBSTACLES,
            target_start=target.start_pos,
            target_end=target.end_pos,
            min_radius=2.0,
            max_radius=6.0
        )
    else:
        # Static target fallback
        obstacles = generate_random_spherical_obstacles(
            n_obstacles=N_OBSTACLES,
            target_start=np.array([0, 0, 0]),
            target_end=np.array([0, 0, 2]),
            min_radius=2.0,
            max_radius=6.0
        )

    print(f"\nGenerated {len(obstacles)} spherical obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  Obs {i}: center={obs['center']}, radius={obs['radius']:.2f}")
    
    
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
        current_target_pos = target.position if moving_target else target_pos
        target_scatter = ax.scatter(current_target_pos[0], current_target_pos[1], current_target_pos[2], 
                  c='red', marker='X', s=500, edgecolors='black', linewidths=2,
                  label='Target')
        
        # Target trajectory line
        if moving_target:
            target_traj_line, = ax.plot([], [], [], 'r--', linewidth=2, alpha=0.5, label='Target Path')
        
        # Plot ideal formation positions (green X's)
        ideal_markers = []
        for i in range(n_agents):
            angle = (2 * np.pi * i) / n_agents
            ideal_pos = current_target_pos + formation_radius * np.array([
                np.cos(angle), np.sin(angle), 0.0
            ])
            marker = ax.scatter(ideal_pos[0], ideal_pos[1], ideal_pos[2],
                      c='green', marker='x', s=200, alpha=0.5,
                      edgecolors='orange', linewidths=2)
            ideal_markers.append(marker)
        
        # ========== OBSTACLE VISUALIZATION ==========
        obstacle_scatters = []
        for obs in obstacles:
            sx, sy, sz = obs["center"]
            # Plot spherical obstacle center
            sc = ax.scatter(sx, sy, sz,
                c="purple", s=150, marker="o", alpha=0.7,
                edgecolors="black", linewidths=2)
            obstacle_scatters.append(sc)

            # Draw 3D wireframe sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = obs["radius"] * np.outer(np.cos(u), np.sin(v)) + sx
            y = obs["radius"] * np.outer(np.sin(u), np.sin(v)) + sy
            z = obs["radius"] * np.outer(np.ones(np.size(u)), np.cos(v)) + sz
            ax.plot_wireframe(x, y, z, color='purple', alpha=0.3, linewidth=1)

        
        # Draw formation circle
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = current_target_pos[0] + formation_radius * np.cos(theta)
        circle_y = current_target_pos[1] + formation_radius * np.sin(theta)
        circle_z = np.ones_like(theta) * current_target_pos[2]
        formation_circle, = ax.plot(circle_x, circle_y, circle_z, 'g--', alpha=0.3, linewidth=2)
        
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
            x_init = [a.position[0] for a in agents]
            y_init = [a.position[1] for a in agents]

            drone_scatter_2d = ax2d.scatter(
                x_init, y_init, s=400, c=colors, marker='o',
                edgecolor='black', linewidths=1.5
            )
                        
            # Plot target in 2D
            target_scatter_2d = ax2d.scatter(current_target_pos[0], current_target_pos[1], 
                       c='red', marker='X', s=500, edgecolors='black', linewidths=2, zorder=5)
            
            if moving_target:
                target_traj_line_2d, = ax2d.plot([], [], 'r--', linewidth=2, alpha=0.5, zorder=1)
            
            # Plot ideal positions in 2D with agent labels
            ideal_markers_2d = []
            for i in range(n_agents):
                angle = (2 * np.pi * i) / n_agents
                ideal_x = current_target_pos[0] + formation_radius * np.cos(angle)
                ideal_y = current_target_pos[1] + formation_radius * np.sin(angle)
                marker = ax2d.scatter(ideal_x, ideal_y, c='green', marker='x', s=300, 
                           alpha=0.5, edgecolors='orange', linewidths=3, zorder=4)
                ideal_markers_2d.append(marker)
                ax2d.text(ideal_x, ideal_y + 0.7, f'Pos{i}', fontsize=9, 
                        ha='center', color='orange', weight='bold')
            
            # 2D top view obstacles
            obstacle_scatters_2d = []
            for obs in obstacles:
                sx, sy, sz = obs["center"]
                sc = ax2d.scatter(sx, sy,
                    c="purple", s=350, marker="o",
                    alpha=0.4, edgecolors="black", linewidths=2)
                obstacle_scatters_2d.append(sc)

                # Draw 2D circle (top view projection)
                theta_2d = np.linspace(0, 2*np.pi, 40)
                circ_x = sx + obs["radius"] * np.cos(theta_2d)
                circ_y = sy + obs["radius"] * np.sin(theta_2d)
                ax2d.plot(circ_x, circ_y,
                    color="purple", linestyle="--", alpha=0.6, linewidth=2)

            
            # Draw formation circle in 2D
            formation_circle_2d, = ax2d.plot(circle_x, circle_y, 'g--', alpha=0.3, linewidth=2, zorder=1)
            
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
        # Step 0: Update moving target
        if moving_target:
            target.update()
            
            # Each agent updates target estimate based on sensing + consensus
            for agent in agents:
                # Check if target is within sensing range
                dist_to_target = np.linalg.norm(agent.position - target.position)
                target_sensing_range = 15.0  # Can directly sense target within 15m
                
                if dist_to_target < target_sensing_range:
                    # Direct sensing: update to true target position
                    agent.target_pos = target.position.copy()
                else:
                    # Out of range: use consensus with neighbors
                    # Collect target estimates from neighbors within communication range
                    neighbor_estimates = []
                    for other_agent in agents:
                        if other_agent.id != agent.id:
                            dist_to_neighbor = np.linalg.norm(agent.position - other_agent.position)
                            if dist_to_neighbor < 8.0:  # Within communication range
                                neighbor_estimates.append(other_agent.target_pos)
                    
                    if len(neighbor_estimates) > 0:
                        # Average neighbor estimates (simple consensus)
                        consensus_estimate = np.mean(neighbor_estimates, axis=0)
                        # Blend own estimate with consensus (80% consensus, 20% own)
                        agent.target_pos = 0.8 * consensus_estimate + 0.2 * agent.target_pos
                    # else: keep last known target position
        
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
                obstacles=obstacles
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
        
        # Track formation center vs target error
        formation_center = np.mean([agent.position for agent in agents], axis=0)
        if moving_target:
            center_to_target_error = np.linalg.norm(formation_center - target.position)
        else:
            center_to_target_error = np.linalg.norm(formation_center - target_pos)
        
        # Store for plotting later
        if not hasattr(agents[0], 'center_error_hist'):
            for agent in agents:
                agent.center_error_hist = []
        for agent in agents:
            agent.center_error_hist.append(center_to_target_error)
        
        # For moving target: only check convergence after target has stopped
        target_stopped = False
        if moving_target:
            target_stopped = (np.linalg.norm(target.velocity) < 0.01)
        else:
            target_stopped = True  # Static target is always "stopped"
        
        # Check if converged (only after target stops)
        if target_stopped and max_error < CONVERGENCE_THRESHOLD and max_vel < CONVERGENCE_VELOCITY:
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
            
            # Update target
            if moving_target:
                current_target_pos = target.position
                target_scatter._offsets3d = ([current_target_pos[0]], [current_target_pos[1]], [current_target_pos[2]])
                
                # Update target trajectory
                traj_target = np.array(target.position_hist)
                target_traj_line.set_data(traj_target[:, 0], traj_target[:, 1])
                target_traj_line.set_3d_properties(traj_target[:, 2])
                
                # Update ideal positions
                for i, marker in enumerate(ideal_markers):
                    angle = (2 * np.pi * i) / n_agents
                    ideal_pos = current_target_pos + formation_radius * np.array([np.cos(angle), np.sin(angle), 0.0])
                    marker._offsets3d = ([ideal_pos[0]], [ideal_pos[1]], [ideal_pos[2]])
                
                # Update formation circle
                circle_x = current_target_pos[0] + formation_radius * np.cos(theta)
                circle_y = current_target_pos[1] + formation_radius * np.sin(theta)
                circle_z = np.ones_like(theta) * current_target_pos[2]
                formation_circle.set_data(circle_x, circle_y)
                formation_circle.set_3d_properties(circle_z)
            
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
                
                # Update target in 2D
                if moving_target:
                    target_scatter_2d.set_offsets([[current_target_pos[0], current_target_pos[1]]])
                    target_traj_line_2d.set_data(traj_target[:, 0], traj_target[:, 1])
                    
                    # Update ideal positions in 2D
                    for i, marker in enumerate(ideal_markers_2d):
                        angle = (2 * np.pi * i) / n_agents
                        ideal_x = current_target_pos[0] + formation_radius * np.cos(angle)
                        ideal_y = current_target_pos[1] + formation_radius * np.sin(angle)
                        marker.set_offsets([[ideal_x, ideal_y]])
                    
                    # Update formation circle 2D
                    formation_circle_2d.set_data(circle_x, circle_y)
                
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
                if moving_target:
                    all_x = np.append(all_x, current_target_pos[0])
                    all_y = np.append(all_y, current_target_pos[1])
                
                # Keep view centered on action
                margin = 10.0  # Larger margin to see GCBF sensing circles
                x_center = np.mean(all_x)
                y_center = np.mean(all_y)
                x_range = max(15.0, (all_x.max() - all_x.min()) / 2 + margin)
                y_range = max(15.0, (all_y.max() - all_y.min()) / 2 + margin)
                
                ax2d.set_xlim(x_center - x_range, x_center + x_range)
                ax2d.set_ylim(y_center - y_range, y_center + y_range)
            
            # Update info text
            avg_acc = np.mean([np.linalg.norm(a.acceleration) for a in agents])
            info_text.set_text(
                f"Iteration: {iteration:04d}\n"
                f"Center→Target: {center_to_target_error:.3f} m\n"
                f"Avg form error: {avg_error:.3f} m\n"
                f"Max form error: {max_error:.3f} m\n"
                f"Avg vel: {avg_vel:.3f} m/s\n"
                f"Max vel: {max_vel:.3f} m/s\n"
                f"Avg acc: {avg_acc:.3f} m/s²"
            )
            
            # Pause briefly for animation smoothness
            plt.pause(0.001)
            
            # Dynamically adjust 3D view limits
            all_x = current_positions[:, 0]
            all_y = current_positions[:, 1]
            all_z = current_positions[:, 2]
            if moving_target:
                all_x = np.append(all_x, current_target_pos[0])
                all_y = np.append(all_y, current_target_pos[1])
                all_z = np.append(all_z, current_target_pos[2])
            
            # Keep view centered on drones with reasonable margin
            margin = 8.0  # Increased margin to see GCBF circles
            x_center = np.mean(all_x)
            y_center = np.mean(all_y)
            z_center = np.mean(all_z)
            x_range = max(15.0, (all_x.max() - all_x.min()) / 2 + margin)
            y_range = max(15.0, (all_y.max() - all_y.min()) / 2 + margin)
            z_range = max(8.0, (all_z.max() - all_z.min()) / 2 + margin)
            
            ax.set_xlim(x_center - x_range, x_center + x_range)
            ax.set_ylim(y_center - y_range, y_center + y_range)
            ax.set_zlim(max(0, z_center - z_range), z_center + z_range)
            
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
                  f"Center→Target: {center_to_target_error:.3f}m, "
                  f"Formation error: avg={avg_error:.3f}m, max={max_error:.3f}m, "
                  f"Avg vel: {avg_vel:.3f}m/s")
        
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
    return agents, target, cbf_stats


def plot_results_3d(agents, target=None):
    """
    Visualize 3D trajectories and final formation.
    
    Args:
        agents: List of DroneAgent objects with history
        target: MovingTarget object or None
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
    
    # Plot target trajectory
    if target is not None:
        traj_target = np.array(target.position_hist)
        ax1.plot(traj_target[:, 0], traj_target[:, 1], traj_target[:, 2],
                'r--', linewidth=3, alpha=0.6, label='Target')
        ax1.scatter(traj_target[0, 0], traj_target[0, 1], traj_target[0, 2],
                   color='red', marker='X', s=200, edgecolors='black')
        ax1.scatter(traj_target[-1, 0], traj_target[-1, 1], traj_target[-1, 2],
                   color='darkred', marker='X', s=300, edgecolors='black')
    
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
    target_pos = agents[0].target_pos
    circle_x = target_pos[0] + agents[0].formation_radius * np.cos(theta)
    circle_y = target_pos[1] + agents[0].formation_radius * np.sin(theta)
    ax2.plot(circle_x, circle_y, 'k--', alpha=0.3, linewidth=2)
    
    # Mark target
    ax2.plot(target_pos[0], target_pos[1], 'r+', markersize=20, markeredgewidth=3)
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('Final Formation (Top View)\no = actual, x = ideal', 
                 fontsize=12, fontweight='bold')
    ax2.axis('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Formation Center Error vs Target Over Time
    ax3 = fig.add_subplot(133)
    
    max_len = max(len(agent.position_hist) for agent in agents)
    time = np.arange(max_len) * agents[0].dt
    
    # Plot formation center to target error (most important metric)
    if hasattr(agents[0], 'center_error_hist'):
        center_errors = agents[0].center_error_hist
        ax3.plot(time[:len(center_errors)], center_errors, 'b-', 
                linewidth=3, label='Formation Center → Target', alpha=0.8)
    
    # Plot individual formation errors (less important)
    for i, agent in enumerate(agents):
        errors = [np.linalg.norm(np.array(agent.position_hist[j]) - 
                                 (agent.target_pos + agent.formation_radius * np.array([
                                     np.cos(2*np.pi*agent.id/NUM_AGENTS),
                                     np.sin(2*np.pi*agent.id/NUM_AGENTS),
                                     0.0
                                 ])))
                 for j in range(len(agent.position_hist))]
        ax3.plot(time[:len(errors)], errors, color=colors[i], 
                linewidth=1, alpha=0.3, label=f'Drone {agent.id} form error')
    
    # Average formation error
    avg_errors = []
    for t in range(max_len):
        errors_at_t = []
        for agent in agents:
            if t < len(agent.position_hist):
                ideal = agent.target_pos + agent.formation_radius * np.array([
                    np.cos(2*np.pi*agent.id/NUM_AGENTS),
                    np.sin(2*np.pi*agent.id/NUM_AGENTS),
                    0.0
                ])
                error = np.linalg.norm(agent.position_hist[t] - ideal)
                errors_at_t.append(error)
        if errors_at_t:
            avg_errors.append(np.mean(errors_at_t))
    
    ax3.plot(time[:len(avg_errors)], avg_errors, 'k--', 
            linewidth=2, label='Avg Formation Error', alpha=0.5)
    
    # Add convergence threshold line
    ax3.axhline(y=CONVERGENCE_THRESHOLD, color='r', linestyle=':', 
               linewidth=2, label=f'Convergence Threshold ({CONVERGENCE_THRESHOLD}m)')
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Error (m)', fontsize=10)
    ax3.set_title('Formation Center → Target Error (Primary)\nvs Individual Formation Errors', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
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
    agents_no_cbf, _, _ = run_formation_with_cbf(
        n_agents=NUM_AGENTS, max_iter=1000, dt=0.1, use_cbf=False, animate=False
    )
    
    # Run with CBF
    print("\n--- Running WITH CBF ---")
    agents_with_cbf, _, cbf_stats = run_formation_with_cbf(
        n_agents=NUM_AGENTS, max_iter=1000, dt=0.1, use_cbf=True, animate=False
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
    # Run with moving target (0,0,0) -> (50,50,10)
    agents, target, cbf_stats = run_formation_with_cbf(
        n_agents=NUM_AGENTS,
        max_iter=2000,
        dt=0.02,
        use_cbf=True,  
        animate=True,
        moving_target=True
    )
    
    # Visualize results
    plot_results_3d(agents, target)
    
    # Option 2: Compare with and without CBF
    # agents_no_cbf, agents_with_cbf = compare_with_without_cbf()
    # plot_results_3d(agents_with_cbf, None)