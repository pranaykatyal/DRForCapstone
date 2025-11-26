# DecentralizedFormation.py - FIXED Synchronous Distributed Formation Control with GCBF
"""
FIXED Distributed multi-drone formation control with safety guarantees.

CRITICAL FIXES:
1. TRUE PARALLEL UPDATES: All agents update simultaneously (synchronous)
2. LIMITED SENSING: Only agents within 7m can see target (matches async version)
3. MATCHED HYPERPARAMETERS: Same as async version (10m comm, 7m visual)

Key Features:
- Each drone uses ONLY local information from neighbors
- Distributed GCBF: each drone solves independent local QP
- Communication range: 10.0m (FIXED - matches async)
- Visual sensing range: 7.0m (FIXED - matches async, LIMITED!)
- Obstacles sensed locally and shared via messages
- TRUE SYNCHRONOUS: All agents compute new state, then all apply updates simultaneously

Architecture:
1. Drones broadcast state (position, velocity) to neighbors
2. Drones sense target/obstacles locally (LIMITED 7m range!)
3. Each drone computes desired acceleration (formation control)
4. Each drone filters through local GCBF using neighbor info
5. All drones compute new positions/velocities (parallel)
6. All drones apply updates SIMULTANEOUSLY (true synchronous)
"""

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
VIS_GRAPH_OVERLAY = True
VIS_BARRIERS = True
VIS_2D_TOPVIEW = True
FOCUS_AGENT = 3
CONVERGENCE_THRESHOLD = 0.05
CONVERGENCE_VELOCITY = 0.01
NUM_AGENTS = 5
N_OBSTACLES = 3

# Communication and Sensing Parameters (FIXED - MATCH ASYNC!)
COMM_RANGE = 10.0              # Communication range (m) - FIXED: was 8.0, now 10.0
VISUAL_SENSING_RANGE = 7.0     # Visual sensing for target & obstacles (m) - FIXED: was 12.0, now 7.0 LIMITED!
TARGET_TIME = 15.0
FORMATION_RADIUS = 5.0

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
    Enhanced 3D drone agent with DISTRIBUTED information architecture.
    
    Information Access:
    - Local: own position, velocity, acceleration
    - Via Communication (10m): neighbor positions, velocities, target estimates
    - Via Visual Sensing (7m LIMITED!): target position, obstacle positions
    """
    
    def __init__(self, id, state_3d, target_pos_3d, formation_radius=5.0, 
                 Kp=0.5, Kd=1.2, dt=0.1, max_velocity=3.0, max_acceleration=2.0):
        """Initialize 3D drone agent with distributed architecture."""
        self.id = id
        self.position = np.array(state_3d, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration = np.zeros(3, dtype=float)
        
        # FIXED: Initialize target_pos as None (agents don't know target location initially)
        self.target_pos = None
        self.formation_radius = formation_radius
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        # Message buffer for neighbor information (includes target estimates)
        self.msgs = []  # List of tuples: (neighbor_id, position, velocity, target_estimate)
        
        # Local obstacle map (sensed within VISUAL_SENSING_RANGE)
        self.local_obstacles = []
        
        # Visualization flag
        self.has_direct_sensing = False
        
        # History for plotting
        self.position_hist = [self.position.copy()]
        self.velocity_hist = [self.velocity.copy()]
        self.acceleration_hist = [self.acceleration.copy()]
        
        # For compatibility with old plotting code
        self.state = self.position[:2]
        self.val = self.state
        
    def msg(self):
        """
        Broadcast current state to neighbors.
        
        Returns:
            Tuple: (id, position, velocity, target_estimate)
        """
        return (self.id, self.position.copy(), self.velocity.copy(), 
                self.target_pos.copy() if self.target_pos is not None else None)
    
    def compute_desired_acceleration(self):
        """
        Compute nominal acceleration from formation control law.
        Uses only local information and target estimate.
        
        FIXED: Returns zero acceleration if target_pos is None (hovering)
        """
        if self.target_pos is None:
            # No target knowledge - hover in place
            position_error = np.zeros(3)
            velocity_error = -self.velocity
            acc = self.Kp * position_error + self.Kd * velocity_error
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.max_acceleration:
                acc = acc / acc_norm * self.max_acceleration
            return acc
        
        n_agents = NUM_AGENTS
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        # Formation in XY plane at target Z height
        desired_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0
        ])
        
        # Adaptive weighting based on distance to center
        formation_center = self.target_pos
        dist_to_center = np.linalg.norm(self.position - formation_center)
        
        far_threshold = 3.0 * self.formation_radius
        approaching_threshold = 2.0 * self.formation_radius
        near_threshold = 1.2 * self.formation_radius
        
        if dist_to_center > far_threshold:
            center_weight = 0.9
            formation_weight = 0.1
        elif dist_to_center > approaching_threshold:
            center_weight = 0.7
            formation_weight = 0.3
        elif dist_to_center > near_threshold:
            center_weight = 0.5
            formation_weight = 0.5
        else:
            center_weight = 0.0
            formation_weight = 1.0
        
        center_error = formation_center - self.position
        formation_error = desired_pos - self.position
        
        combined_error = center_weight * center_error + formation_weight * formation_error
        
        vel_error = -self.velocity
        
        acc_desired = self.Kp * combined_error + self.Kd * vel_error
        
        # Saturate acceleration
        acc_norm = np.linalg.norm(acc_desired)
        if acc_norm > self.max_acceleration:
            acc_desired = (acc_desired / acc_norm) * self.max_acceleration
        
        return acc_desired
    
    def update_dynamics(self, acc_safe):
        """Update position and velocity using safe acceleration from CBF."""
        self.acceleration = acc_safe.copy()
        
        self.velocity = self.velocity + self.acceleration * self.dt
        
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > self.max_velocity:
            self.velocity = (self.velocity / vel_norm) * self.max_velocity
        
        self.position = self.position + self.velocity * self.dt
        
        self.state = self.position[:2]
        self.val = self.state
        
        self.position_hist.append(self.position.copy())
        self.velocity_hist.append(self.velocity.copy())
        self.acceleration_hist.append(self.acceleration.copy())
    
    def get_formation_error(self):
        """Compute distance from ideal formation position."""
        if self.target_pos is None:
            return 999.0  # Large error if no target knowledge
            
        n_agents = NUM_AGENTS
        assigned_angle = (2 * np.pi * self.id) / n_agents
        
        ideal_pos = self.target_pos + self.formation_radius * np.array([
            np.cos(assigned_angle),
            np.sin(assigned_angle),
            0.0
        ])
        
        return np.linalg.norm(self.position - ideal_pos)
    
    def clear_msgs(self):
        """Clear message buffer for next iteration."""
        self.msgs = []
    
    def add_msg(self, msg):
        """Receive message from neighbor."""
        self.msgs.append(msg)
    
    # Legacy methods for compatibility
    def stf(self):
        """Legacy: State Transition Function."""
        acc_desired = self.compute_desired_acceleration()
        self.update_dynamics(acc_desired)
    
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
        for j in range(n):
            if j != focus_agent:
                h_vals = [
                    np.linalg.norm(agents[focus_agent].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{focus_agent},{j}", linewidth=2)
        plt.title(f'GCBF Safety Functions for Agent {focus_agent} Over Time (SYNC-FIXED)', fontsize=14)
    else:
        for i in range(n):
            for j in range(i+1, n):
                h_vals = [
                    np.linalg.norm(agents[i].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{i},{j}")
        plt.title('All Pairwise GCBF Safety Functions Over Time (SYNC-FIXED)', fontsize=14)
    
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
        target_end=np.array([25.0, 25.0, 5.0]),
        min_radius=1.0,
        max_radius=2.0):
    """Generate random spherical obstacles along the target's path."""
    obstacles = []

    for _ in range(n_obstacles):
        t = np.random.uniform(0.1, 0.9)
        base = target_start + t * (target_end - target_start)

        offset = np.random.uniform(-10, 10, 3)
        offset[2] = np.random.uniform(-2, 2)

        center = base + offset
        radius = np.random.uniform(min_radius, max_radius)

        obstacles.append({
            "center": center,
            "radius": radius
        })

    return obstacles

# ==================== SIMULATION FUNCTION ====================

def run_formation_with_cbf(n_agents=NUM_AGENTS, max_iter=500, dt=0.02, consensus_alpha=0.3,
                           use_cbf=True, animate=True, moving_target=True):
    """
    Run FIXED SYNCHRONOUS DISTRIBUTED formation control simulation with GCBF safety filter.
    
    CRITICAL FIXES:
    1. TRUE PARALLEL UPDATES: All agents compute new state, then all update simultaneously
    2. LIMITED SENSING: Only agents within 7m can see target (matches async)
    3. MATCHED HYPERPARAMETERS: 10m comm, 7m visual (matches async)
    4. SAME CONSENSUS ALGORITHM: Gradient-based with adjustable alpha (matches async)
    
    Key Changes from Old Synchronous:
    - Agents start with target_pos=None (no initial knowledge)
    - Only agents within 7m can sense target directly
    - Agents beyond 7m must use consensus with neighbors
    - All position/velocity updates applied simultaneously (true parallel)
    - Uses same gradient consensus: x_i(n+1) = x_i(n) + α·∑(x_j(n) - x_i(n))
    
    Args:
        n_agents: Number of drones
        max_iter: Maximum iterations
        dt: Timestep (seconds)
        consensus_alpha: Consensus weight (0-1) - same as async version
        use_cbf: Enable CBF safety filtering
        animate: Enable visualization
        moving_target: Enable moving target
    """
    
    # Import DISTRIBUTED CBF filter
    if use_cbf:
        try:
            from DecentralizedGCBF import GraphCBFSafetyFilter
        except ImportError:
            print("WARNING: Could not import GraphCBFSafetyFilter from DecentralizedGCBF.py")
            print("Place DecentralizedGCBF.py in the same directory")
            use_cbf = False
    
    # Initialize CBF filter
    cbf_filter = None
    if use_cbf:
        cbf_filter = GraphCBFSafetyFilter(
            n_drones=n_agents,
            safety_distance=2.0,
            sensing_radius=COMM_RANGE,  # Uses communication range
            alpha1=3.0,
            alpha2=1.5,
            max_acceleration=5.0
        )
        
        v_max = 5.0
        R_min = cbf_filter.compute_minimum_sensing_radius(
            gamma=cbf_filter.alpha2, 
            v_max=v_max
        )
        print(f"Minimum sensing radius for safety: {R_min:.2f}m")
        if cbf_filter.R_sense < R_min:
            print(f"WARNING: Using R={cbf_filter.R_sense}m < {R_min:.2f}m")
    
    # Initialize target
    formation_radius = FORMATION_RADIUS
    
    if moving_target:
        target = MovingTarget(
            start_pos=np.array([0.0, 0.0, 0.0]),
            end_pos=np.array([25.0, 25.0, 5.0]),
            duration=TARGET_TIME,
            dt=dt
        )
        print(f"Moving target: {target.start_pos} -> {target.end_pos} over {target.duration}s")
    else:
        target = None
        target_pos = np.array([0.0, 0.0, 2.0])
    
    agents = []
    np.random.seed(42)
    
    print(f"\n{'='*60}")
    print(f"SYNCHRONOUS DISTRIBUTED Formation Control (FIXED)")
    print(f"{'='*60}")
    print(f"Agents: {n_agents}")
    print(f"Target: {'MOVING' if moving_target else 'STATIC'}")
    print(f"Formation radius: {formation_radius}m")
    print(f"CBF enabled: {use_cbf} (DISTRIBUTED)")
    print(f"Communication range: {COMM_RANGE}m (FIXED - matches async)")
    print(f"Visual sensing range: {VISUAL_SENSING_RANGE}m (FIXED - LIMITED, matches async)")
    print(f"Consensus alpha: {consensus_alpha} (gradient-based, matches async)")
    print(f"Timestep: {dt}s, Max Duration: {max_iter*dt:.1f}s")
    print(f"Controller: PD with Kp=0.5, Kd=1.2")
    print(f"Update mode: TRUE PARALLEL (synchronous)")
    print(f"{'='*60}\n")
    
    # FIXED: Start agents at same position as async version (7m center)
    agents_start_center = np.array([7.0, 0.0, 2.0])
    current_target_pos = target.position if moving_target else target_pos
    
    print(f"Formation center start: {agents_start_center}")
    print(f"Distance from target: {np.linalg.norm(agents_start_center - current_target_pos):.1f}m")
    print(f"Visual range: {VISUAL_SENSING_RANGE}m")
    print(f"Formation radius: {formation_radius}m")
    
    # Calculate initial visibility
    temp_seeing_count = 0
    for i in range(n_agents):
        angle = (2 * np.pi * i) / n_agents
        init_pos = agents_start_center + formation_radius * np.array([
            np.cos(angle), np.sin(angle), 0.0
        ])
        dist_to_target = np.linalg.norm(init_pos - current_target_pos)
        if dist_to_target < VISUAL_SENSING_RANGE:
            temp_seeing_count += 1
    
    print(f"→ Initial visibility: {temp_seeing_count}/{n_agents} agents within {VISUAL_SENSING_RANGE}m range")
    print(f"→ This creates information asymmetry (same as async)")
    print()
    
    for i in range(n_agents):
        angle = (2 * np.pi * i) / n_agents
        init_pos = agents_start_center + formation_radius * np.array([
            np.cos(angle), np.sin(angle), 0.0
        ])
        
        agents.append(DroneAgent(
            id=i,
            state_3d=init_pos,
            target_pos_3d=None,  # FIXED: No initial target knowledge!
            formation_radius=formation_radius,
            Kp=0.5,
            Kd=1.2,
            dt=dt,
            max_velocity=5.0,
            max_acceleration=4.0
        ))
        print(f"Agent {i} initialized at: {init_pos}")
    
    # Generate obstacles (same as async)
    if moving_target:
        obstacles = generate_random_spherical_obstacles(
            n_obstacles=N_OBSTACLES,
            target_start=target.start_pos,
            target_end=target.end_pos,
            min_radius=1.0,
            max_radius=2.0
        )
    else:
        obstacles = []

    print(f"\nGenerated {len(obstacles)} spherical obstacles:")
    for i, obs in enumerate(obstacles):
        print(f"  Obs {i}: center={obs['center']}, radius={obs['radius']:.2f}")
    print()
    
    # Check initial visibility
    initial_seeing = sum(1 for a in agents 
                        if np.linalg.norm(a.position - current_target_pos) < VISUAL_SENSING_RANGE)
    print(f"→ {initial_seeing}/{n_agents} agents can initially see target")
    print("→ Remaining agents must use consensus!\n")
    
    # ==================== ANIMATION SETUP ====================
    if animate:
        plt.ion()
        
        if VIS_2D_TOPVIEW:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(121, projection='3d')
            ax2d = fig.add_subplot(122)
        else:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax2d = None
        
        colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
        
        drone_scatter = ax.scatter([], [], [], s=200, c='blue', marker='o', 
                                  edgecolors='black', linewidths=2)
        
        current_target_pos = target.position if moving_target else target_pos
        target_scatter = ax.scatter(current_target_pos[0], current_target_pos[1], current_target_pos[2], 
                  c='red', marker='X', s=500, edgecolors='black', linewidths=2,
                  label='Target')
        
        if moving_target:
            target_traj_line, = ax.plot([], [], [], 'r--', linewidth=2, alpha=0.5, label='Target Path')
        
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
        
        obstacle_scatters = []
        for obs in obstacles:
            sx, sy, sz = obs["center"]
            sc = ax.scatter(sx, sy, sz,
                c="purple", s=150, marker="o", alpha=0.7,
                edgecolors="black", linewidths=2)
            obstacle_scatters.append(sc)

            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = obs["radius"] * np.outer(np.cos(u), np.sin(v)) + sx
            y = obs["radius"] * np.outer(np.sin(u), np.sin(v)) + sy
            z = obs["radius"] * np.outer(np.ones(np.size(u)), np.cos(v)) + sz
            ax.plot_wireframe(x, y, z, color='purple', alpha=0.3, linewidth=1)
        
        theta = np.linspace(0, 2*np.pi, 50)
        circle_x = current_target_pos[0] + formation_radius * np.cos(theta)
        circle_y = current_target_pos[1] + formation_radius * np.sin(theta)
        circle_z = np.ones_like(theta) * current_target_pos[2]
        formation_circle, = ax.plot(circle_x, circle_y, circle_z, 'g--', alpha=0.3, linewidth=2)
        
        trajectory_lines = []
        for i in range(n_agents):
            line, = ax.plot([], [], [], color=colors[i], linewidth=1.5, 
                          alpha=0.6, label=f'Drone {i}')
            trajectory_lines.append(line)
        
        quiver = None
        graph_lines = []
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D SYNC Formation (GREEN=Direct, COLOR=Consensus)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 30)
        ax.set_ylim(-15, 30)
        ax.set_zlim(0, 10)
        ax.grid(True, alpha=0.3)
        
        if VIS_2D_TOPVIEW and ax2d is not None:
            x_init = [a.position[0] for a in agents]
            y_init = [a.position[1] for a in agents]

            drone_scatter_2d = ax2d.scatter(
                x_init, y_init, s=400, c=colors, marker='o',
                edgecolor='black', linewidths=1.5
            )
                        
            target_scatter_2d = ax2d.scatter(current_target_pos[0], current_target_pos[1], 
                       c='red', marker='X', s=500, edgecolors='black', linewidths=2, zorder=5)
            
            if moving_target:
                target_traj_line_2d, = ax2d.plot([], [], 'r--', linewidth=2, alpha=0.5, zorder=1)
            
            ideal_markers_2d = []
            position_labels_2d = []
            for i in range(n_agents):
                angle = (2 * np.pi * i) / n_agents
                ideal_x = current_target_pos[0] + formation_radius * np.cos(angle)
                ideal_y = current_target_pos[1] + formation_radius * np.sin(angle)
                marker = ax2d.scatter(ideal_x, ideal_y, c='green', marker='x', s=300, 
                           alpha=0.5, edgecolors='orange', linewidths=3, zorder=4)
                ideal_markers_2d.append(marker)
                label = ax2d.text(ideal_x, ideal_y + 0.7, f'Pos{i}', fontsize=9, 
                        ha='center', color='orange', weight='bold', zorder=10)
                position_labels_2d.append(label)
            
            obstacle_scatters_2d = []
            for obs in obstacles:
                sx, sy, sz = obs["center"]
                sc = ax2d.scatter(sx, sy,
                    c="purple", s=350, marker="o",
                    alpha=0.4, edgecolors="black", linewidths=2)
                obstacle_scatters_2d.append(sc)

                theta_2d = np.linspace(0, 2*np.pi, 40)
                circ_x = sx + obs["radius"] * np.cos(theta_2d)
                circ_y = sy + obs["radius"] * np.sin(theta_2d)
                ax2d.plot(circ_x, circ_y,
                    color="purple", linestyle="--", alpha=0.6, linewidth=2)
            
            formation_circle_2d, = ax2d.plot(circle_x, circle_y, 'g--', alpha=0.3, linewidth=2, zorder=1)
            
            trajectory_lines_2d = []
            for i in range(n_agents):
                line, = ax2d.plot([], [], color=colors[i], linewidth=2, 
                                alpha=0.4, zorder=2)
                trajectory_lines_2d.append(line)
            
            graph_lines_2d = []
            
            drone_labels_2d = []
            for i in range(n_agents):
                label = ax2d.text(0, 0, f'D{i}', fontsize=10, ha='center', va='center',
                                 color='white', weight='bold', zorder=15)
                drone_labels_2d.append(label)
            
            ax2d.set_xlabel('X (m)', fontsize=12)
            ax2d.set_ylabel('Y (m)', fontsize=12)
            ax2d.set_title(f'2D Top View - SYNC (TRUE PARALLEL)', 
                         fontsize=14, fontweight='bold')
            ax2d.axis('equal')
            ax2d.grid(True, alpha=0.3)
            ax2d.set_xlim(-15, 30)
            ax2d.set_ylim(-15, 30)
        
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # ==================== END ANIMATION SETUP ====================
    
    # Main simulation loop
    print(f"Starting SYNCHRONOUS (TRUE PARALLEL) simulation...")
    converged = False
    convergence_count = 0
    convergence_patience = 20
    
    for iteration in range(max_iter):
        # ====================================================================
        # STEP 0: Update moving target
        # ====================================================================
        if moving_target:
            target.update()
            true_target = target.position.copy()
        else:
            true_target = target_pos
        
        # ====================================================================
        # STEP 1: TARGET SENSING AND CONSENSUS (DISTRIBUTED, LIMITED 7m!)
        # FIXED: Only agents within 7m can see target directly
        # FIXED: Uses same gradient-based consensus as async version
        # ====================================================================
        for agent in agents:
            dist_to_target = np.linalg.norm(agent.position - true_target)
            
            if dist_to_target < VISUAL_SENSING_RANGE:  # FIXED: Limited 7m sensing!
                # Direct sensing
                agent.target_pos = true_target.copy()
                agent.has_direct_sensing = True
            else:
                # Out of visual range: use consensus with neighbors
                agent.has_direct_sensing = False
                
                if agent.target_pos is None:
                    # First time - initialize from any available neighbor
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).ravel()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                agent.target_pos = target_arr.copy()
                                break
                else:
                    # Apply gradient-based consensus (same as async!)
                    # x_i(n+1) = x_i(n) + α·∑_j(x_j(n) - x_i(n))
                    consensus_gradient = np.zeros(3)
                    num_neighbors = 0
                    
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).ravel()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                # Accumulate gradient: (neighbor_estimate - my_estimate)
                                consensus_gradient += (target_arr - agent.target_pos)
                                num_neighbors += 1
                    
                    if num_neighbors > 0:
                        # Apply gradient update: x_i(n+1) = x_i(n) + α·∑(x_j - x_i(n))
                        new_estimate = agent.target_pos + consensus_alpha * consensus_gradient
                        
                        if np.all(np.isfinite(new_estimate)):
                            agent.target_pos = new_estimate
                        else:
                            print(f"WARNING: Agent {agent.id} computed invalid consensus estimate")
        
        # ====================================================================
        # STEP 2: MESSAGE PASSING (DISTRIBUTED)
        # FIXED: Messages now include target estimates
        # ====================================================================
        for agent in agents:
            agent.clear_msgs()
        
        for agent in agents:
            msg = agent.msg()  # (id, position, velocity, target_estimate)
            
            for other_agent in agents:
                if agent.id != other_agent.id:
                    dist = np.linalg.norm(agent.position - other_agent.position)
                    if dist <= COMM_RANGE:
                        other_agent.add_msg(msg)
        
        # ====================================================================
        # STEP 3: LOCAL OBSTACLE SENSING (DISTRIBUTED)
        # ====================================================================
        for agent in agents:
            agent.local_obstacles = []
            for obs in obstacles:
                dist_to_obs = np.linalg.norm(agent.position - obs['center'])
                if dist_to_obs <= VISUAL_SENSING_RANGE:
                    agent.local_obstacles.append(obs)
        
        # ====================================================================
        # STEP 4: Compute DESIRED accelerations (formation controller)
        # ====================================================================
        acc_desired = np.zeros((n_agents, 3))
        for i, agent in enumerate(agents):
            acc_desired[i] = agent.compute_desired_acceleration()
        
        # ====================================================================
        # STEP 5: DISTRIBUTED CBF FILTERING
        # ====================================================================
        acc_safe = np.zeros((n_agents, 3))
        
        if use_cbf:
            for i, agent in enumerate(agents):
                neighbor_positions = []
                neighbor_velocities = []
                
                for msg in agent.msgs:
                    neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                    neighbor_positions.append(neighbor_pos)
                    neighbor_velocities.append(neighbor_vel)
                
                acc_safe[i] = cbf_filter.filter_acceleration_single_drone(
                    my_position=agent.position,
                    my_velocity=agent.velocity,
                    my_acc_desired=acc_desired[i],
                    neighbor_positions=neighbor_positions,
                    neighbor_velocities=neighbor_velocities,
                    obstacles=agent.local_obstacles
                )
            
            positions = np.array([agent.position for agent in agents])
            is_safe, violations = cbf_filter.check_safety(positions, obstacles)
            if not is_safe:
                print(f"\nSAFETY VIOLATION at iteration {iteration}:")
                for v in violations:
                    print(f"   {v}")
        else:
            acc_safe = acc_desired
        
        # ====================================================================
        # STEP 6: TRUE PARALLEL UPDATE (SYNCHRONOUS!)
        # CRITICAL FIX: Compute ALL new states first, THEN apply simultaneously
        # ====================================================================
        new_positions = []
        new_velocities = []
        new_accelerations = []
        
        for i, agent in enumerate(agents):
            # Compute new velocity
            new_vel = agent.velocity + acc_safe[i] * agent.dt
            vel_norm = np.linalg.norm(new_vel)
            if vel_norm > agent.max_velocity:
                new_vel = (new_vel / vel_norm) * agent.max_velocity
            
            # Compute new position
            new_pos = agent.position + new_vel * agent.dt
            
            new_positions.append(new_pos)
            new_velocities.append(new_vel)
            new_accelerations.append(acc_safe[i])
        
        # Apply ALL updates simultaneously (TRUE SYNCHRONOUS!)
        for i, agent in enumerate(agents):
            agent.position = new_positions[i].copy()
            agent.velocity = new_velocities[i].copy()
            agent.acceleration = new_accelerations[i].copy()
            
            agent.state = agent.position[:2]
            agent.val = agent.state
            
            agent.position_hist.append(agent.position.copy())
            agent.velocity_hist.append(agent.velocity.copy())
            agent.acceleration_hist.append(agent.acceleration.copy())
        
        # ==================== CONVERGENCE CHECK ====================
        formation_errors = [agent.get_formation_error() for agent in agents]
        avg_error = np.mean(formation_errors)
        max_error = np.max(formation_errors)
        avg_vel = np.mean([np.linalg.norm(a.velocity) for a in agents])
        max_vel = np.max([np.linalg.norm(a.velocity) for a in agents])
        
        formation_center = np.mean([agent.position for agent in agents], axis=0)
        center_to_target_error = np.linalg.norm(formation_center - true_target)
        
        if not hasattr(agents[0], 'center_error_hist'):
            for agent in agents:
                agent.center_error_hist = []
        for agent in agents:
            agent.center_error_hist.append(center_to_target_error)
        
        target_stopped = False
        if moving_target:
            target_stopped = (np.linalg.norm(target.velocity) < 0.01)
        else:
            target_stopped = True
        
        if target_stopped and max_error < CONVERGENCE_THRESHOLD and max_vel < CONVERGENCE_VELOCITY:
            convergence_count += 1
            if convergence_count >= convergence_patience:
                converged = True
                print(f"\nFORMATION CONVERGED at iteration {iteration}")
                print(f"   Max error: {max_error:.3f}m < {CONVERGENCE_THRESHOLD}m")
                print(f"   Max velocity: {max_vel:.3f}m/s < {CONVERGENCE_VELOCITY}m/s")
                if not animate:
                    break
        else:
            convergence_count = 0
        
        # ==================== ANIMATION UPDATE ====================
        if animate and iteration % 2 == 0:
            current_positions = np.array([agent.position for agent in agents])
            drone_scatter._offsets3d = (current_positions[:, 0], 
                                        current_positions[:, 1], 
                                        current_positions[:, 2])
            
            # Color based on direct sensing (green) vs consensus (color)
            agent_colors = []
            for agent in agents:
                if agent.has_direct_sensing:
                    agent_colors.append([0, 1, 0, 1])  # GREEN
                else:
                    agent_colors.append(colors[agent.id])
            drone_scatter.set_color(agent_colors)
            
            for i, agent in enumerate(agents):
                traj = np.array(agent.position_hist)
                trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                trajectory_lines[i].set_3d_properties(traj[:, 2])
            
            if moving_target:
                current_target_pos = target.position
                target_scatter._offsets3d = ([current_target_pos[0]], [current_target_pos[1]], [current_target_pos[2]])
                
                traj_target = np.array(target.position_hist)
                target_traj_line.set_data(traj_target[:, 0], traj_target[:, 1])
                target_traj_line.set_3d_properties(traj_target[:, 2])
                
                for i, marker in enumerate(ideal_markers):
                    angle = (2 * np.pi * i) / n_agents
                    ideal_pos = current_target_pos + formation_radius * np.array([np.cos(angle), np.sin(angle), 0.0])
                    marker._offsets3d = ([ideal_pos[0]], [ideal_pos[1]], [ideal_pos[2]])
                
                circle_x = current_target_pos[0] + formation_radius * np.cos(theta)
                circle_y = current_target_pos[1] + formation_radius * np.sin(theta)
                circle_z = np.ones_like(theta) * current_target_pos[2]
                formation_circle.set_data(circle_x, circle_y)
                formation_circle.set_3d_properties(circle_z)
            
            if quiver is not None:
                quiver.remove()
            
            current_velocities = np.array([agent.velocity for agent in agents])
            vel_scale = 0.5
            quiver = ax.quiver(current_positions[:, 0], current_positions[:, 1], current_positions[:, 2],
                              current_velocities[:, 0]*vel_scale, 
                              current_velocities[:, 1]*vel_scale, 
                              current_velocities[:, 2]*vel_scale,
                              color='blue', alpha=0.6, arrow_length_ratio=0.3, linewidths=1.5)
            
            if VIS_GRAPH_OVERLAY and use_cbf:
                for line in graph_lines:
                    line.remove()
                graph_lines.clear()
                
                if FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
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
                                
                                line = ax.plot([current_positions[FOCUS_AGENT,0], current_positions[j,0]],
                                             [current_positions[FOCUS_AGENT,1], current_positions[j,1]],
                                             [current_positions[FOCUS_AGENT,2], current_positions[j,2]],
                                             color=color, linewidth=linewidth, alpha=alpha)[0]
                                graph_lines.append(line)
            
            if VIS_2D_TOPVIEW and ax2d is not None:
                drone_scatter_2d.set_offsets(current_positions[:, :2])
                
                for i in range(n_agents):
                    drone_labels_2d[i].set_position((current_positions[i, 0], current_positions[i, 1]))
                
                for i, agent in enumerate(agents):
                    traj = np.array(agent.position_hist)
                    trajectory_lines_2d[i].set_data(traj[:, 0], traj[:, 1])
                
                if moving_target:
                    target_scatter_2d.set_offsets([[current_target_pos[0], current_target_pos[1]]])
                    target_traj_line_2d.set_data(traj_target[:, 0], traj_target[:, 1])
                    
                    for i, marker in enumerate(ideal_markers_2d):
                        angle = (2 * np.pi * i) / n_agents
                        ideal_x = current_target_pos[0] + formation_radius * np.cos(angle)
                        ideal_y = current_target_pos[1] + formation_radius * np.sin(angle)
                        marker.set_offsets([[ideal_x, ideal_y]])
                        position_labels_2d[i].set_position((ideal_x, ideal_y + 0.7))
                    
                    formation_circle_2d.set_data(circle_x, circle_y)
                
                if use_cbf and FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                    for patch in list(ax2d.patches):
                        if isinstance(patch, plt.Circle):
                            patch.remove()
                    
                    agent_pos = current_positions[FOCUS_AGENT]
                    
                    sensing_circle = plt.Circle((agent_pos[0], agent_pos[1]), 
                                              cbf_filter.R_sense, 
                                              fill=False, color='blue', 
                                              linestyle=':', linewidth=2, alpha=0.4,
                                              label='Comm Range', zorder=3)
                    ax2d.add_patch(sensing_circle)
                    
                    safety_circle = plt.Circle((agent_pos[0], agent_pos[1]), 
                                             cbf_filter.d_safe, 
                                             fill=True, color='red', 
                                             alpha=0.15, zorder=2)
                    ax2d.add_patch(safety_circle)
                    
                    safety_circle_edge = plt.Circle((agent_pos[0], agent_pos[1]), 
                                                   cbf_filter.d_safe, 
                                                   fill=False, color='red', 
                                                   linestyle='--', linewidth=2.5, alpha=0.7,
                                                   label='Safety Zone', zorder=3)
                    ax2d.add_patch(safety_circle_edge)
                
                for line in graph_lines_2d:
                    line.remove()
                graph_lines_2d.clear()
                
                if VIS_GRAPH_OVERLAY and use_cbf:
                    if FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
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
                
                all_x = current_positions[:, 0]
                all_y = current_positions[:, 1]
                if moving_target:
                    all_x = np.append(all_x, current_target_pos[0])
                    all_y = np.append(all_y, current_target_pos[1])
                
                all_x = all_x[np.isfinite(all_x)]
                all_y = all_y[np.isfinite(all_y)]
                
                if len(all_x) > 0 and len(all_y) > 0:
                    margin = 10.0
                    x_center = np.mean(all_x)
                    y_center = np.mean(all_y)
                    x_range = max(15.0, (all_x.max() - all_x.min()) / 2 + margin)
                    y_range = max(15.0, (all_y.max() - all_y.min()) / 2 + margin)
                    
                    ax2d.set_xlim(x_center - x_range, x_center + x_range)
                    ax2d.set_ylim(y_center - y_range, y_center + y_range)
            
            seeing_count = sum(1 for a in agents if a.has_direct_sensing)
            consensus_only = n_agents - seeing_count
            avg_acc = np.mean([np.linalg.norm(a.acceleration) for a in agents])
            info_text.set_text(
                f"Iteration: {iteration:04d}\n"
                f"===SYNC (PARALLEL)===\n"
                f"Direct Sensing: {seeing_count}/{n_agents}\n"
                f"Consensus Only: {consensus_only}/{n_agents}\n"
                f"Alpha: {consensus_alpha:.2f}\n"
                f"==================\n"
                f"Center->Target: {center_to_target_error:.3f} m\n"
                f"Avg form error: {avg_error:.3f} m\n"
                f"Max form error: {max_error:.3f} m\n"
                f"Avg vel: {avg_vel:.3f} m/s\n"
                f"Max vel: {max_vel:.3f} m/s\n"
                f"Avg acc: {avg_acc:.3f} m/s^2"
            )
            
            plt.pause(0.001)
            
            all_x = current_positions[:, 0]
            all_y = current_positions[:, 1]
            all_z = current_positions[:, 2]
            if moving_target:
                all_x = np.append(all_x, current_target_pos[0])
                all_y = np.append(all_y, current_target_pos[1])
                all_z = np.append(all_z, current_target_pos[2])
            
            margin = 8.0
            x_center = np.mean(all_x)
            y_center = np.mean(all_y)
            z_center = np.mean(all_z)
            x_range = max(15.0, (all_x.max() - all_x.min()) / 2 + margin)
            y_range = max(15.0, (all_y.max() - all_y.min()) / 2 + margin)
            z_range = max(8.0, (all_z.max() - all_z.min()) / 2 + margin)
            
            ax.set_xlim(x_center - x_range, x_center + x_range)
            ax.set_ylim(y_center - y_range, y_center + y_range)
            ax.set_zlim(max(0, z_center - z_range), z_center + z_range)
            
            plt.draw()
            plt.pause(0.01)
            
            if converged:
                plt.pause(1.0)
                break
        # ==================== END ANIMATION UPDATE ====================
        
        if iteration % 50 == 0:
            seeing_count = sum(1 for a in agents if a.has_direct_sensing)
            no_target = [a.id for a in agents if a.target_pos is None]
            if no_target and iteration < 200:
                print(f"  WARNING: Agents without target: {no_target}")
            
            print(f"Iter {iteration:3d}: "
                  f"Center->Target: {center_to_target_error:.3f}m, "
                  f"Formation error: avg={avg_error:.3f}m, max={max_error:.3f}m, "
                  f"Seeing: {seeing_count}/{n_agents}")
        
        if converged and not animate:
            break
    
    if animate:
        plt.ioff()
        plt.show()
        print("\nAnimation window closed.")
    
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    
    if use_cbf:
        stats = cbf_filter.get_statistics()
        print("\nCBF Statistics (DISTRIBUTED):")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
    
    final_errors = [agent.get_formation_error() for agent in agents]
    print(f"\nFinal Formation Errors:")
    for i, error in enumerate(final_errors):
        if agents[i].target_pos is None:
            print(f"  Agent {i}: {error:.3f}m (NO TARGET KNOWLEDGE)")
        else:
            print(f"  Agent {i}: {error:.3f}m")
    print(f"  Average: {np.mean(final_errors):.3f}m")
    print(f"  Maximum: {np.max(final_errors):.3f}m")
    
    final_velocities = [np.linalg.norm(agent.velocity) for agent in agents]
    print(f"\nFinal Velocities:")
    for i, vel in enumerate(final_velocities):
        print(f"  Agent {i}: {vel:.3f}m/s")
    print(f"  Average: {np.mean(final_velocities):.3f}m/s")
    
    if converged:
        actual_iter = len(agents[0].position_hist) - 1
        print(f"\nFormation converged after {actual_iter} iterations ({actual_iter*dt:.1f}s)")
    else:
        print(f"\nFormation did not converge within {max_iter} iterations")
    
    # Compute drift metric (same as async)
    motion_end = int(15.0 / dt)
    avg_drift_motion = np.mean([agents[0].center_error_hist[i] for i in range(min(motion_end, len(agents[0].center_error_hist)))])
    print(f"\nAverage drift during target motion (0-15s): {avg_drift_motion:.3f}m")
    
    cbf_stats = cbf_filter.get_statistics() if use_cbf else None
    return agents, target, cbf_stats


def plot_results_3d(agents, target=None):
    """Visualize 3D trajectories and final formation."""
    n_agents = len(agents)
    
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
    
    for i, agent in enumerate(agents):
        traj = np.array(agent.position_hist)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors[i], linewidth=2, label=f'Drone {agent.id}')
        
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                   color=colors[i], marker='o', s=100, edgecolors='black')
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   color=colors[i], marker='*', s=200, edgecolors='black')
    
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
    ax1.set_title('3D Drone Trajectories (SYNC-FIXED)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(132)
    
    for i, agent in enumerate(agents):
        ax2.plot(agent.position[0], agent.position[1], 'o', 
                markersize=12, color=colors[i], label=f'Drone {agent.id}')
        
        if agent.target_pos is not None:
            angle = (2 * np.pi * agent.id) / n_agents
            ideal_x = agent.target_pos[0] + agent.formation_radius * np.cos(angle)
            ideal_y = agent.target_pos[1] + agent.formation_radius * np.sin(angle)
            ax2.plot(ideal_x, ideal_y, 'x', markersize=12, 
                    color=colors[i], markeredgewidth=3)
            
            ax2.plot([agent.position[0], ideal_x], 
                    [agent.position[1], ideal_y], 
                    '--', color=colors[i], alpha=0.5, linewidth=1)
    
    theta = np.linspace(0, 2*np.pi, 100)
    target_pos = None
    for agent in agents:
        if agent.target_pos is not None:
            target_pos = agent.target_pos
            break
    
    if target_pos is not None:
        circle_x = target_pos[0] + agents[0].formation_radius * np.cos(theta)
        circle_y = target_pos[1] + agents[0].formation_radius * np.sin(theta)
        ax2.plot(circle_x, circle_y, 'k--', alpha=0.3, linewidth=2)
        ax2.plot(target_pos[0], target_pos[1], 'r+', markersize=20, markeredgewidth=3)
    
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_title('Final Formation (Top View)\no = actual, x = ideal', 
                 fontsize=12, fontweight='bold')
    ax2.axis('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(133)
    
    max_len = max(len(agent.position_hist) for agent in agents)
    time = np.arange(max_len) * agents[0].dt
    
    if hasattr(agents[0], 'center_error_hist'):
        center_errors = agents[0].center_error_hist
        ax3.plot(time[:len(center_errors)], center_errors, 'b-', 
                linewidth=3, label='Formation Center -> Target', alpha=0.8)
    
    for i, agent in enumerate(agents):
        if agent.target_pos is None:
            continue
            
        errors = [np.linalg.norm(np.array(agent.position_hist[j]) - 
                                 (agent.target_pos + agent.formation_radius * np.array([
                                     np.cos(2*np.pi*agent.id/NUM_AGENTS),
                                     np.sin(2*np.pi*agent.id/NUM_AGENTS),
                                     0.0
                                 ])))
                 for j in range(len(agent.position_hist))]
        ax3.plot(time[:len(errors)], errors, color=colors[i], 
                linewidth=1, alpha=0.3, label=f'Drone {agent.id} form error')
    
    avg_errors = []
    for t in range(max_len):
        errors_at_t = []
        for agent in agents:
            if t < len(agent.position_hist) and agent.target_pos is not None:
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
    
    ax3.axhline(y=CONVERGENCE_THRESHOLD, color='r', linestyle=':', 
               linewidth=2, label=f'Convergence Threshold ({CONVERGENCE_THRESHOLD}m)')
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Error (m)', fontsize=10)
    ax3.set_title('Formation Errors Over Time (SYNC-FIXED)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ==================== MAIN ====================

if __name__ == "__main__":
    agents, target, cbf_stats = run_formation_with_cbf(
        n_agents=NUM_AGENTS,
        max_iter=2000,
        dt=0.02,
        consensus_alpha=0.7,  # Try 0.3, 0.7 to compare with async
        use_cbf=True,  
        animate=True,
        moving_target=True
    )
    
    if VIS_BARRIERS and cbf_stats is not None:
        from DecentralizedGCBF import GraphCBFSafetyFilter
        cbf_filter = GraphCBFSafetyFilter(
            n_drones=NUM_AGENTS,
            safety_distance=2.0,
            sensing_radius=COMM_RANGE,
            alpha1=3.0,
            alpha2=1.5,
            max_acceleration=5.0
        )
        plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT)
    
    plot_results_3d(agents, target)