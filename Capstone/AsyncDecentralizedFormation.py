import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import logging
logging.getLogger('matplotlib.axes._base').setLevel(logging.ERROR)

# Import from DecentralizedFormation.py
try:
    from DecentralizedFormation import (
        DroneAgent, MovingTarget, NUM_AGENTS, COMM_RANGE, 
        FORMATION_RADIUS
    )
    print("[OK] Imported from DecentralizedFormation.py")
except ImportError as e:
    print(f"[ERROR] Could not import: {e}")
    sys.exit(1)

# Import CBF filter
try:
    from DecentralizedGCBF import GraphCBFSafetyFilter
    CBF_AVAILABLE = True
    print("[OK] Imported GraphCBFSafetyFilter")
except ImportError as e:
    print(f"[WARNING] Could not import CBF: {e}")
    CBF_AVAILABLE = False

# Visualization toggles (match DecentralizedFormation.py)
VIS_GRAPH_OVERLAY = True
VIS_2D_TOPVIEW = True
FOCUS_AGENT = 3
CONVERGENCE_THRESHOLD = 0.05
CONVERGENCE_VELOCITY = 0.01

VISUAL_SENSING_RANGE_LIMITED = 7.0  

COMM_RANGE = 10.0  


class AsyncDroneAgent(DroneAgent):
    """
    Asynchronous drone agent with phase lag.
    
    Extends DroneAgent with:
    - phase_lag: Time offset for sequential updates (seconds)
    - async_alpha: Consensus weight for target estimation
    """
    
    def __init__(self, id, state_3d, target_pos_3d, formation_radius=FORMATION_RADIUS,
                 phase_lag=0.0, async_alpha=0.3, Kp=0.5, Kd=1.2, dt=0.02, 
                 max_velocity=5.0, max_acceleration=4.0):
        super().__init__(id, state_3d, target_pos_3d, formation_radius,
                        Kp, Kd, dt, max_velocity, max_acceleration)
        self.phase_lag = phase_lag
        self.async_alpha = async_alpha
        self.cbf_activations = 0
        self.has_direct_sensing = False  # Track for visualization
    
    def msg(self):
        """
        Override to include target_pos in message for consensus.
        
        Returns:
            Tuple: (id, position, velocity, target_pos)
            
        Note: Returns None for target_pos if it's None or contains NaN/Inf
        """
        target_to_send = None
        if self.target_pos is not None:
            # Check if target_pos is valid (no NaN or Inf)
            if np.all(np.isfinite(self.target_pos)):
                target_to_send = self.target_pos.copy()
            else:
                # Don't broadcast corrupted data
                target_to_send = None
        
        return (self.id, self.position.copy(), self.velocity.copy(), target_to_send)
    
    def compute_desired_acceleration(self):
        """
        Override to handle None target_pos (agent doesn't know target location yet).
        
        If target_pos is None, agent hovers in place until it discovers target.
        """
        if self.target_pos is None:

            position_error = np.zeros(3)  # Stay where you are
            velocity_error = -self.velocity  # Damp velocity to zero
            
            acc = self.Kp * position_error + self.Kd * velocity_error
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.max_acceleration:
                acc = acc / acc_norm * self.max_acceleration
            return acc
        else:
            # Has target knowledge - use normal formation controller
            return super().compute_desired_acceleration()
    
    def set_formation_active(self, active):
        """Control whether formation control is active."""
        self.formation_active = active


def run_async_formation_with_cbf(n_agents=NUM_AGENTS, max_iter=2000, dt=0.02, 
                                  phase_spread_ms=16.0, consensus_alpha=0.3,
                                  use_cbf=True, animate=True, moving_target=True):
    """
    Run ASYNCHRONOUS formation control with GCBF safety filter.
    
    Architecture:
    1. Agents broadcast state via messages (COMM_RANGE = 6m)
    2. ASYNC target consensus: sequential updates by phase lag 
    3. SYNC formation control: parallel acceleration computation
    4. Distributed GCBF: local QP per drone
    5. Update dynamics in parallel
    
    Args:
        n_agents: Number of drones
        max_iter: Maximum iterations
        dt: Timestep (0.02s to match DecentralizedFormation)
        phase_spread_ms: Phase lag spread across agents (milliseconds)
        consensus_alpha: Consensus weight
        use_cbf: Enable distributed CBF safety filtering
        animate: Enable real-time visualization
        moving_target: Enable moving target
    """
    
    # Import CBF filter
    if use_cbf:
        if not CBF_AVAILABLE:
            print("WARNING: CBF requested but not available")
            use_cbf = False
    
    # Initialize CBF filter
    cbf_filter = None
    if use_cbf:
        cbf_filter = GraphCBFSafetyFilter(
            n_drones=n_agents,
            safety_distance=2.0,
            sensing_radius=COMM_RANGE,
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
    formation_radius = 5.0  
    
    if moving_target:
        target = MovingTarget(
            start_pos=np.array([0.0, 0.0, 0.0]), # start 
            end_pos=np.array([25.0, 25.0, 5.0]), # end
            duration=15.0,
            dt=dt
        )
        print(f"Moving target: {target.start_pos} -> {target.end_pos} over {target.duration}s")
    else:
        target = None
        target_pos = np.array([0.0, 0.0, 2.0])
    
    # Phase lags for asynchronous updates
    phase_lags_sec = np.linspace(0, phase_spread_ms/1000, n_agents)
    
    agents = []
    np.random.seed(42)
    
    
    agents_start_center = np.array([7.0, 0.0, 2.0])  
    current_target_pos = target.position if moving_target else target_pos
    
    print(f"Formation center start: {agents_start_center}")
    print(f"Distance from target: {np.linalg.norm(agents_start_center - current_target_pos):.1f}m")
    print(f"Visual range: {VISUAL_SENSING_RANGE_LIMITED}m")
    print(f"Formation radius: {formation_radius}m")
    
    # Calculate which agents will see target initially
    temp_seeing_count = 0
    for i in range(n_agents):
        angle = (2 * np.pi * i) / n_agents
        init_pos = agents_start_center + formation_radius * np.array([
            np.cos(angle), np.sin(angle), 0.0
        ])
        dist_to_target = np.linalg.norm(init_pos - current_target_pos)
        if dist_to_target < VISUAL_SENSING_RANGE_LIMITED:
            temp_seeing_count += 1
    
    print(f"Initial visibility: {temp_seeing_count}/{n_agents} agents within {VISUAL_SENSING_RANGE_LIMITED}m range")
    
    # Generate obstacles 
    if moving_target:
        obstacles = generate_random_spherical_obstacles(
            n_obstacles=3,
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
    
    for i in range(n_agents):
        angle = (2 * np.pi * i) / n_agents
        init_pos = agents_start_center + formation_radius * np.array([
            np.cos(angle), np.sin(angle), 0.0
        ])
        
        initial_target_estimate = np.array([999.0, 999.0, 999.0])
        
        agent = AsyncDroneAgent(
            id=i,
            state_3d=init_pos,
            target_pos_3d=initial_target_estimate,
            formation_radius=formation_radius,
            phase_lag=phase_lags_sec[i],
            async_alpha=consensus_alpha,
            Kp=0.5,
            Kd=1.2,
            dt=dt,
            max_velocity=5.0,
            max_acceleration=4.0
        )
        agent.target_pos = None
        
        agents.append(agent)
        print(f"Agent {i} initialized at: {init_pos}, phase_lag: {phase_lags_sec[i]*1000:.1f}ms")
    
    # Check initial visibility
    initial_seeing = sum(1 for a in agents 
                        if np.linalg.norm(a.position - current_target_pos) < VISUAL_SENSING_RANGE_LIMITED)
    print(f"\n→ {initial_seeing}/{n_agents} agents can initially see target")
    
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
        
        # 3D visualization objects
        drone_scatter = ax.scatter([], [], [], s=200, c='blue', marker='o',
                                  edgecolors='black', linewidths=2)
        
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
        
        # Add obstacles visualization
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
        ax.set_title('3D ASYNC Formation (GREEN=Direct Sensing, COLOR=Consensus Only)', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 30)
        ax.set_ylim(-15, 30)
        ax.set_zlim(0, 10)
        ax.grid(True, alpha=0.3)
        
        # 2D top view
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
            
            # Add obstacles to 2D view
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
            ax2d.set_title(f'2D Top View - ASYNC (Alpha={consensus_alpha:.2f}, Spread={phase_spread_ms:.0f}ms)',
                         fontsize=14, fontweight='bold')
            ax2d.axis('equal')
            ax2d.grid(True, alpha=0.3)
            ax2d.set_xlim(-15, 30)
            ax2d.set_ylim(-15, 30)
        
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes,
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # ==================== END ANIMATION SETUP ====================
    
    
    # Define initial target position for bootstrap
    if moving_target:
        true_target = target.position.copy()
    else:
        true_target = target_pos
    
    for agent in agents:
        msg = agent.msg()
        for other_agent in agents:
            if agent.id != other_agent.id:
                dist = np.linalg.norm(agent.position - other_agent.position)
                if dist <= COMM_RANGE:
                    other_agent.add_msg(msg)
    

    print(f"Running initial consensus to propagate target info...")
    for bootstrap_round in range(10):  # 10 rounds should propagate through network
        agents_by_phase = sorted(agents, key=lambda a: a.phase_lag)
        
        for agent in agents_by_phase:
            dist_to_target = np.linalg.norm(agent.position - true_target)
            neighbor_estimates = []
            
            # Add own sensor if available
            if dist_to_target < VISUAL_SENSING_RANGE_LIMITED:
                neighbor_estimates.append(true_target.copy())
            
            # Collect neighbor estimates
            for msg in agent.msgs:
                neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                if neighbor_target is not None:
                    target_arr = np.array(neighbor_target).ravel()
                    if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                        neighbor_estimates.append(target_arr)
            
            # Apply consensus
            if len(neighbor_estimates) > 0:
                consensus = np.mean(np.vstack(neighbor_estimates), axis=0)
                if np.all(np.isfinite(consensus)):
                    if agent.target_pos is None:
                        agent.target_pos = consensus.copy()
                    else:
                        if np.all(np.isfinite(agent.target_pos)):
                            agent.target_pos = (agent.async_alpha * consensus + 
                                               (1 - agent.async_alpha) * agent.target_pos)
            
            # Broadcast updated estimate
            msg = agent.msg()
            for other_agent in agents:
                if agent.id != other_agent.id:
                    dist = np.linalg.norm(agent.position - other_agent.position)
                    if dist <= COMM_RANGE:
                        other_agent.add_msg(msg)
        
        # Clear messages after round
        for agent in agents:
            agent.clear_msgs()
        
        # Check if all agents have target info
        agents_with_target = sum(1 for a in agents if a.target_pos is not None)
        print(f" round {bootstrap_round+1}: {agents_with_target}/{n_agents} agents have target info")
        if agents_with_target == n_agents:
            print(f" All agents have target info after {bootstrap_round+1} rounds")
            break
    
    # Final broadcast before main loop
    for agent in agents:
        msg = agent.msg()
        for other_agent in agents:
            if agent.id != other_agent.id:
                dist = np.linalg.norm(agent.position - other_agent.position)
                if dist <= COMM_RANGE:
                    other_agent.add_msg(msg)
    
    # Main simulation loop
    print(f"Starting Asynchronous simulation with consensus...")
    converged = False
    convergence_count = 0
    convergence_patience = 20
    cbf_activation_count = 0
    
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
        # STEP 1: ASYNCHRONOUS TARGET CONSENSUS (sequential by phase lag)
        # 
        # CRITICAL: Do NOT clear messages here! Agents need messages from 
        # previous iteration + fresh updates from earlier agents in THIS iteration.
        # 
        # Paper equation (Section 3.1):
        # x_i(n+1) = x_i(n) + α∑_{j∈N_i^+}(x_j(n+1)-x_i(n)) + α∑_{j∈N_i^-}(x_j(n)-x_i(n))
        # 
        # where N_i^+ = neighbors updating before agent i (have n+1 state in messages)
        #       N_i^- = neighbors updating after agent i (have n state in messages)
        # ====================================================================
        agents_by_phase = sorted(agents, key=lambda a: a.phase_lag)
        
        # Debug tracking for problematic iterations
        debug_this_iter = (iteration >= 340 and iteration <= 460 and iteration % 10 == 0)
        if debug_this_iter:
            print(f"\n{'='*70}")
            print(f"DEBUG: Iteration {iteration} - Detailed Consensus Tracking")
            print(f"{'='*70}")
        
        for agent in agents_by_phase:
            dist_to_target = np.linalg.norm(agent.position - true_target)
            
            old_target_pos = agent.target_pos.copy() if agent.target_pos is not None else None
            
            # Check if agent has direct sensing
            if dist_to_target < VISUAL_SENSING_RANGE_LIMITED:
                agent.target_pos = true_target.copy()
                agent.has_direct_sensing = True
                
                if debug_this_iter:
                    print(f"Agent {agent.id}: DIRECT SENSING (dist={dist_to_target:.2f}m)")
            else:
                agent.has_direct_sensing = False
                
                # Paper equation: x_i(n+1) = x_i(n) + α∑_{j∈N_i}(x_j - x_i(n))
                # Where N_i includes both N_i^+ and N_i^-
                
                if agent.target_pos is None:
                    # First time - initialize from any available neighbor
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).ravel()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                agent.target_pos = target_arr.copy()
                                if debug_this_iter:
                                    print(f"Agent {agent.id}: INITIALIZED from neighbor {neighbor_id}")
                                break
                else:
                    # Apply asynchronous consensus gradient update
                    # x_i(n+1) = x_i(n) + α·∑_j(x_j - x_i(n))
                    consensus_gradient = np.zeros(3)
                    num_neighbors = 0
                    neighbor_info = []
                    
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).ravel()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                # Accumulate gradient: (neighbor_estimate - my_estimate)
                                diff = target_arr - agent.target_pos
                                consensus_gradient += diff
                                num_neighbors += 1
                                if debug_this_iter:
                                    neighbor_info.append((neighbor_id, np.linalg.norm(diff)))
                    
                    if num_neighbors > 0:
                        # Apply update: x_i(n+1) = x_i(n) + α·∑(x_j - x_i(n))
                        new_estimate = agent.target_pos + agent.async_alpha * consensus_gradient
                        
                        # Debug: Check for invalid updates
                        if not np.all(np.isfinite(new_estimate)):
                            print(f"  Agent {agent.id} computed INVALID estimate at iter {iteration}")
                            print(f"    Old estimate: {agent.target_pos}")
                            print(f"    Gradient: {consensus_gradient}")
                            print(f"    Alpha: {agent.async_alpha}")
                            print(f"    Num neighbors: {num_neighbors}")
                        else:
                            update_magnitude = np.linalg.norm(new_estimate - agent.target_pos)
                            agent.target_pos = new_estimate
                            
                            if debug_this_iter:
                                print(f"Agent {agent.id}: CONSENSUS UPDATE")
                                print(f"  Old: {old_target_pos}")
                                print(f"  New: {agent.target_pos}")
                                print(f"  Update magnitude: {update_magnitude:.3f}m")
                                print(f"  Gradient norm: {np.linalg.norm(consensus_gradient):.3f}m")
                                print(f"  Num neighbors: {num_neighbors}")
                                print(f"  Neighbor diffs: {neighbor_info}")
                    else:
                        if debug_this_iter:
                            print(f"Agent {agent.id}: NO VALID NEIGHBORS (keeping old estimate)")
            
            # Log if target_pos is None after update
            if agent.target_pos is None and debug_this_iter:
                print(f"  Agent {agent.id}: target_pos is STILL None after update!")
            
            # Broadcast AFTER updating (critical for async!)
            msg = agent.msg()
            for other_agent in agents:
                if agent.id != other_agent.id:
                    dist = np.linalg.norm(agent.position - other_agent.position)
                    if dist <= COMM_RANGE:
                        other_agent.add_msg(msg)
        
        # ====================================================================
        # STEP 2: Clear messages AFTER all agents have updated
        # This ensures later agents receive fresh updates from earlier agents
        # ====================================================================
        for agent in agents:
            agent.clear_msgs()
        
        # ====================================================================
        # STEP 3: LOCAL OBSTACLE SENSING (DISTRIBUTED)
        # ====================================================================
        for agent in agents:
            agent.local_obstacles = []
            for obs in obstacles:
                dist_to_obs = np.linalg.norm(agent.position - obs['center'])
                if dist_to_obs <= VISUAL_SENSING_RANGE_LIMITED:
                    agent.local_obstacles.append(obs)
        
        # ====================================================================
        # STEP 4: Compute DESIRED accelerations (formation controller)
        # ====================================================================
        acc_desired = np.zeros((n_agents, 3))
        for i, agent in enumerate(agents):
            acc_desired[i] = agent.compute_desired_acceleration()
            
            if np.any(np.isnan(acc_desired[i])):
                print(f"WARNING: Agent {i} has NaN acceleration!")
                acc_desired[i] = np.zeros(3)
        
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
                
                if np.any(np.isnan(acc_safe[i])) or np.any(np.isinf(acc_safe[i])):
                    print(f"WARNING: CBF returned NaN/Inf for agent {i}")
                    acc_safe[i] = acc_desired[i]
                
                if not np.allclose(acc_safe[i], acc_desired[i], atol=1e-3):
                    cbf_activation_count += 1
                    agent.cbf_activations += 1
            
            positions = np.array([agent.position for agent in agents])
            is_safe, violations = cbf_filter.check_safety(positions, obstacles)
            if not is_safe:
                print(f"\nSAFETY VIOLATION at iteration {iteration}:")
                for v in violations:
                    print(f"   {v}")
        else:
            acc_safe = acc_desired
        
        # ====================================================================
        # STEP 6: Update dynamics
        # ====================================================================
        for i, agent in enumerate(agents):
            agent.update_dynamics(acc_safe[i])
        
        # ==================== CONVERGENCE CHECK ====================
        formation_errors = []
        for agent in agents:
            if agent.target_pos is None:
                formation_errors.append(999.0)
            else:
                formation_errors.append(agent.get_formation_error())
        
        avg_error = np.mean(formation_errors)
        max_error = np.max(formation_errors)
        
        # Debug: Track which agents have None target_pos and formation errors
        if debug_this_iter:
            print(f"\n--- Formation Error Breakdown ---")
            for i, agent in enumerate(agents):
                if agent.target_pos is None:
                    print(f"  Agent {agent.id}: target_pos=None (error=999.0m) ")
                else:
                    form_err = agent.get_formation_error()
                    print(f"  Agent {agent.id}: error={form_err:.3f}m, target_pos={agent.target_pos}")
            print(f"  AVG error: {avg_error:.3f}m, MAX error: {max_error:.3f}m")
            print(f"{'='*70}\n")
        
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
            
            agent_colors = []
            for agent in agents:
                if agent.has_direct_sensing:
                    agent_colors.append([0, 1, 0, 1])  
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
            
            # Graph overlay
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
            
            # Update 2D if enabled
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
                
                # CBF safety zones
                if use_cbf and FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                    for patch in list(ax2d.patches):
                        if isinstance(patch, plt.Circle):
                            patch.remove()
                    
                    agent_pos = current_positions[FOCUS_AGENT]
                    
                    sensing_circle = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.R_sense,
                                               fill=False, color='blue',
                                               linestyle=':', linewidth=2, alpha=0.4,
                                               label='Comm Range', zorder=3)
                    ax2d.add_patch(sensing_circle)
                    
                    safety_circle = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.d_safe,
                                              fill=True, color='red',
                                              alpha=0.15, zorder=2)
                    ax2d.add_patch(safety_circle)
                    
                    safety_circle_edge = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.d_safe,
                                                   fill=False, color='red',
                                                   linestyle='--', linewidth=2.5, alpha=0.7,
                                                   label='Safety Zone', zorder=3)
                    ax2d.add_patch(safety_circle_edge)
                
                # Graph overlay 2D
                if VIS_GRAPH_OVERLAY and use_cbf:
                    for line in graph_lines_2d:
                        line.remove()
                    graph_lines_2d.clear()
                    
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
                
                # Auto-scale view
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
            
            # Info text
            seeing_count = sum(1 for a in agents if a.has_direct_sensing)
            consensus_only = n_agents - seeing_count
            avg_acc = np.mean([np.linalg.norm(a.acceleration) for a in agents])
            info_text.set_text(
                f"Iteration: {iteration:04d}\n"
                f"Direct Sensing: {seeing_count}/{n_agents} (GREEN)\n"
                f"Consensus Only: {consensus_only}/{n_agents} (COLOR)\n"
                f"Alpha: {consensus_alpha:.2f}\n"
                f"Phase: 0-{phase_spread_ms:.0f}ms\n"
                f"==================\n"
                f"Center->Target: {center_to_target_error:.3f} m\n"
                f"Avg form error: {avg_error:.3f} m\n"
                f"Max form error: {max_error:.3f} m\n"
                f"Avg vel: {avg_vel:.3f} m/s\n"
                f"Max vel: {max_vel:.3f} m/s\n"
                f"Avg acc: {avg_acc:.3f} m/s^2\n"
                f"CBF activations: {cbf_activation_count}"
            )
            
            plt.pause(0.001)
            
            # Auto-scale 3D view
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
        print(f"  Total activations: {cbf_activation_count}")
    
    final_errors = []
    for agent in agents:
        if agent.target_pos is None:
            final_errors.append(999.0)
        else:
            final_errors.append(agent.get_formation_error())
    
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
    
    # Compute async benefit metric
    motion_end = int(15.0 / dt)
    avg_drift_motion = np.mean([agents[0].center_error_hist[i] for i in range(min(motion_end, len(agents[0].center_error_hist)))])
    print(f"\nAverage drift during target motion : {avg_drift_motion:.3f}m")
    
    cbf_stats = cbf_filter.get_statistics() if use_cbf else None
    return agents, target, cbf_filter, cbf_stats


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
        plt.title(f'GCBF Safety Functions for Agent {focus_agent} Over Time (ASYNC)', fontsize=14)
    else:
        for i in range(n):
            for j in range(i+1, n):
                h_vals = [
                    np.linalg.norm(agents[i].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{i},{j}")
        plt.title('All Pairwise GCBF Safety Functions Over Time (ASYNC)', fontsize=14)
    
    plt.axhline(0, color='k', linestyle='--', label='Safety boundary', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Barrier Function $h_{ij}(t)$', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_results_3d(agents, target=None):
    """Visualize 3D trajectories and final formation."""
    n_agents = len(agents)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: 3D Trajectories
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
    ax1.set_title('3D Drone Trajectories (ASYNC)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Formation
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
    
    # Plot 3: Formation Errors
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
    ax3.set_title('Formation Errors Over Time (ASYNC)',
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_random_spherical_obstacles(
        n_obstacles=3,
        target_start=np.array([0.0, 0.0, 0.0]),
        target_end=np.array([25.0, 25.0, 5.0]),
        min_radius=1.0,
        max_radius=2.0):
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


# ==================== MAIN ====================

if __name__ == "__main__":
    print("Starting Async Code")
    agents, target, cbf_filter, cbf_stats = run_async_formation_with_cbf(
        n_agents=NUM_AGENTS,
        max_iter=2000,
        dt=0.02,
        phase_spread_ms=50.0,  
        consensus_alpha=0.7,   #0.3,0.5, 0.7
        use_cbf=True,
        animate=True,
        moving_target=True
    )
    
    plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT)
    plot_results_3d(agents, target)
