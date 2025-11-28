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

# Visualization toggles
VIS_GRAPH_OVERLAY = True
VIS_2D_TOPVIEW = True
FOCUS_AGENT = 3
CONVERGENCE_THRESHOLD = 0.05
CONVERGENCE_VELOCITY = 0.01

VISUAL_SENSING_RANGE_LIMITED = 7.0  
COMM_RANGE = 10.0  


def compute_connected_components(agents, comm_range):
    """
    Compute connected components using BFS based on communication range.
    
    Returns:
        list of sets: Each set contains agent IDs in the same connected component
    """
    adjacency = {a.id: set() for a in agents}
    for i, agent_i in enumerate(agents):
        for j, agent_j in enumerate(agents):
            if i != j:
                dist = np.linalg.norm(agent_i.position - agent_j.position)
                if dist <= comm_range:
                    adjacency[agent_i.id].add(agent_j.id)
                    adjacency[agent_j.id].add(agent_i.id)
    
    visited = set()
    components = []
    
    for agent in agents:
        if agent.id not in visited:
            component = set()
            queue = [agent.id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)
    
    return components


def assign_formation_slots_and_centroids(agents, components, true_target=None):
    """
    Assign formation slots AND compute component centroid for each agent.
    
    Priority for centroid:
    1. Average of valid target_pos estimates in component
    2. true_target if provided and no valid estimates
    3. Average of agent positions as last resort
    """
    agent_dict = {a.id: a for a in agents}
    
    for component in components:
        sorted_ids = sorted(component)
        n_in_component = len(component)
        
        # Compute component centroid from agents' TARGET estimates
        valid_targets = []
        for agent_id in component:
            agent = agent_dict[agent_id]
            if agent.target_pos is not None:
                tp = np.array(agent.target_pos).flatten()
                if len(tp) == 3 and np.all(np.isfinite(tp)):
                    valid_targets.append(tp)
        
        if len(valid_targets) > 0:
            component_centroid = np.mean(valid_targets, axis=0)
        elif true_target is not None:
            # Use true target as fallback
            component_centroid = np.array(true_target).flatten()
        else:
            # Last resort: use position centroid
            positions = [agent_dict[aid].position for aid in component]
            component_centroid = np.mean(positions, axis=0)
        
        # Ensure it's a valid 3D array
        component_centroid = np.array(component_centroid).flatten()
        if len(component_centroid) != 3 or not np.all(np.isfinite(component_centroid)):
            # Ultimate fallback
            positions = [agent_dict[aid].position for aid in component]
            component_centroid = np.mean(positions, axis=0)
        
        # Assign to each agent in component
        for agent_id in component:
            agent = agent_dict[agent_id]
            agent.connected_agents = component
            agent.n_in_formation = n_in_component
            agent.formation_slot = sorted_ids.index(agent_id)
            agent.component_centroid = component_centroid.copy()


class AsyncDroneAgent(DroneAgent):
    """
    Asynchronous drone agent with dynamic formation based on connected components.
    """
    
    def __init__(self, id, state_3d, target_pos_3d, formation_radius=FORMATION_RADIUS,
                 phase_lag=0.0, async_alpha=0.3, Kp=0.5, Kd=1.2, dt=0.02, 
                 max_velocity=5.0, max_acceleration=4.0):
        super().__init__(id, state_3d, target_pos_3d, formation_radius,
                        Kp, Kd, dt, max_velocity, max_acceleration)
        self.phase_lag = phase_lag
        self.async_alpha = async_alpha
        self.cbf_activations = 0
        self.has_direct_sensing = False
        
        # Dynamic formation tracking
        self.connected_agents = set([id])
        self.formation_slot = 0
        self.n_in_formation = 1
        self.component_centroid = None
    
    def msg(self):
        """Return message with target estimate for consensus."""
        target_to_send = None
        if self.target_pos is not None:
            tp = np.array(self.target_pos).flatten()
            if len(tp) == 3 and np.all(np.isfinite(tp)):
                target_to_send = tp.copy()
        return (self.id, self.position.copy(), self.velocity.copy(), target_to_send)
    
    def compute_desired_acceleration(self):
        """
        Use component_centroid as formation center.
        """
        # Check if we have a valid centroid
        if self.component_centroid is None:
            formation_center = None
        else:
            fc = np.array(self.component_centroid).flatten()
            if len(fc) == 3 and np.all(np.isfinite(fc)):
                formation_center = fc
            else:
                formation_center = None
        
        if formation_center is None:
            # Hover in place
            velocity_error = -self.velocity
            acc = self.Kd * velocity_error
            acc_norm = np.linalg.norm(acc)
            if acc_norm > self.max_acceleration:
                acc = acc / acc_norm * self.max_acceleration
            return acc
        
        # Dynamic formation slot angle
        angle = (2 * np.pi * self.formation_slot) / self.n_in_formation
        
        ideal_position = formation_center + self.formation_radius * np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])
        
        position_error = ideal_position - self.position
        velocity_error = -self.velocity
        
        acc = self.Kp * position_error + self.Kd * velocity_error
        
        acc_norm = np.linalg.norm(acc)
        if acc_norm > self.max_acceleration:
            acc = acc / acc_norm * self.max_acceleration
        
        return acc
    
    def get_formation_error(self):
        """Error relative to component centroid and dynamic slot."""
        if self.component_centroid is None:
            return 999.0
        
        fc = np.array(self.component_centroid).flatten()
        if len(fc) != 3 or not np.all(np.isfinite(fc)):
            return 999.0
        
        angle = (2 * np.pi * self.formation_slot) / self.n_in_formation
        ideal_position = fc + self.formation_radius * np.array([
            np.cos(angle),
            np.sin(angle),
            0.0
        ])
        return np.linalg.norm(self.position - ideal_position)
    
    def get_target_tracking_error(self, true_target):
        """Error between agent's estimate and true target."""
        if self.target_pos is None:
            return 999.0
        tp = np.array(self.target_pos).flatten()
        if len(tp) != 3 or not np.all(np.isfinite(tp)):
            return 999.0
        return np.linalg.norm(tp - true_target)


def run_async_formation_with_cbf(n_agents=NUM_AGENTS, max_iter=2000, dt=0.02, 
                                  phase_spread_ms=16.0, consensus_alpha=0.3,
                                  use_cbf=True, animate=True, moving_target=True):
    """
    Run asynchronous formation control with dynamic component-based formations.
    """
    
    if use_cbf and not CBF_AVAILABLE:
        print("WARNING: CBF requested but not available")
        use_cbf = False
    
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
        R_min = cbf_filter.compute_minimum_sensing_radius(gamma=cbf_filter.alpha2, v_max=v_max)
        print(f"Minimum sensing radius for safety: {R_min:.2f}m")
    
    formation_radius = 5.0  
    
    if moving_target:
        target = MovingTarget(
            start_pos=np.array([0.0, 0.0, 0.0]),
            end_pos=np.array([25.0, 25.0, 5.0]),
            duration=15.0,
            dt=dt
        )
        print(f"Moving target: {target.start_pos} -> {target.end_pos} over {target.duration}s")
    else:
        target = None
        target_pos = np.array([0.0, 0.0, 2.0])
    
    phase_lags_sec = np.linspace(0, phase_spread_ms/1000, n_agents)
    
    agents = []
    np.random.seed(42)
    
    agents_start_center = np.array([7.0, 0.0, 2.0])  
    current_target_pos = target.position if moving_target else target_pos
    
    print(f"Formation center start: {agents_start_center}")
    print(f"Distance from target: {np.linalg.norm(agents_start_center - current_target_pos):.1f}m")
    
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
    
    print(f"\nGenerated {len(obstacles)} spherical obstacles")
    
    for i in range(n_agents):
        angle = (2 * np.pi * i) / n_agents
        init_pos = agents_start_center + formation_radius * np.array([
            np.cos(angle), np.sin(angle), 0.0
        ])
        
        agent = AsyncDroneAgent(
            id=i,
            state_3d=init_pos,
            target_pos_3d=np.array([999.0, 999.0, 999.0]),
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
        agent.component_centroid = None
        agents.append(agent)
        print(f"Agent {i} initialized at: {init_pos}, phase_lag: {phase_lags_sec[i]*1000:.1f}ms")
    
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
        
        target_scatter = ax.scatter(current_target_pos[0], current_target_pos[1], current_target_pos[2],
                                   c='red', marker='X', s=500, edgecolors='black', linewidths=2, label='Target')
        
        if moving_target:
            target_traj_line, = ax.plot([], [], [], 'r--', linewidth=2, alpha=0.5, label='Target Path')
        
        ideal_markers = []
        for i in range(n_agents):
            angle = (2 * np.pi * i) / n_agents
            ideal_pos = current_target_pos + formation_radius * np.array([np.cos(angle), np.sin(angle), 0.0])
            marker = ax.scatter(ideal_pos[0], ideal_pos[1], ideal_pos[2],
                              c='green', marker='x', s=200, alpha=0.5, linewidths=2)
            ideal_markers.append(marker)
        
        for obs in obstacles:
            sx, sy, sz = obs["center"]
            ax.scatter(sx, sy, sz, c="purple", s=150, marker="o", alpha=0.7, edgecolors="black", linewidths=2)
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
            line, = ax.plot([], [], [], color=colors[i], linewidth=1.5, alpha=0.6, label=f'Drone {i}')
            trajectory_lines.append(line)
        
        quiver = None
        graph_lines = []
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('3D ASYNC Formation (GREEN=Direct Sensing)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-15, 30)
        ax.set_ylim(-15, 30)
        ax.set_zlim(0, 10)
        ax.grid(True, alpha=0.3)
        
        if VIS_2D_TOPVIEW and ax2d is not None:
            x_init = [a.position[0] for a in agents]
            y_init = [a.position[1] for a in agents]
            
            drone_scatter_2d = ax2d.scatter(x_init, y_init, s=400, c=colors, marker='o',
                                           edgecolor='black', linewidths=1.5)
            
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
                                     alpha=0.5, linewidths=3, zorder=4)
                ideal_markers_2d.append(marker)
                label = ax2d.text(ideal_x, ideal_y + 0.7, f'Slot{i}', fontsize=9,
                                ha='center', color='orange', weight='bold', zorder=10)
                position_labels_2d.append(label)
            
            for obs in obstacles:
                sx, sy, sz = obs["center"]
                ax2d.scatter(sx, sy, c="purple", s=350, marker="o", alpha=0.4, edgecolors="black", linewidths=2)
                theta_2d = np.linspace(0, 2*np.pi, 40)
                circ_x = sx + obs["radius"] * np.cos(theta_2d)
                circ_y = sy + obs["radius"] * np.sin(theta_2d)
                ax2d.plot(circ_x, circ_y, color="purple", linestyle="--", alpha=0.6, linewidth=2)
            
            formation_circle_2d, = ax2d.plot(circle_x, circle_y, 'g--', alpha=0.3, linewidth=2, zorder=1)
            
            trajectory_lines_2d = []
            for i in range(n_agents):
                line, = ax2d.plot([], [], color=colors[i], linewidth=2, alpha=0.4, zorder=2)
                trajectory_lines_2d.append(line)
            
            graph_lines_2d = []
            
            drone_labels_2d = []
            for i in range(n_agents):
                label = ax2d.text(0, 0, f'D{i}', fontsize=10, ha='center', va='center',
                                color='white', weight='bold', zorder=15)
                drone_labels_2d.append(label)
            
            ax2d.set_xlabel('X (m)', fontsize=12)
            ax2d.set_ylabel('Y (m)', fontsize=12)
            ax2d.set_title(f'2D Top View - ASYNC (Alpha={consensus_alpha:.2f})', fontsize=14, fontweight='bold')
            ax2d.axis('equal')
            ax2d.grid(True, alpha=0.3)
            ax2d.set_xlim(-15, 30)
            ax2d.set_ylim(-15, 30)
        
        info_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Bootstrap consensus
    if moving_target:
        true_target = target.position.copy()
    else:
        true_target = target_pos
    
    # Initial message broadcast
    for agent in agents:
        msg = agent.msg()
        for other_agent in agents:
            if agent.id != other_agent.id:
                dist = np.linalg.norm(agent.position - other_agent.position)
                if dist <= COMM_RANGE:
                    other_agent.add_msg(msg)
    
    print(f"Running initial consensus...")
    for bootstrap_round in range(10):
        agents_by_phase = sorted(agents, key=lambda a: a.phase_lag)
        
        for agent in agents_by_phase:
            dist_to_target = np.linalg.norm(agent.position - true_target)
            neighbor_estimates = []
            
            if dist_to_target < VISUAL_SENSING_RANGE_LIMITED:
                neighbor_estimates.append(true_target.copy())
            
            for msg in agent.msgs:
                neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                if neighbor_target is not None:
                    target_arr = np.array(neighbor_target).flatten()
                    if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                        neighbor_estimates.append(target_arr)
            
            if len(neighbor_estimates) > 0:
                consensus = np.mean(neighbor_estimates, axis=0)
                if np.all(np.isfinite(consensus)):
                    if agent.target_pos is None:
                        agent.target_pos = consensus.copy()
                    else:
                        old_tp = np.array(agent.target_pos).flatten()
                        if len(old_tp) == 3 and np.all(np.isfinite(old_tp)):
                            agent.target_pos = (agent.async_alpha * consensus + 
                                               (1 - agent.async_alpha) * old_tp)
                        else:
                            agent.target_pos = consensus.copy()
            
            msg = agent.msg()
            for other_agent in agents:
                if agent.id != other_agent.id:
                    dist = np.linalg.norm(agent.position - other_agent.position)
                    if dist <= COMM_RANGE:
                        other_agent.add_msg(msg)
        
        for agent in agents:
            agent.clear_msgs()
        
        agents_with_target = sum(1 for a in agents if a.target_pos is not None)
        print(f" round {bootstrap_round+1}: {agents_with_target}/{n_agents} agents have target info")
        if agents_with_target == n_agents:
            break
    
    # Initial formation assignment
    components = compute_connected_components(agents, COMM_RANGE)
    assign_formation_slots_and_centroids(agents, components, true_target)
    
    # Broadcast for main loop
    for agent in agents:
        msg = agent.msg()
        for other_agent in agents:
            if agent.id != other_agent.id:
                dist = np.linalg.norm(agent.position - other_agent.position)
                if dist <= COMM_RANGE:
                    other_agent.add_msg(msg)
    
    # Main loop
    print(f"Starting main simulation...")
    converged = False
    convergence_count = 0
    convergence_patience = 20
    cbf_activation_count = 0
    
    for iteration in range(max_iter):
        # Update target
        if moving_target:
            target.update()
            true_target = target.position.copy()
        else:
            true_target = target_pos
        
        # Compute connected components
        components = compute_connected_components(agents, COMM_RANGE)
        
        # Async consensus
        agents_by_phase = sorted(agents, key=lambda a: a.phase_lag)
        
        for agent in agents_by_phase:
            dist_to_target = np.linalg.norm(agent.position - true_target)
            
            if dist_to_target < VISUAL_SENSING_RANGE_LIMITED:
                agent.target_pos = true_target.copy()
                agent.has_direct_sensing = True
            else:
                agent.has_direct_sensing = False
                
                if agent.target_pos is None:
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).flatten()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                agent.target_pos = target_arr.copy()
                                break
                else:
                    consensus_gradient = np.zeros(3)
                    num_neighbors = 0
                    
                    for msg in agent.msgs:
                        neighbor_id, neighbor_pos, neighbor_vel, neighbor_target = msg
                        if neighbor_target is not None:
                            target_arr = np.array(neighbor_target).flatten()
                            if len(target_arr) == 3 and np.all(np.isfinite(target_arr)):
                                old_tp = np.array(agent.target_pos).flatten()
                                if len(old_tp) == 3 and np.all(np.isfinite(old_tp)):
                                    diff = target_arr - old_tp
                                    consensus_gradient += diff
                                    num_neighbors += 1
                    
                    if num_neighbors > 0:
                        old_tp = np.array(agent.target_pos).flatten()
                        if len(old_tp) == 3 and np.all(np.isfinite(old_tp)):
                            new_estimate = old_tp + agent.async_alpha * consensus_gradient
                            if np.all(np.isfinite(new_estimate)):
                                agent.target_pos = new_estimate
            
            # Broadcast after updating
            msg = agent.msg()
            for other_agent in agents:
                if agent.id != other_agent.id:
                    dist = np.linalg.norm(agent.position - other_agent.position)
                    if dist <= COMM_RANGE:
                        other_agent.add_msg(msg)
        
        # Clear messages
        for agent in agents:
            agent.clear_msgs()
        
        # Assign formation slots and centroids AFTER consensus
        assign_formation_slots_and_centroids(agents, components, true_target)
        
        # Obstacle sensing
        for agent in agents:
            agent.local_obstacles = []
            for obs in obstacles:
                dist_to_obs = np.linalg.norm(agent.position - obs['center'])
                if dist_to_obs <= VISUAL_SENSING_RANGE_LIMITED:
                    agent.local_obstacles.append(obs)
        
        # Compute accelerations
        acc_desired = np.zeros((n_agents, 3))
        for i, agent in enumerate(agents):
            acc_desired[i] = agent.compute_desired_acceleration()
            if np.any(np.isnan(acc_desired[i])):
                acc_desired[i] = np.zeros(3)
        
        # CBF filtering
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
        
        # Update dynamics
        for i, agent in enumerate(agents):
            agent.update_dynamics(acc_safe[i])
        
        # Compute errors
        formation_errors = [agent.get_formation_error() for agent in agents]
        tracking_errors = [agent.get_target_tracking_error(true_target) for agent in agents]
        
        valid_formation_errors = [e for e in formation_errors if e < 900]
        valid_tracking_errors = [e for e in tracking_errors if e < 900]
        
        if len(valid_formation_errors) > 0:
            avg_form_error = np.mean(valid_formation_errors)
            max_form_error = np.max(valid_formation_errors)
        else:
            avg_form_error = 999.0
            max_form_error = 999.0
        
        if len(valid_tracking_errors) > 0:
            avg_track_error = np.mean(valid_tracking_errors)
            max_track_error = np.max(valid_tracking_errors)
        else:
            avg_track_error = 999.0
            max_track_error = 999.0
        
        avg_vel = np.mean([np.linalg.norm(a.velocity) for a in agents])
        max_vel = np.max([np.linalg.norm(a.velocity) for a in agents])
        
        formation_center = np.mean([agent.position for agent in agents], axis=0)
        center_to_target_error = np.linalg.norm(formation_center - true_target)
        
        if not hasattr(agents[0], 'center_error_hist'):
            for agent in agents:
                agent.center_error_hist = []
        for agent in agents:
            agent.center_error_hist.append(center_to_target_error)
        
        target_stopped = (np.linalg.norm(target.velocity) < 0.01) if moving_target else True
        
        if target_stopped and max_form_error < CONVERGENCE_THRESHOLD and max_vel < CONVERGENCE_VELOCITY:
            convergence_count += 1
            if convergence_count >= convergence_patience:
                converged = True
                print(f"\nFORMATION CONVERGED at iteration {iteration}")
                print(f"   Max formation error: {max_form_error:.3f}m")
                if not animate:
                    break
        else:
            convergence_count = 0
        
        # Animation update
        if animate and iteration % 2 == 0:
            current_positions = np.array([agent.position for agent in agents])
            drone_scatter._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_positions[:, 2])
            
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
                
                largest_component = max(components, key=len)
                n_in_largest = len(largest_component)
                
                agent_in_largest = next(a for a in agents if a.id in largest_component)
                largest_centroid = agent_in_largest.component_centroid
                if largest_centroid is None or not np.all(np.isfinite(largest_centroid)):
                    largest_centroid = current_target_pos
                
                for i, marker in enumerate(ideal_markers):
                    if i < n_in_largest:
                        angle = (2 * np.pi * i) / n_in_largest
                        ideal_pos = largest_centroid + formation_radius * np.array([np.cos(angle), np.sin(angle), 0.0])
                        marker._offsets3d = ([ideal_pos[0]], [ideal_pos[1]], [ideal_pos[2]])
                        marker.set_alpha(0.5)
                    else:
                        marker._offsets3d = ([999], [999], [999])
                        marker.set_alpha(0.0)
                
                circle_x = largest_centroid[0] + formation_radius * np.cos(theta)
                circle_y = largest_centroid[1] + formation_radius * np.sin(theta)
                circle_z = np.ones_like(theta) * largest_centroid[2]
                formation_circle.set_data(circle_x, circle_y)
                formation_circle.set_3d_properties(circle_z)
            
            if quiver is not None:
                quiver.remove()
            
            current_velocities = np.array([agent.velocity for agent in agents])
            vel_scale = 0.5
            quiver = ax.quiver(current_positions[:, 0], current_positions[:, 1], current_positions[:, 2],
                              current_velocities[:, 0]*vel_scale, current_velocities[:, 1]*vel_scale,
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
                        if i < n_in_largest:
                            angle = (2 * np.pi * i) / n_in_largest
                            ideal_x = largest_centroid[0] + formation_radius * np.cos(angle)
                            ideal_y = largest_centroid[1] + formation_radius * np.sin(angle)
                            marker.set_offsets([[ideal_x, ideal_y]])
                            marker.set_alpha(0.5)
                            position_labels_2d[i].set_position((ideal_x, ideal_y + 0.7))
                            position_labels_2d[i].set_text(f'Slot{i}')
                            position_labels_2d[i].set_alpha(1.0)
                        else:
                            marker.set_offsets([[999, 999]])
                            marker.set_alpha(0.0)
                            position_labels_2d[i].set_alpha(0.0)
                    
                    formation_circle_2d.set_data(circle_x, circle_y)
                
                if use_cbf and FOCUS_AGENT >= 0 and FOCUS_AGENT < n_agents:
                    for patch in list(ax2d.patches):
                        if isinstance(patch, plt.Circle):
                            patch.remove()
                    
                    agent_pos = current_positions[FOCUS_AGENT]
                    
                    sensing_circle = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.R_sense,
                                               fill=False, color='blue', linestyle=':', linewidth=2, alpha=0.4, zorder=3)
                    ax2d.add_patch(sensing_circle)
                    
                    safety_circle = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.d_safe,
                                              fill=True, color='red', alpha=0.15, zorder=2)
                    ax2d.add_patch(safety_circle)
                    
                    safety_circle_edge = plt.Circle((agent_pos[0], agent_pos[1]), cbf_filter.d_safe,
                                                   fill=False, color='red', linestyle='--', linewidth=2.5, alpha=0.7, zorder=3)
                    ax2d.add_patch(safety_circle_edge)
                
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
            n_components = len(components)
            largest_size = max(len(c) for c in components)
            n_valid = len(valid_formation_errors)
            
            info_text.set_text(
                f"Iteration: {iteration:04d}\n"
                f"Direct Sensing: {seeing_count}/{n_agents}\n"
                f"Alpha: {consensus_alpha:.2f}\n"
                f"==================\n"
                f"Components: {n_components} (largest: {largest_size})\n"
                f"Center->Target: {center_to_target_error:.3f} m\n"
                f"Form err ({n_valid} valid):\n"
                f"  avg={avg_form_error:.3f}m\n"
                f"  max={max_form_error:.3f}m\n"
                f"Track err: avg={avg_track_error:.3f}m\n"
                f"Vel: avg={avg_vel:.3f}, max={max_vel:.3f}\n"
                f"CBF: {cbf_activation_count}"
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
        
        if iteration % 50 == 0:
            seeing_count = sum(1 for a in agents if a.has_direct_sensing)
            comp_sizes = [len(c) for c in components]
            
            # Debug: check centroids
            centroid_status = []
            for a in agents:
                if a.component_centroid is None:
                    centroid_status.append("None")
                elif not np.all(np.isfinite(a.component_centroid)):
                    centroid_status.append("Invalid")
                else:
                    centroid_status.append("OK")
            
            print(f"Iter {iteration:3d}: Center->Target: {center_to_target_error:.3f}m, "
                  f"Form: avg={avg_form_error:.3f}m max={max_form_error:.3f}m ({len(valid_formation_errors)} valid), "
                  f"Track: {avg_track_error:.3f}m, Comps: {comp_sizes}, Centroids: {centroid_status}")
        
        if converged and not animate:
            break
    
    if animate:
        plt.ioff()
        plt.show()
    
    print(f"\n{'='*60}")
    print("SIMULATION COMPLETE")
    print(f"{'='*60}")
    
    if use_cbf:
        stats = cbf_filter.get_statistics()
        print("\nCBF Statistics:")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.3f}")
            else:
                print(f"  {key}: {val}")
    
    print(f"\nFinal per-component formation:")
    for comp in components:
        sorted_comp = sorted(comp)
        print(f"  Component {sorted_comp}:")
        for agent_id in sorted_comp:
            agent = next(a for a in agents if a.id == agent_id)
            print(f"    Agent {agent_id}: form_err={agent.get_formation_error():.3f}m, "
                  f"track_err={agent.get_target_tracking_error(true_target):.3f}m, "
                  f"slot={agent.formation_slot}/{agent.n_in_formation}")
    
    print(f"\n  Overall: Form avg={avg_form_error:.3f}m max={max_form_error:.3f}m ({len(valid_formation_errors)}/{n_agents} valid)")
    
    if converged:
        print(f"\nConverged after {len(agents[0].position_hist)-1} iterations")
    else:
        print(f"\nDid not converge within {max_iter} iterations")
    
    return agents, target, cbf_filter, cbf_filter.get_statistics() if use_cbf else None, components


def plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT):
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
        plt.title(f'GCBF Safety Functions for Agent {focus_agent}', fontsize=14)
    else:
        for i in range(n):
            for j in range(i+1, n):
                h_vals = [
                    np.linalg.norm(agents[i].position_hist[k] - agents[j].position_hist[k])**2 - cbf_filter.d_safe**2
                    for k in range(T)
                ]
                plt.plot(time, h_vals, label=f"h_{i},{j}")
        plt.title('All Pairwise GCBF Safety Functions', fontsize=14)
    
    plt.axhline(0, color='k', linestyle='--', label='Safety boundary', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Barrier Function $h_{ij}(t)$', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_results_3d(agents, target=None, components=None):
    n_agents = len(agents)
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, n_agents))
    
    for i, agent in enumerate(agents):
        traj = np.array(agent.position_hist)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], linewidth=2, label=f'Drone {agent.id}')
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=colors[i], marker='o', s=100, edgecolors='black')
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=colors[i], marker='*', s=200, edgecolors='black')
    
    if target is not None:
        traj_target = np.array(target.position_hist)
        ax1.plot(traj_target[:, 0], traj_target[:, 1], traj_target[:, 2], 'r--', linewidth=3, alpha=0.6, label='Target')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectories')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(132)
    theta = np.linspace(0, 2*np.pi, 100)
    
    for i, agent in enumerate(agents):
        ax2.plot(agent.position[0], agent.position[1], 'o', markersize=12, color=colors[i], label=f'D{agent.id}')
        
        formation_center = agent.component_centroid
        if formation_center is not None and np.all(np.isfinite(formation_center)):
            angle = (2 * np.pi * agent.formation_slot) / agent.n_in_formation
            ideal_x = formation_center[0] + agent.formation_radius * np.cos(angle)
            ideal_y = formation_center[1] + agent.formation_radius * np.sin(angle)
            ax2.plot(ideal_x, ideal_y, 'x', markersize=12, color=colors[i], markeredgewidth=3)
            ax2.plot([agent.position[0], ideal_x], [agent.position[1], ideal_y], '--', color=colors[i], alpha=0.5)
    
    if components is not None:
        comp_colors = ['green', 'blue', 'orange', 'purple', 'cyan']
        for idx, comp in enumerate(components):
            agent_in_comp = next(a for a in agents if a.id in comp)
            centroid = agent_in_comp.component_centroid
            if centroid is not None and np.all(np.isfinite(centroid)):
                circle_x = centroid[0] + agents[0].formation_radius * np.cos(theta)
                circle_y = centroid[1] + agents[0].formation_radius * np.sin(theta)
                ax2.plot(circle_x, circle_y, '--', color=comp_colors[idx % len(comp_colors)], alpha=0.3, linewidth=2)
                ax2.plot(centroid[0], centroid[1], '+', markersize=15, color=comp_colors[idx % len(comp_colors)], markeredgewidth=3)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Final Formation\no=actual, x=ideal')
    ax2.axis('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(133)
    if hasattr(agents[0], 'center_error_hist'):
        time = np.arange(len(agents[0].center_error_hist)) * agents[0].dt
        ax3.plot(time, agents[0].center_error_hist, 'b-', linewidth=2, label='Center->Target')
    ax3.axhline(y=CONVERGENCE_THRESHOLD, color='r', linestyle=':', linewidth=2, label=f'Threshold ({CONVERGENCE_THRESHOLD}m)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Formation Center Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_random_spherical_obstacles(n_obstacles=3, target_start=np.array([0.0, 0.0, 0.0]),
                                        target_end=np.array([25.0, 25.0, 5.0]),
                                        min_radius=1.0, max_radius=2.0):
    obstacles = []
    for _ in range(n_obstacles):
        t = np.random.uniform(0.1, 0.9)
        base = target_start + t * (target_end - target_start)
        offset = np.random.uniform(-10, 10, 3)
        offset[2] = np.random.uniform(-2, 2)
        center = base + offset
        radius = np.random.uniform(min_radius, max_radius)
        obstacles.append({"center": center, "radius": radius})
    return obstacles


if __name__ == "__main__":
    print("Starting Async Formation with DYNAMIC COMPONENT-BASED FORMATIONS")
    agents, target, cbf_filter, cbf_stats, components = run_async_formation_with_cbf(
        n_agents=NUM_AGENTS,
        max_iter=2000,
        dt=0.02,
        phase_spread_ms=50.0,
        consensus_alpha=0.7,
        use_cbf=True,
        animate=True,
        moving_target=True
    )
    
    plot_barrier_functions(agents, cbf_filter, focus_agent=FOCUS_AGENT)
    plot_results_3d(agents, target, components)