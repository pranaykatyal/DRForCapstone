# DecentralizedGCBF.py - Distributed GCBF Implementation
"""
Distributed Control Barrier Function (CBF) implementation for multi-drone safety.

Key Features:
- Each drone solves LOCAL QP using only neighbor information
- Conservative neighbor assumption (a_neighbor = 0)
- No central coordinator needed
- Scalable to large swarms

Based on:
- Borrmann et al. 2015: "Control Barrier Certificates for Safe Swarm Behavior"
- GCBF+ (Zhang et al.) 2025: "Graph Control Barrier Function Framework"
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple

class GraphCBFSafetyFilter:
    """
    Distributed graph-based CBF implementation for multi-drone collision avoidance.
    
    Key Concepts:
    - Uses sensing radius R to define neighbor graph (scalable)
    - Second-order CBF for double-integrator dynamics
    - DISTRIBUTED: Each drone solves local QP independently
    - Conservative: Assumes neighbors maintain constant velocity
    
    Theory:
    For barrier function h(x) where safe set is {x | h(x) ≥ 0}:
    - Second-order CBF: ḧ + α₁·ḣ + α₂·h ≥ 0
    - This guarantees forward invariance of safe set
    
    For collision avoidance: h_ij(x) = ||p_i - p_j||² - d_safe²
    """
    
    def __init__(self, 
                 n_drones: int,
                 safety_distance: float = 1.0,
                 sensing_radius: float = 5.0,
                 obstacle_margin: float = 0.5,
                 alpha1: float = 2.0,
                 alpha2: float = 1.0,
                 max_acceleration: float = 5.0):
        """
        Initialize Graph-Based CBF safety filter.
        
        Args:
            n_drones: Number of drones in system (for centralized comparison only)
            safety_distance: Minimum distance between drones (meters)
            sensing_radius: Range for neighbor detection (meters)
            obstacle_margin: Safety margin around obstacles (meters)
            alpha1: CBF parameter for ḣ term (controls responsiveness)
            alpha2: CBF parameter for h term (controls convergence rate)
            max_acceleration: Maximum allowed acceleration magnitude (m/s²)
        
        Notes:
            - sensing_radius should be > safety_distance for proper operation
            - Recommended: sensing_radius ≥ compute_minimum_sensing_radius()
            - alpha1, alpha2 > 0 for stability
        """
        self.n_drones = n_drones
        self.d_safe = safety_distance
        self.R_sense = sensing_radius
        self.obs_margin = obstacle_margin
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.a_max = max_acceleration
        
        # Validate parameters
        if self.R_sense <= self.d_safe:
            print(f"WARNING: sensing_radius ({self.R_sense}m) should be > "
                  f"safety_distance ({self.d_safe}m) for proper operation")
        
        # Statistics tracking
        self.qp_solve_times = []
        self.constraint_violations = 0
        self.cbf_activations = 0  # How many times CBF modified control
        
    # ========================================================================
    # DISTRIBUTED IMPLEMENTATION (New - Primary Method)
    # ========================================================================
    
    def filter_acceleration_single_drone(self,
                                        my_position: np.ndarray,
                                        my_velocity: np.ndarray,
                                        my_acc_desired: np.ndarray,
                                        neighbor_positions: List[np.ndarray],
                                        neighbor_velocities: List[np.ndarray],
                                        obstacles: Optional[List[dict]] = None) -> np.ndarray:
        """
        DISTRIBUTED: Solve local CBF-QP for a single drone using only neighbor information.
        
        This is the key difference from centralized approach:
        - Only uses information from neighbors within communication range
        - Solves small local QP (3 variables) instead of global QP (3*n_drones variables)
        - Can run in parallel on each drone
        - Conservative: assumes neighbors maintain constant velocity
        
        Args:
            my_position: (3,) My current position [x, y, z]
            my_velocity: (3,) My current velocity [vx, vy, vz]
            my_acc_desired: (3,) My desired acceleration from formation controller
            neighbor_positions: List of neighbor positions (within comm range)
            neighbor_velocities: List of neighbor velocities (within comm range)
            obstacles: List of obstacle dicts (locally sensed or shared)
        
        Returns:
            (3,) Safe acceleration for this drone only
        
        Example:
            # Drone 0 receives messages from neighbors
            neighbor_pos = [agent1.position, agent2.position]
            neighbor_vel = [agent1.velocity, agent2.velocity]
            
            # Solve local QP
            acc_safe = cbf.filter_acceleration_single_drone(
                my_position=agent0.position,
                my_velocity=agent0.velocity,
                my_acc_desired=acc_desired[0],
                neighbor_positions=neighbor_pos,
                neighbor_velocities=neighbor_vel,
                obstacles=local_obstacles
            )
        """
        import time
        t_start = time.time()
        
        # Decision variable: acceleration for THIS drone only (not all drones!)
        a = cp.Variable(3)  # ← Just 3D vector, not (n_drones, 3) matrix!
        
        # Objective: minimize deviation from my desired acceleration
        cost = cp.sum_squares(a - my_acc_desired)
        
        constraints = []
        num_safety_constraints = 0
        
        # 1. DISTRIBUTED inter-drone constraints (only with neighbors I can communicate with)
        for neighbor_pos, neighbor_vel in zip(neighbor_positions, neighbor_velocities):
            cbf_constraint = self._inter_drone_cbf_constraint_distributed(
                my_position, my_velocity, a,
                neighbor_pos, neighbor_vel
            )
            constraints.append(cbf_constraint)
            num_safety_constraints += 1
        
        # 2. Obstacle constraints (locally sensed)
        if obstacles is not None:
            for obs in obstacles:
                obs_constraint = self._obstacle_cbf_constraint_distributed(
                    my_position, my_velocity, a, obs
                )
                constraints.append(obs_constraint)
                num_safety_constraints += 1
        
        # 3. Acceleration limits
        constraints.append(a <= self.a_max)
        constraints.append(a >= -self.a_max)
        
        # Solve local QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        except Exception as e:
            # Solver failed - return desired acceleration (fallback)
            return my_acc_desired
        
        # Track solve time
        t_solve = time.time() - t_start
        self.qp_solve_times.append(t_solve)
        
        # Check if solution is valid
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            self.constraint_violations += 1
            return my_acc_desired
        
        # Check if CBF modified control
        deviation = np.linalg.norm(a.value - my_acc_desired)
        if deviation > 1e-3:
            self.cbf_activations += 1
        
        return a.value
    
    def _inter_drone_cbf_constraint_distributed(self,
                                               my_position: np.ndarray,
                                               my_velocity: np.ndarray,
                                               my_acc: cp.Variable,
                                               neighbor_position: np.ndarray,
                                               neighbor_velocity: np.ndarray) -> cp.Expression:
        """
        DISTRIBUTED: Generate CBF constraint between me and ONE neighbor.
        
        Key difference from centralized:
        - Assumes neighbor's acceleration is unknown/uncontrolled
        - Conservative: treats neighbor as maintaining constant velocity (a_neighbor = 0)
        - Only my acceleration (my_acc) is decision variable
        
        From Borrmann 2015 Section IV-B:
        "Each agent can compute its own control based on local information 
        by assuming that neighboring agents will maintain constant velocity"
        
        Barrier function: h = ||p_me - p_neighbor||² - d_safe²
        
        Args:
            my_position: (3,) My position
            my_velocity: (3,) My velocity
            my_acc: cvxpy Variable (3,) - MY acceleration (decision variable)
            neighbor_position: (3,) Neighbor's position
            neighbor_velocity: (3,) Neighbor's velocity
        
        Returns:
            cvxpy constraint expression
        """
        # Relative states
        p_diff = my_position - neighbor_position
        v_diff = my_velocity - neighbor_velocity
        
        # Barrier function value
        h = np.dot(p_diff, p_diff) - self.d_safe**2
        
        # First derivative: ḣ = 2·Δp·Δv
        h_dot = 2 * np.dot(p_diff, v_diff)
        
        # Second derivative (CONSERVATIVE: assume a_neighbor = 0)
        # h_ddot = 2||Δv||² + 2·Δp·(a_me - a_neighbor)
        # With a_neighbor = 0:
        h_ddot = 2 * np.dot(v_diff, v_diff) + 2 * p_diff @ my_acc
        
        # CBF constraint: ḧ + α₁·ḣ + α₂·h ≥ 0
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def _obstacle_cbf_constraint_distributed(self,
                                            my_position: np.ndarray,
                                            my_velocity: np.ndarray,
                                            my_acc: cp.Variable,
                                            obstacle: dict) -> cp.Expression:
        """
        DISTRIBUTED: Generate CBF constraint for obstacle avoidance.
        
        Same as centralized version since obstacle is static.
        
        Args:
            my_position: (3,) My position
            my_velocity: (3,) My velocity
            my_acc: cvxpy Variable (3,)
            obstacle: dict with 'center' and 'radius'
        
        Returns:
            cvxpy constraint
        """
        p_obs = np.array(obstacle['center'])
        r_obs = obstacle['radius'] + self.obs_margin
        
        p_diff = my_position - p_obs
        h = np.dot(p_diff, p_diff) - r_obs**2
        
        # For static obstacle: v_obs = 0, a_obs = 0
        h_dot = 2 * np.dot(p_diff, my_velocity)
        h_ddot = 2 * np.dot(my_velocity, my_velocity) + 2 * p_diff @ my_acc
        
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    # ========================================================================
    # CENTRALIZED IMPLEMENTATION (For Comparison)
    # ========================================================================
    
    def get_neighbors(self, i: int, positions: np.ndarray) -> List[int]:
        """
        Get neighbors of drone i within sensing radius R.
        This defines the graph structure for GCBF.
        
        Args:
            i: Drone index
            positions: (n_drones, 3) array of positions
        
        Returns:
            List of neighbor indices (excluding i itself)
        """
        neighbors = []
        p_i = positions[i]
        
        for j in range(self.n_drones):
            if i == j:
                continue
            p_j = positions[j]
            dist = np.linalg.norm(p_i - p_j)
            
            if dist <= self.R_sense:
                neighbors.append(j)
                
        return neighbors
    
    def filter_accelerations(self,
                            positions: np.ndarray,
                            velocities: np.ndarray,
                            acc_desired: np.ndarray,
                            obstacles: Optional[List[dict]] = None) -> np.ndarray:
        """
        CENTRALIZED: Filter desired accelerations through CBF-QP for safety.
        
        This method is kept for comparison with distributed approach.
        Uses global information about all drones.
        
        Args:
            positions: (n_drones, 3) current positions
            velocities: (n_drones, 3) current velocities
            acc_desired: (n_drones, 3) desired accelerations
            obstacles: List of obstacle dicts
        
        Returns:
            (n_drones, 3) safe accelerations
        """
        import time
        t_start = time.time()
        
        # Decision variables: acceleration for each drone
        a = cp.Variable((self.n_drones, 3))
        
        # Objective: minimize deviation from desired
        cost = cp.sum_squares(a - acc_desired)
        
        constraints = []
        num_safety_constraints = 0
        
        # 1. Inter-drone constraints (graph-based)
        for i in range(self.n_drones):
            neighbors_i = self.get_neighbors(i, positions)
            
            for j in neighbors_i:
                if j > i:  # Avoid duplicates
                    cbf_constraint = self._inter_drone_cbf_constraint(
                        i, j, positions, velocities, a
                    )
                    constraints.append(cbf_constraint)
                    num_safety_constraints += 1
        
        # 2. Obstacle constraints
        if obstacles is not None:
            for i in range(self.n_drones):
                for obs in obstacles:
                    obs_constraint = self._obstacle_cbf_constraint(
                        i, positions, velocities, a, obs
                    )
                    constraints.append(obs_constraint)
                    num_safety_constraints += 1
        
        # 3. Acceleration limits
        for i in range(self.n_drones):
            constraints.append(a[i, :] <= self.a_max)
            constraints.append(a[i, :] >= -self.a_max)
        
        # Solve QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        except Exception as e:
            print(f"QP Solver Error: {e}")
            return acc_desired
        
        t_solve = time.time() - t_start
        self.qp_solve_times.append(t_solve)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            self.constraint_violations += 1
            return acc_desired
        
        deviation = np.linalg.norm(a.value - acc_desired)
        if deviation > 1e-3:
            self.cbf_activations += 1
        
        return a.value
    
    def _inter_drone_cbf_constraint(self, 
                                   i: int, 
                                   j: int,
                                   positions: np.ndarray,
                                   velocities: np.ndarray,
                                   a: cp.Variable) -> cp.Expression:
        """
        CENTRALIZED: Generate second-order CBF constraint for drones i and j.
        """
        p_i, p_j = positions[i], positions[j]
        v_i, v_j = velocities[i], velocities[j]
        
        p_diff = p_i - p_j
        h = np.dot(p_diff, p_diff) - self.d_safe**2
        
        v_diff = v_i - v_j
        h_dot = 2 * np.dot(p_diff, v_diff)
        
        h_ddot = 2 * np.dot(v_diff, v_diff) + 2 * p_diff @ (a[i, :] - a[j, :])
        
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def _obstacle_cbf_constraint(self,
                                i: int,
                                positions: np.ndarray,
                                velocities: np.ndarray,
                                a: cp.Variable,
                                obstacle: dict) -> cp.Expression:
        """
        CENTRALIZED: Generate CBF constraint for drone i avoiding obstacle.
        """
        p_i = positions[i]
        v_i = velocities[i]
        
        p_obs = np.array(obstacle['center'])
        r_obs = obstacle['radius'] + self.obs_margin
        
        p_diff = p_i - p_obs
        h = np.dot(p_diff, p_diff) - r_obs**2
        
        h_dot = 2 * np.dot(p_diff, v_i)
        h_ddot = 2 * np.dot(v_i, v_i) + 2 * p_diff @ a[i, :]
        
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def check_safety(self, 
                    positions: np.ndarray, 
                    obstacles: Optional[List[dict]] = None) -> Tuple[bool, List[str]]:
        """
        Check if current configuration is safe (all barriers positive).
        
        Args:
            positions: (n_drones, 3) current positions
            obstacles: Optional list of obstacles
        
        Returns:
            is_safe: True if all barriers are positive (h ≥ 0)
            violations: List of violation descriptions
        """
        violations = []
        
        # Check inter-drone distances
        for i in range(self.n_drones):
            neighbors_i = self.get_neighbors(i, positions)
            for j in neighbors_i:
                if j > i:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.d_safe:
                        violations.append(
                            f"Drones {i} and {j} too close: "
                            f"{dist:.3f}m < {self.d_safe}m"
                        )
        
        # Check obstacle distances
        if obstacles is not None:
            for i in range(self.n_drones):
                for obs_idx, obs in enumerate(obstacles):
                    dist = np.linalg.norm(positions[i] - obs['center'])
                    min_dist = obs['radius'] + self.obs_margin
                    if dist < min_dist:
                        violations.append(
                            f"Drone {i} too close to obstacle {obs_idx}: "
                            f"{dist:.3f}m < {min_dist:.3f}m"
                        )
        
        return len(violations) == 0, violations
    
    def compute_minimum_sensing_radius(self, gamma: float, v_max: float) -> float:
        """
        Compute minimum sensing radius for safety guarantees.
        
        From Borrmann et al. 2015 Eq. (14):
            D_N = D_s + (1/(2Δa_max)) * (√(2Δa_max/γ) + Δv_max)²
        
        Args:
            gamma: CBF decay rate (use alpha2)
            v_max: Maximum speed of a single drone (m/s)
        
        Returns:
            Minimum sensing radius D_N in meters
        """
        delta_a_max = 2 * self.a_max
        delta_v_max = 2 * v_max
        
        D_N = self.d_safe + (1/(2*delta_a_max)) * \
            (np.sqrt(2*delta_a_max/gamma) + delta_v_max)**2
        
        return D_N
    
    def get_statistics(self) -> dict:
        """Get CBF performance statistics for analysis."""
        if not self.qp_solve_times:
            return {"message": "No QP solves yet"}
        
        total_solves = len(self.qp_solve_times)
        
        return {
            "mean_solve_time_ms": np.mean(self.qp_solve_times) * 1000,
            "max_solve_time_ms": np.max(self.qp_solve_times) * 1000,
            "total_solves": total_solves,
            "cbf_activations": self.cbf_activations,
            "activation_rate": self.cbf_activations / total_solves if total_solves > 0 else 0,
            "constraint_violations": self.constraint_violations
        }


# ================== TESTING ==================

if __name__ == "__main__":
    """
    Test both centralized and distributed implementations.
    """
    
    print("="*70)
    print("TESTING DISTRIBUTED vs CENTRALIZED GCBF")
    print("="*70)
    
    # Setup
    cbf = GraphCBFSafetyFilter(
        n_drones=3,
        safety_distance=2.0,
        sensing_radius=8.0,
        obstacle_margin=0.5,
        alpha1=3.0,
        alpha2=2.5,
        max_acceleration=5.0
    )
    
    # Scenario: Three drones, two on collision course
    positions = np.array([
        [0.0, 0.0, 1.0],   # Drone 0
        [3.0, 0.0, 1.0],   # Drone 1 (heading toward Drone 0)
        [0.0, 5.0, 1.0]    # Drone 2 (safe distance)
    ])
    
    velocities = np.array([
        [1.0, 0.0, 0.0],   # Drone 0: moving right
        [-1.0, 0.0, 0.0],  # Drone 1: moving left (collision course!)
        [0.0, 0.0, 0.0]    # Drone 2: stationary
    ])
    
    # Formation controller wants to accelerate (bad!)
    acc_desired = np.array([
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Add an obstacle
    obstacles = [{'center': np.array([1.5, 0.0, 1.0]), 'radius': 0.5}]
    
    print("\n--- INITIAL CONDITIONS ---")
    print(f"Positions:\n{positions}")
    print(f"Velocities:\n{velocities}")
    print(f"Desired accelerations:\n{acc_desired}")
    print(f"Obstacles: {obstacles}")
    
    is_safe, violations = cbf.check_safety(positions, obstacles)
    print(f"\nInitially safe: {is_safe}")
    if violations:
        for v in violations:
            print(f"  - {v}")
    
    # ========== TEST 1: CENTRALIZED ===========
    print("\n" + "="*70)
    print("TEST 1: CENTRALIZED APPROACH")
    print("="*70)
    
    acc_safe_centralized = cbf.filter_accelerations(
        positions, velocities, acc_desired, obstacles
    )
    
    print("\nCentralized safe accelerations:")
    print(acc_safe_centralized)
    print("\nDeviation from desired:")
    for i in range(3):
        dev = np.linalg.norm(acc_safe_centralized[i] - acc_desired[i])
        print(f"  Drone {i}: {dev:.4f} m/s²")
    
    # ========== TEST 2: DISTRIBUTED ===========
    print("\n" + "="*70)
    print("TEST 2: DISTRIBUTED APPROACH")
    print("="*70)
    
    acc_safe_distributed = np.zeros((3, 3))
    
    for i in range(3):
        # Simulate each drone gathering neighbor info
        neighbor_positions = []
        neighbor_velocities = []
        
        for j in range(3):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= cbf.R_sense:  # Within communication range
                    neighbor_positions.append(positions[j])
                    neighbor_velocities.append(velocities[j])
        
        # Solve local QP
        acc_safe_distributed[i] = cbf.filter_acceleration_single_drone(
            my_position=positions[i],
            my_velocity=velocities[i],
            my_acc_desired=acc_desired[i],
            neighbor_positions=neighbor_positions,
            neighbor_velocities=neighbor_velocities,
            obstacles=obstacles
        )
    
    print("\nDistributed safe accelerations:")
    print(acc_safe_distributed)
    print("\nDeviation from desired:")
    for i in range(3):
        dev = np.linalg.norm(acc_safe_distributed[i] - acc_desired[i])
        print(f"  Drone {i}: {dev:.4f} m/s²")
    
    # ========== COMPARISON ===========
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print("\nDifference between centralized and distributed:")
    for i in range(3):
        diff = np.linalg.norm(acc_safe_centralized[i] - acc_safe_distributed[i])
        print(f"  Drone {i}: {diff:.4f} m/s²")
    
    print("\nNote: Distributed is more conservative (larger deviation from desired)")
    print("      because it assumes neighbors maintain constant velocity.")
    
    # Statistics
    print("\n" + "="*70)
    stats = cbf.get_statistics()
    print("CBF Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
        else:
            print(f"  {key}: {val}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)