# Centralized
"""
Fixed Control Barrier Function (CBF) implementation for multi-drone safety.
This version merges CBFSafetyFilter and GraphCBFSafetyFilter into one working class.

Based on:
- Borrmann et al. 2015: "Control Barrier Certificates for Safe Swarm Behavior"
- GCBF+ (Zhang et al.) 2025: "Graph Control Barrier Function Framework"
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple

class GraphCBFSafetyFilter:
    """
    Complete graph-based CBF implementation for multi-drone collision avoidance.
    
    Key Concepts:
    - Uses sensing radius R to define neighbor graph (scalable)
    - Second-order CBF for double-integrator dynamics
    - QP minimizes deviation from nominal control while ensuring safety
    - Works without neural networks (suitable for small drone swarms)
    
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
            n_drones: Number of drones in system
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
        
    def get_neighbors(self, i: int, positions: np.ndarray) -> List[int]:
        """
        Get neighbors of drone i within sensing radius R.
        This defines the graph structure for GCBF.
        
        Args:
            i: Drone index
            positions: (n_drones, 3) array of positions
        
        Returns:
            List of neighbor indices (excluding i itself)
        
        Implementation Note:
            From Borrmann 2015: "Only agents within sensing disk need CBF constraints"
            From GCBF+ Definition 1: Neighbors are within radius R
        """
        neighbors = []
        p_i = positions[i]
        
        for j in range(self.n_drones):
            if i == j:
                continue
            p_j = positions[j]
            dist = np.linalg.norm(p_i - p_j)
            
            # Only consider agents within sensing radius
            if dist <= self.R_sense:
                neighbors.append(j)
                
        return neighbors
    
    def filter_accelerations(self,
                            positions: np.ndarray,
                            velocities: np.ndarray,
                            acc_desired: np.ndarray,
                            obstacles: Optional[List[dict]] = None) -> np.ndarray:
        """
        Filter desired accelerations through CBF-QP for safety.
        
        This is the main method that implements the safety filter described in
        Borrmann Eq. (13) and GCBF+ Eq. (17).
        
        Args:
            positions: (n_drones, 3) current positions [x, y, z] in meters
            velocities: (n_drones, 3) current velocities [vx, vy, vz] in m/s
            acc_desired: (n_drones, 3) desired accelerations from formation controller
            obstacles: List of dicts with 'center' and 'radius' keys (optional)
        
        Returns:
            (n_drones, 3) safe accelerations that satisfy CBF constraints
        
        QP Formulation:
            minimize: ||a - a_desired||²
            subject to:
                - Inter-drone CBF constraints (only with neighbors)
                - Obstacle CBF constraints
                - Acceleration magnitude limits
        
        Notes:
            - Uses OSQP solver for efficiency
            - Falls back to desired acceleration if QP infeasible
            - Only forms constraints between neighbors (graph-based approach)
        """
        import time
        t_start = time.time()
        
        # Decision variables: acceleration for each drone
        a = cp.Variable((self.n_drones, 3))
        
        # Objective: minimize deviation from desired (Eq. 13 in Borrmann)
        cost = cp.sum_squares(a - acc_desired)
        
        constraints = []
        num_safety_constraints = 0
        
        # 1. GRAPH-BASED inter-drone constraints
        #    Key difference from naive CBF: only between neighbors!
        for i in range(self.n_drones):
            neighbors_i = self.get_neighbors(i, positions)
            
            # Only form CBF with neighbors within sensing radius
            for j in neighbors_i:
                if j > i:  # Avoid duplicate constraints (symmetric)
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
        
        # 3. Acceleration magnitude limits (Eq. 13 in Borrmann)
        for i in range(self.n_drones):
            # Box constraints: -a_max ≤ a[i,:] ≤ a_max
            constraints.append(a[i, :] <= self.a_max)
            constraints.append(a[i, :] >= -self.a_max)
        
        # Solve QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
        except Exception as e:
            print(f"QP Solver Error: {e}")
            print(f"Falling back to desired accelerations")
            return acc_desired
        
        # Track solve time
        t_solve = time.time() - t_start
        self.qp_solve_times.append(t_solve)
        
        # Check if solution is valid
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"Warning: QP status = {problem.status}")
            print(f"  Number of safety constraints: {num_safety_constraints}")
            print(f"  Falling back to desired accelerations")
            self.constraint_violations += 1
            return acc_desired
        
        # Check if CBF actually modified the control
        deviation = np.linalg.norm(a.value - acc_desired)
        if deviation > 1e-3:  # Threshold for "significant" modification
            self.cbf_activations += 1
        
        return a.value
    
    def _inter_drone_cbf_constraint(self, 
                                   i: int, 
                                   j: int,
                                   positions: np.ndarray,
                                   velocities: np.ndarray,
                                   a: cp.Variable) -> cp.Expression:
        """
        Generate second-order CBF constraint for collision between drones i and j.
        
        Implements the pairwise CBF from GCBF+ Eq. (30-31) and Borrmann Eq. (10-11).
        
        Barrier function: h_ij(x) = ||p_i - p_j||² - d_safe²
        
        Safe set: {x | h_ij(x) ≥ 0} means drones are at least d_safe apart
        
        Derivatives:
            First:  ḣ_ij = 2(p_i - p_j)·(v_i - v_j)
            Second: ḧ_ij = 2||v_i - v_j||² + 2(p_i - p_j)·(a_i - a_j)
        
        CBF condition (Eq. 34 in GCBF+): ḧ_ij + α₁·ḣ_ij + α₂·h_ij ≥ 0
        
        Args:
            i, j: Drone indices (i ≠ j)
            positions: (n_drones, 3) positions
            velocities: (n_drones, 3) velocities
            a: cvxpy Variable for accelerations
        
        Returns:
            cvxpy constraint expression
        """
        # Extract states
        p_i, p_j = positions[i], positions[j]
        v_i, v_j = velocities[i], velocities[j]
        
        # Barrier function value: h = ||Δp||² - d²
        p_diff = p_i - p_j
        h = np.dot(p_diff, p_diff) - self.d_safe**2
        
        # First derivative: ḣ = 2Δp·Δv
        v_diff = v_i - v_j
        h_dot = 2 * np.dot(p_diff, v_diff)
        
        # Second derivative (with control variables): ḧ = 2||Δv||² + 2Δp·Δa
        # Note: The control variables (a[i,:] - a[j,:]) make this linear in decision variables
        h_ddot = 2 * np.dot(v_diff, v_diff) + 2 * p_diff @ (a[i, :] - a[j, :])
        
        # CBF constraint: ḧ + α₁ḣ + α₂h ≥ 0
        # This ensures the safe set is forward invariant
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def _obstacle_cbf_constraint(self,
                                i: int,
                                positions: np.ndarray,
                                velocities: np.ndarray,
                                a: cp.Variable,
                                obstacle: dict) -> cp.Expression:
        """
        Generate CBF constraint for drone i avoiding static obstacle.
        
        For static obstacles: v_obs = 0, a_obs = 0
        This simplifies the second derivative calculation.
        
        Barrier function: h_i,obs = ||p_i - p_obs||² - (r_obs + margin)²
        
        Args:
            i: Drone index
            positions: (n_drones, 3) positions
            velocities: (n_drones, 3) velocities
            a: cvxpy Variable
            obstacle: dict with 'center' (3D point) and 'radius' (float)
        
        Returns:
            cvxpy constraint
        """
        p_i = positions[i]
        v_i = velocities[i]
        
        p_obs = np.array(obstacle['center'])
        r_obs = obstacle['radius'] + self.obs_margin
        
        # Barrier function
        p_diff = p_i - p_obs
        h = np.dot(p_diff, p_diff) - r_obs**2
        
        # First derivative (v_obs = 0 simplifies this)
        h_dot = 2 * np.dot(p_diff, v_i)
        
        # Second derivative (a_obs = 0 simplifies this)
        h_ddot = 2 * np.dot(v_i, v_i) + 2 * p_diff @ a[i, :]
        
        # CBF constraint
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def check_safety(self, 
                    positions: np.ndarray, 
                    obstacles: Optional[List[dict]] = None) -> Tuple[bool, List[str]]:
        """
        Check if current configuration is safe (all barriers positive).
        
        This is useful for:
        1. Validating initial conditions before simulation
        2. Detecting if CBF-QP failed to maintain safety
        3. Logging violations for analysis
        
        Args:
            positions: (n_drones, 3) current positions
            obstacles: Optional list of obstacles
        
        Returns:
            is_safe: True if all barriers are positive (h ≥ 0)
            violations: List of violation descriptions
        """
        violations = []
        
        # Check inter-drone distances (only with neighbors for efficiency)
        for i in range(self.n_drones):
            neighbors_i = self.get_neighbors(i, positions)
            for j in neighbors_i:
                if j > i:  # Avoid checking twice
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
        
        Where:
            - D_s: safety distance (self.d_safe)
            - Δa_max: maximum relative braking acceleration (2 * self.a_max)
            - γ: CBF decay rate (related to alpha2)
            - Δv_max: maximum relative velocity (2 * v_max)
        
        This ensures that if two drones are outside each other's sensing radius,
        they will never collide even in worst-case scenario.
        
        Args:
            gamma: CBF decay rate (use alpha2 typically)
            v_max: Maximum speed of a single drone (m/s)
        
        Returns:
            Minimum sensing radius D_N in meters
        
        Usage:
            v_max = 3.0  # m/s
            R_min = cbf.compute_minimum_sensing_radius(cbf.alpha2, v_max)
            if cbf.R_sense < R_min:
                print(f"WARNING: Increase sensing_radius to {R_min:.2f}m for guarantees")
        """
        delta_a_max = 2 * self.a_max  # Both agents can brake
        delta_v_max = 2 * v_max       # Worst case: head-on collision
        
        # Borrmann Eq. (14)
        D_N = self.d_safe + (1/(2*delta_a_max)) * \
            (np.sqrt(2*delta_a_max/gamma) + delta_v_max)**2
        
        return D_N
    
    def get_statistics(self) -> dict:
        """
        Get CBF performance statistics for analysis.
        
        Returns:
            Dictionary with:
                - mean_solve_time_ms: Average QP solve time
                - max_solve_time_ms: Worst-case QP solve time
                - total_solves: Number of QP problems solved
                - cbf_activations: Times CBF modified control
                - activation_rate: Fraction of time CBF was active
                - constraint_violations: Times QP failed
        """
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


# ================== USAGE EXAMPLE ==================

if __name__ == "__main__":
    """
    Example usage: Two drones on collision course
    """
    
    # Setup
    cbf = GraphCBFSafetyFilter(
        n_drones=2,
        safety_distance=1.5,
        sensing_radius=5.0,
        alpha1=2.0,
        alpha2=1.0,
        max_acceleration=5.0
    )
    
    # Initial state: two drones 3m apart, heading towards each other
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ])
    
    velocities = np.array([
        [1.0, 0.0, 0.0],   # Drone 0: moving right at 1 m/s
        [-1.0, 0.0, 0.0]   # Drone 1: moving left at 1 m/s
    ])
    
    # Formation controller wants to accelerate even more (bad!)
    acc_desired = np.array([
        [0.5, 0.0, 0.0],   # Drone 0: wants to speed up
        [-0.5, 0.0, 0.0]   # Drone 1: wants to speed up
    ])
    
    # Check if initially safe
    is_safe, violations = cbf.check_safety(positions)
    print(f"Initially safe: {is_safe}")
    
    # Filter through CBF
    acc_safe = cbf.filter_accelerations(positions, velocities, acc_desired)
    
    print("\nDesired accelerations:")
    print(acc_desired)
    print("\nSafe accelerations (filtered by CBF):")
    print(acc_safe)
    print("\nDeviation from desired:")
    print(np.linalg.norm(acc_safe - acc_desired, axis=1))
    
    # Statistics
    stats = cbf.get_statistics()
    print(f"\nCBF Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")