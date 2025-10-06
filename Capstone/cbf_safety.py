"""
Control Barrier Function (CBF) based safety filter using QP optimization.
Ensures collision avoidance between drones and with static obstacles.
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional, Tuple

class CBFSafetyFilter:
    """
    Second-order CBF implementation for multi-drone safety.
    
    Theory:
    For barrier function h(x) where safe set is {x | h(x) ≥ 0}:
    - Second-order CBF: ḧ + α₁·ḣ + α₂·h ≥ 0
    - This guarantees forward invariance of safe set
    
    For collision avoidance: h(x) = ||p_i - p_j||² - d_safe²
    """
    
    def __init__(self, 
                 n_drones: int,
                 safety_distance: float = 1.0,
                 obstacle_margin: float = 0.5,
                 alpha1: float = 2.0,
                 alpha2: float = 1.0,
                 max_acceleration: float = 5.0):
        """
        Initialize CBF safety filter.
        
        Args:
            n_drones: Number of drones in system
            safety_distance: Minimum distance between drones (meters)
            obstacle_margin: Safety margin around obstacles (meters)
            alpha1: CBF parameter for ḣ term (controls aggressiveness)
            alpha2: CBF parameter for h term (controls convergence)
            max_acceleration: Maximum allowed acceleration magnitude (m/s²)
        """
        self.n_drones = n_drones
        self.d_safe = safety_distance
        self.obs_margin = obstacle_margin
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.a_max = max_acceleration
        
        # Statistics
        self.qp_solve_times = []
        self.constraint_violations = 0
        
    def filter_accelerations(self,
                            positions: np.ndarray,
                            velocities: np.ndarray,
                            acc_desired: np.ndarray,
                            obstacles: Optional[List[dict]] = None) -> np.ndarray:
        """
        Filter desired accelerations through CBF-QP to ensure safety.
        
        Args:
            positions: (n_drones, 3) array of positions
            velocities: (n_drones, 3) array of velocities
            acc_desired: (n_drones, 3) array of desired accelerations
            obstacles: List of obstacle dicts with 'center' and 'radius'
            
        Returns:
            acc_safe: (n_drones, 3) array of safe accelerations
        """
        import time
        start_time = time.time()
        
        # Decision variables: safe accelerations for all drones
        a = cp.Variable((self.n_drones, 3))
        
        # Objective: minimize deviation from desired accelerations
        cost = cp.sum_squares(a - acc_desired)
        
        # Constraints list
        constraints = []
        
        # 1. Inter-drone collision avoidance constraints
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                cbf_constraint = self._inter_drone_cbf_constraint(
                    i, j, positions, velocities, a
                )
                constraints.append(cbf_constraint)
        
        # 2. Obstacle avoidance constraints
        if obstacles is not None:
            for i in range(self.n_drones):
                for obs in obstacles:
                    obs_constraint = self._obstacle_cbf_constraint(
                        i, positions, velocities, a, obs
                    )
                    constraints.append(obs_constraint)
        
        # 3. Acceleration magnitude limits
        for i in range(self.n_drones):
            # Box constraints: -a_max ≤ a[i] ≤ a_max for each component
            constraints.append(a[i, :] <= self.a_max)
            constraints.append(a[i, :] >= -self.a_max)
        
        # Solve QP
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status != cp.OPTIMAL:
                print(f"Warning: QP status = {problem.status}, using desired accelerations")
                return acc_desired
            
            acc_safe = a.value
            
            # Track solve time
            solve_time = time.time() - start_time
            self.qp_solve_times.append(solve_time)
            
            return acc_safe
            
        except Exception as e:
            print(f"CBF-QP solve failed: {e}")
            print("Falling back to desired accelerations (UNSAFE)")
            return acc_desired
    
    def _inter_drone_cbf_constraint(self, 
                                   i: int, 
                                   j: int,
                                   positions: np.ndarray,
                                   velocities: np.ndarray,
                                   a: cp.Variable) -> cp.Expression:
        """
        Generate second-order CBF constraint for collision between drones i and j.
        
        Barrier function: h_ij(x) = ||p_i - p_j||² - d_safe²
        
        First derivative: ḣ_ij = 2(p_i - p_j)·(v_i - v_j)
        
        Second derivative: ḧ_ij = 2||v_i - v_j||² + 2(p_i - p_j)·(a_i - a_j)
        
        CBF condition: ḧ_ij + α₁·ḣ_ij + α₂·h_ij ≥ 0
        """
        # Extract states
        p_i, p_j = positions[i], positions[j]
        v_i, v_j = velocities[i], velocities[j]
        
        # Barrier function value
        p_diff = p_i - p_j
        h = np.dot(p_diff, p_diff) - self.d_safe**2
        
        # First derivative
        v_diff = v_i - v_j
        h_dot = 2 * np.dot(p_diff, v_diff)
        
        # Second derivative (with control variables)
        h_ddot = 2 * np.dot(v_diff, v_diff) + 2 * p_diff @ (a[i, :] - a[j, :])
        
        # CBF constraint: h_ddot + α₁·h_dot + α₂·h ≥ 0
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def _obstacle_cbf_constraint(self,
                                i: int,
                                positions: np.ndarray,
                                velocities: np.ndarray,
                                a: cp.Variable,
                                obstacle: dict) -> cp.Expression:
        """
        Generate CBF constraint for drone i avoiding static obstacle.
        
        For static obstacle: v_obs = 0, a_obs = 0
        Simplifies the second derivative calculation.
        """
        p_i = positions[i]
        v_i = velocities[i]
        
        p_obs = np.array(obstacle['center'])
        r_obs = obstacle['radius'] + self.obs_margin
        
        # Barrier function
        p_diff = p_i - p_obs
        h = np.dot(p_diff, p_diff) - r_obs**2
        
        # First derivative (v_obs = 0)
        h_dot = 2 * np.dot(p_diff, v_i)
        
        # Second derivative (a_obs = 0)
        h_ddot = 2 * np.dot(v_i, v_i) + 2 * p_diff @ a[i, :]
        
        # CBF constraint
        return h_ddot + self.alpha1 * h_dot + self.alpha2 * h >= 0
    
    def check_safety(self, 
                    positions: np.ndarray, 
                    obstacles: Optional[List[dict]] = None) -> Tuple[bool, List[str]]:
        """
        Check if current configuration is safe (all barriers positive).
        
        Returns:
            is_safe: True if all barriers are positive
            violations: List of violation descriptions
        """
        violations = []
        
        # Check inter-drone distances
        for i in range(self.n_drones):
            for j in range(i + 1, self.n_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.d_safe:
                    violations.append(
                        f"Drones {i} and {j} too close: {dist:.3f}m < {self.d_safe}m"
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
    
    def get_statistics(self) -> dict:
        """Get CBF performance statistics."""
        if not self.qp_solve_times:
            return {"message": "No QP solves yet"}
        
        return {
            "mean_solve_time_ms": np.mean(self.qp_solve_times) * 1000,
            "max_solve_time_ms": np.max(self.qp_solve_times) * 1000,
            "total_solves": len(self.qp_solve_times),
            "constraint_violations": self.constraint_violations
        }