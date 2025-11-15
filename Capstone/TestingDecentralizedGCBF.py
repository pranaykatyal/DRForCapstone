# test_distributed_gcbf.py - Comprehensive Testing Suite
"""
Stress testing for distributed GCBF implementation.

Tests cover:
1. Head-on collision (2 drones)
2. Multi-drone convergence (5 drones)
3. Narrow passage with obstacles
4. Dynamic neighbor changes (drones entering/leaving comm range)
"""

import numpy as np
from DecentralizedGCBF import GraphCBFSafetyFilter
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================================
# TEST UTILITIES
# ============================================================================

def simulate_step(positions, velocities, accelerations, dt=0.1):
    """Simulate one timestep of dynamics."""
    velocities_new = velocities + accelerations * dt
    positions_new = positions + velocities_new * dt
    return positions_new, velocities_new

def run_distributed_cbf(cbf, positions, velocities, acc_desired, obstacles=None):
    """
    Run distributed CBF for all drones (simulates parallel execution).
    
    Each drone:
    1. Gathers neighbor info within comm range
    2. Solves local QP independently
    3. Returns safe acceleration
    """
    n_drones = len(positions)
    acc_safe = np.zeros_like(acc_desired)
    
    for i in range(n_drones):
        # Gather neighbor information (within comm range)
        neighbor_positions = []
        neighbor_velocities = []
        
        for j in range(n_drones):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= cbf.R_sense:  # Within communication range
                    neighbor_positions.append(positions[j])
                    neighbor_velocities.append(velocities[j])
        
        # Solve local QP for this drone
        acc_safe[i] = cbf.filter_acceleration_single_drone(
            my_position=positions[i],
            my_velocity=velocities[i],
            my_acc_desired=acc_desired[i],
            neighbor_positions=neighbor_positions,
            neighbor_velocities=neighbor_velocities,
            obstacles=obstacles
        )
    
    return acc_safe

def check_violations(positions, safety_distance=2.0):
    """Check for any safety violations."""
    n_drones = len(positions)
    violations = []
    
    for i in range(n_drones):
        for j in range(i+1, n_drones):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < safety_distance:
                violations.append((i, j, dist))
    
    return violations

def print_test_header(test_name):
    """Print formatted test header."""
    print("\n" + "="*80)
    print(f"TEST: {test_name}")
    print("="*80)

def print_test_result(passed, message=""):
    """Print test result."""
    if passed:
        print(f"✅ PASSED: {message}")
    else:
        print(f"❌ FAILED: {message}")
    print()

# ============================================================================
# TEST 1: HEAD-ON COLLISION (2 Drones)
# ============================================================================

def test_head_on_collision():
    """
    Test: Two drones on direct collision course.
    Expected: CBF should brake both drones to prevent collision.
    """
    print_test_header("HEAD-ON COLLISION (2 Drones)")
    
    cbf = GraphCBFSafetyFilter(
        n_drones=2,
        safety_distance=2.0,
        sensing_radius=8.0,
        alpha1=3.0,
        alpha2=2.5,
        max_acceleration=5.0
    )
    
    # Initial conditions: 4m apart, heading toward each other at 2 m/s
    positions = np.array([
        [0.0, 0.0, 1.0],
        [4.0, 0.0, 1.0]
    ])
    
    velocities = np.array([
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0]
    ])
    
    # Controller wants to accelerate (worst case)
    acc_desired = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    
    print("Initial conditions:")
    print(f"  Distance: {np.linalg.norm(positions[0] - positions[1]):.2f}m")
    print(f"  Relative velocity: {np.linalg.norm(velocities[0] - velocities[1]):.2f} m/s")
    print(f"  Time to collision (without CBF): {2.0 / 4.0:.2f}s")
    
    # Simulate for 2 seconds
    dt = 0.02
    steps = int(2.0 / dt)
    min_distance = float('inf')
    had_violation = False
    
    position_history = [positions.copy()]
    
    for step in range(steps):
        # Run distributed CBF
        acc_safe = run_distributed_cbf(cbf, positions, velocities, acc_desired)
        
        # Update dynamics
        positions, velocities = simulate_step(positions, velocities, acc_safe, dt)
        position_history.append(positions.copy())
        
        # Check distance
        dist = np.linalg.norm(positions[0] - positions[1])
        min_distance = min(min_distance, dist)
        
        # Check for violation
        if dist < 2.0:
            had_violation = True
            print(f"  ⚠️  Violation at t={step*dt:.2f}s: dist={dist:.3f}m")
    
    print(f"\nResults after {steps*dt:.1f}s:")
    print(f"  Minimum distance: {min_distance:.3f}m")
    print(f"  Final distance: {np.linalg.norm(positions[0] - positions[1]):.3f}m")
    
    # Test passes if no collision occurred
    passed = min_distance >= 1.95  # Allow small numerical error
    print_test_result(passed, 
        f"Min distance: {min_distance:.3f}m >= 2.0m safety threshold" if passed 
        else f"COLLISION! Min distance: {min_distance:.3f}m")
    
    return passed, position_history

# ============================================================================
# TEST 2: MULTI-DRONE CONVERGENCE (5 Drones)
# ============================================================================

def test_multi_drone_convergence():
    """
    Test: 5 drones starting randomly, all trying to reach center.
    Expected: No collisions while converging to common point.
    """
    print_test_header("MULTI-DRONE CONVERGENCE (5 Drones)")
    
    cbf = GraphCBFSafetyFilter(
        n_drones=5,
        safety_distance=2.0,
        sensing_radius=8.0,
        alpha1=3.0,
        alpha2=2.5,
        max_acceleration=5.0
    )
    
    # Random starting positions around a circle
    np.random.seed(42)
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    radius = 8.0
    positions = np.array([
        [radius * np.cos(angle), radius * np.sin(angle), 1.0]
        for angle in angles
    ])
    
    velocities = np.zeros((5, 3))
    
    print("Initial configuration:")
    print(f"  Drones arranged in circle of radius {radius}m")
    print(f"  All trying to reach center at (0, 0, 1)")
    
    # Simulate for 5 seconds
    dt = 0.02
    steps = int(5.0 / dt)
    violations = []
    min_distances = {(i, j): float('inf') for i in range(5) for j in range(i+1, 5)}
    
    position_history = [positions.copy()]
    
    for step in range(steps):
        # Desired: accelerate toward center
        target = np.array([0.0, 0.0, 1.0])
        acc_desired = np.zeros((5, 3))
        
        for i in range(5):
            direction = target - positions[i]
            dist_to_target = np.linalg.norm(direction)
            if dist_to_target > 0.1:  # Don't accelerate if very close
                acc_desired[i] = 2.0 * direction / dist_to_target  # Unit vector * gain
        
        # Run distributed CBF
        acc_safe = run_distributed_cbf(cbf, positions, velocities, acc_desired)
        
        # Update dynamics
        positions, velocities = simulate_step(positions, velocities, acc_safe, dt)
        position_history.append(positions.copy())
        
        # Check all pairwise distances
        for i in range(5):
            for j in range(i+1, 5):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distances[(i, j)] = min(min_distances[(i, j)], dist)
                
                if dist < 2.0:
                    violations.append((step*dt, i, j, dist))
    
    print(f"\nResults after {steps*dt:.1f}s:")
    print(f"  Total violations: {len(violations)}")
    
    if violations:
        print(f"  First violation at t={violations[0][0]:.2f}s: "
              f"Drones {violations[0][1]}-{violations[0][2]}, dist={violations[0][3]:.3f}m")
    
    # Print minimum distances
    print("\n  Minimum pairwise distances:")
    for (i, j), dist in sorted(min_distances.items()):
        status = "✓" if dist >= 1.95 else "✗"
        print(f"    Drones {i}-{j}: {dist:.3f}m {status}")
    
    # Test passes if all minimum distances >= 1.95m
    passed = all(dist >= 1.95 for dist in min_distances.values())
    print_test_result(passed, 
        "All drones maintained safe distance" if passed 
        else f"Safety violations detected!")
    
    return passed, position_history

# ============================================================================
# TEST 3: NARROW PASSAGE WITH OBSTACLES
# ============================================================================

def test_narrow_passage():
    """
    Test: 3 drones must navigate through narrow gap between obstacles.
    Expected: Drones avoid both obstacles and each other.
    """
    print_test_header("NARROW PASSAGE WITH OBSTACLES")
    
    cbf = GraphCBFSafetyFilter(
        n_drones=3,
        safety_distance=2.0,
        sensing_radius=8.0,
        obstacle_margin=0.5,
        alpha1=3.0,
        alpha2=2.5,
        max_acceleration=5.0
    )
    
    # Starting positions: left side
    positions = np.array([
        [-5.0, -2.5, 1.0],
        [-5.0, 0.0, 1.0],
        [-5.0, 2.5, 1.0]
    ])
    
    velocities = np.zeros((3, 3))
    
    # Two obstacles creating a narrow passage
    obstacles = [
        {'center': np.array([0.0, 3.0, 1.0]), 'radius': 1.5},
        {'center': np.array([0.0, -3.0, 1.0]), 'radius': 1.5}
    ]
    
    # Gap width: 6.0m - 2*1.5m - 2*0.5m (margins) = 2.0m
    gap_width = 6.0 - 2 * (1.5 + 0.5)
    
    print("Configuration:")
    print(f"  Drones start at x=-5.0, trying to reach x=+5.0")
    print(f"  Two obstacles at y=±3.0m with radius 1.5m")
    print(f"  Effective gap width: {gap_width:.1f}m")
    print(f"  Drones must coordinate to pass through")
    
    # Simulate for 5 seconds
    dt = 0.02
    steps = int(5.0 / dt)
    violations_drones = []
    violations_obstacles = []
    
    position_history = [positions.copy()]
    
    for step in range(steps):
        # Desired: accelerate toward goal on right side
        goal = np.array([5.0, 0.0, 1.0])
        acc_desired = np.zeros((3, 3))
        
        for i in range(3):
            direction = goal - positions[i]
            dist_to_goal = np.linalg.norm(direction)
            if dist_to_goal > 0.5:
                acc_desired[i] = 1.5 * direction / dist_to_goal
        
        # Run distributed CBF
        acc_safe = run_distributed_cbf(cbf, positions, velocities, acc_desired, obstacles)
        
        # Update dynamics
        positions, velocities = simulate_step(positions, velocities, acc_safe, dt)
        position_history.append(positions.copy())
        
        # Check drone-drone distances
        for i in range(3):
            for j in range(i+1, 3):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2.0:
                    violations_drones.append((step*dt, i, j, dist))
        
        # Check drone-obstacle distances
        for i in range(3):
            for obs_idx, obs in enumerate(obstacles):
                dist = np.linalg.norm(positions[i] - obs['center'])
                min_dist = obs['radius'] + cbf.obs_margin
                if dist < min_dist:
                    violations_obstacles.append((step*dt, i, obs_idx, dist, min_dist))
    
    print(f"\nResults after {steps*dt:.1f}s:")
    print(f"  Drone-drone violations: {len(violations_drones)}")
    print(f"  Drone-obstacle violations: {len(violations_obstacles)}")
    
    if violations_drones:
        v = violations_drones[0]
        print(f"    First drone violation at t={v[0]:.2f}s: D{v[1]}-D{v[2]}, dist={v[3]:.3f}m")
    
    if violations_obstacles:
        v = violations_obstacles[0]
        print(f"    First obstacle violation at t={v[0]:.2f}s: "
              f"D{v[1]}-Obs{v[2]}, dist={v[3]:.3f}m < {v[4]:.3f}m")
    
    # Final positions
    print("\n  Final positions:")
    for i in range(3):
        print(f"    Drone {i}: x={positions[i, 0]:.2f}, y={positions[i, 1]:.2f}")
    
    # Test passes if no violations
    passed = len(violations_drones) == 0 and len(violations_obstacles) == 0
    print_test_result(passed, 
        "Successfully navigated narrow passage" if passed 
        else "Collisions detected during passage")
    
    return passed, position_history

# ============================================================================
# TEST 4: DYNAMIC NEIGHBOR CHANGES
# ============================================================================

def test_dynamic_neighbors():
    """
    Test: Drones moving in/out of communication range.
    Expected: CBF handles dynamic graph changes gracefully.
    """
    print_test_header("DYNAMIC NEIGHBOR CHANGES")
    
    cbf = GraphCBFSafetyFilter(
        n_drones=4,
        safety_distance=2.0,
        sensing_radius=6.0,  # Smaller range to test dynamic changes
        alpha1=3.0,
        alpha2=2.5,
        max_acceleration=5.0
    )
    
    # Positions: form two pairs initially out of range
    positions = np.array([
        [-10.0, 0.0, 1.0],  # Pair 1
        [-8.0, 0.0, 1.0],
        [8.0, 0.0, 1.0],    # Pair 2
        [10.0, 0.0, 1.0]
    ])
    
    # Moving toward center
    velocities = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    
    print("Initial configuration:")
    print(f"  Two pairs of drones, initially {np.linalg.norm(positions[1] - positions[2]):.1f}m apart")
    print(f"  Communication range: {cbf.R_sense}m")
    print(f"  Pairs will enter each other's comm range during simulation")
    
    # Track neighbor counts over time
    neighbor_counts = []
    violations = []
    
    # Simulate for 6 seconds
    dt = 0.02
    steps = int(6.0 / dt)
    
    position_history = [positions.copy()]
    
    for step in range(steps):
        # Controller wants to maintain velocity (zero acceleration)
        acc_desired = np.zeros((4, 3))
        
        # Run distributed CBF
        acc_safe = run_distributed_cbf(cbf, positions, velocities, acc_desired)
        
        # Count neighbors for each drone
        step_neighbors = []
        for i in range(4):
            count = 0
            for j in range(4):
                if i != j and np.linalg.norm(positions[i] - positions[j]) <= cbf.R_sense:
                    count += 1
            step_neighbors.append(count)
        neighbor_counts.append(step_neighbors)
        
        # Update dynamics
        positions, velocities = simulate_step(positions, velocities, acc_safe, dt)
        position_history.append(positions.copy())
        
        # Check violations
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 2.0:
                    violations.append((step*dt, i, j, dist))
    
    # Analyze neighbor changes
    neighbor_counts = np.array(neighbor_counts)
    
    print(f"\nResults after {steps*dt:.1f}s:")
    print(f"  Neighbor count evolution:")
    print(f"    Initial: {neighbor_counts[0]}")
    print(f"    Mid-point (t=3s): {neighbor_counts[int(3.0/dt)]}")
    print(f"    Final: {neighbor_counts[-1]}")
    
    print(f"\n  Safety violations: {len(violations)}")
    if violations:
        v = violations[0]
        print(f"    First violation at t={v[0]:.2f}s: D{v[1]}-D{v[2]}, dist={v[3]:.3f}m")
    
    # Test passes if no violations despite neighbor changes
    passed = len(violations) == 0
    print_test_result(passed, 
        "Handled dynamic neighbors without collisions" if passed 
        else "Violations during neighbor changes")
    
    return passed, position_history

# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def run_all_tests():
    """Run all tests and generate summary."""
    print("\n" + "#"*80)
    print("# DISTRIBUTED GCBF COMPREHENSIVE TEST SUITE")
    print("#"*80)
    
    results = {}
    
    # Run all tests
    results['test1'] = test_head_on_collision()[0]
    results['test2'] = test_multi_drone_convergence()[0]
    results['test3'] = test_narrow_passage()[0]
    results['test4'] = test_dynamic_neighbors()[0]
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTests passed: {passed_tests}/{total_tests}")
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*80)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Distributed GCBF is working correctly.")
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed. Review implementation.")
    
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    results = run_all_tests()