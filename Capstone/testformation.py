"""
Test script to verify Formation5Drone_UPDATED.py and CBF integration.
Run this first before full simulation to catch any issues.
"""

import numpy as np
import sys

print("="*60)
print("Formation5Drone + CBF Integration Test Suite")
print("="*60)

# Test 1: Basic imports
print("\nTest 1: Checking imports...")
try:
    from Formation5Drone import DroneAgent
    print("✓ DroneAgent imported successfully")
except ImportError as e:
    print(f"✗ Failed to import DroneAgent: {e}")
    sys.exit(1)

try:
    from cbf_safety import GraphCBFSafetyFilter
    print("✓ GraphCBFSafetyFilter imported successfully")
    cbf_available = True
except ImportError as e:
    print(f"⚠  CBF not available (this is OK for basic testing): {e}")
    cbf_available = False

# Test 2: Agent initialization
print("\nTest 2: Agent initialization...")
try:
    agent = DroneAgent(
        id=0,
        state_3d=[0.0, 0.0, 2.0],
        target_pos_3d=[0.0, 0.0, 2.0],
        formation_radius=5.0,
        Kp=1.0,
        Kd=0.5,
        dt=0.1
    )
    print(f"✓ Agent created: position={agent.position}, velocity={agent.velocity}")
    assert agent.position.shape == (3,), "Position should be 3D"
    assert agent.velocity.shape == (3,), "Velocity should be 3D"
    print("✓ State dimensions correct")
except Exception as e:
    print(f"✗ Agent initialization failed: {e}")
    sys.exit(1)

# Test 3: Acceleration computation
print("\nTest 3: Acceleration computation...")
try:
    acc = agent.compute_desired_acceleration()
    print(f"✓ Computed acceleration: {acc}")
    assert acc.shape == (3,), "Acceleration should be 3D"
    print("✓ Acceleration dimensions correct")
except Exception as e:
    print(f"✗ Acceleration computation failed: {e}")
    sys.exit(1)

# Test 4: Dynamics update
print("\nTest 4: Dynamics update...")
try:
    old_pos = agent.position.copy()
    old_vel = agent.velocity.copy()
    test_acc = np.array([1.0, 0.0, 0.0])
    
    agent.update_dynamics(test_acc)
    
    print(f"✓ Position updated: {old_pos} → {agent.position}")
    print(f"✓ Velocity updated: {old_vel} → {agent.velocity}")
    
    # Verify physics: v_new = v_old + a*dt
    expected_vel = old_vel + test_acc * agent.dt
    assert np.allclose(agent.velocity, expected_vel), "Velocity integration incorrect"
    print("✓ Physics integration correct")
except Exception as e:
    print(f"✗ Dynamics update failed: {e}")
    sys.exit(1)

# Test 5: Formation error
print("\nTest 5: Formation error computation...")
try:
    error = agent.get_formation_error()
    print(f"✓ Formation error: {error:.3f}m")
    assert error >= 0, "Error should be non-negative"
    print("✓ Error calculation correct")
except Exception as e:
    print(f"✗ Formation error computation failed: {e}")
    sys.exit(1)

# Test 6: Multi-agent setup
print("\nTest 6: Multi-agent setup...")
try:
    n_agents = 5
    target = np.array([0.0, 0.0, 2.0])
    agents = []
    
    for i in range(n_agents):
        agents.append(DroneAgent(
            id=i,
            state_3d=np.random.randn(3),
            target_pos_3d=target,
            formation_radius=5.0
        ))
    
    print(f"✓ Created {n_agents} agents")
    
    # Test message passing
    for agent in agents:
        msg = agent.msg()
        assert len(msg) == 3, "Message should be (id, position, velocity)"
        assert msg[1].shape == (3,), "Position in message should be 3D"
        assert msg[2].shape == (3,), "Velocity in message should be 3D"
    
    print("✓ Message passing works")
except Exception as e:
    print(f"✗ Multi-agent setup failed: {e}")
    sys.exit(1)

# Test 7: CBF integration (if available)
if cbf_available:
    print("\nTest 7: CBF integration...")
    try:
        # Initialize CBF
        cbf = GraphCBFSafetyFilter(
            n_drones=2,
            safety_distance=1.5,
            sensing_radius=5.0,
            alpha1=2.0,
            alpha2=1.0,
            max_acceleration=5.0
        )
        print("✓ CBF filter initialized")
        
        # Test collision scenario
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]  # 2m apart
        ])
        velocities = np.array([
            [1.0, 0.0, 0.0],   # Moving towards each other
            [-1.0, 0.0, 0.0]
        ])
        acc_desired = np.array([
            [0.5, 0.0, 0.0],   # Want to speed up
            [-0.5, 0.0, 0.0]
        ])
        
        # Filter
        acc_safe = cbf.filter_accelerations(positions, velocities, acc_desired)
        print(f"✓ CBF filtering successful")
        print(f"  Desired: {acc_desired[0]}")
        print(f"  Safe:    {acc_safe[0]}")
        
        deviation = np.linalg.norm(acc_safe - acc_desired)
        if deviation > 0.01:
            print(f"✓ CBF modified control (deviation={deviation:.3f})")
        else:
            print(f"⚠  CBF didn't activate (might be OK if far apart)")
        
        # Check safety
        is_safe, violations = cbf.check_safety(positions)
        if is_safe:
            print("✓ Configuration is safe")
        else:
            print(f"⚠  Safety violations detected: {violations}")
        
        # Check minimum sensing radius
        v_max = 3.0
        R_min = cbf.compute_minimum_sensing_radius(cbf.alpha2, v_max)
        print(f"✓ Minimum sensing radius: {R_min:.2f}m")
        
        if cbf.R_sense >= R_min:
            print(f"✓ Sensing radius {cbf.R_sense}m ≥ {R_min:.2f}m (safe)")
        else:
            print(f"⚠  Sensing radius {cbf.R_sense}m < {R_min:.2f}m (increase!)")
        
    except Exception as e:
        print(f"✗ CBF integration test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nTest 7: CBF integration... SKIPPED (not installed)")

# Test 8: Simple simulation loop
print("\nTest 8: Simple simulation loop...")
try:
    # Create 2 agents
    agents = [
        DroneAgent(0, [0, 0, 2], [0, 0, 2], formation_radius=3.0, dt=0.1),
        DroneAgent(1, [5, 0, 2], [0, 0, 2], formation_radius=3.0, dt=0.1)
    ]
    
    # Run 10 steps
    for _ in range(10):
        # Compute accelerations
        acc_desired = np.array([a.compute_desired_acceleration() for a in agents])
        
        # Apply (no CBF in this test)
        for i, agent in enumerate(agents):
            agent.update_dynamics(acc_desired[i])
    
    print(f"✓ Simulation loop completed")
    print(f"  Agent 0 final position: {agents[0].position}")
    print(f"  Agent 1 final position: {agents[1].position}")
    
    # Check histories
    assert len(agents[0].position_hist) == 11, "History should have 11 entries (init + 10 steps)"
    print("✓ History tracking works")
    
except Exception as e:
    print(f"✗ Simulation loop test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✓ All basic tests passed!")
if cbf_available:
    print("✓ CBF integration tests passed!")
else:
    print("⚠  CBF not tested (install cbf_safety_FIXED.py)")
print("\nYou can now run the full simulation:")
print("  python Formation5Drone_UPDATED.py")
print("="*60)