## Comprehensive CBF Formation Control Project TODO

### **Phase 1: Theory Foundation (Week 1)**

**Tuesday - Borrmann 2015 Paper**
- [ ] Read Section 1-2: Problem formulation & CBF background
- [ ] Understand equation (10-11): Barrier function construction
- [ ] Study equation (13): QP formulation for centralized case
- [ ] Review Section 3.3: Notion of neighbor
- [ ] Sketch out 2-agent collision scenario on paper
- [ ] Map key equations to mathematical concepts

**Wednesday - GCBF+ Paper** 
- [ ] Read Section III: GCBF theory (Definition 1, Theorem 1)
- [ ] Focus on Section IV-B: Loss functions (equations 19-22)
- [ ] Understand Section IV-C: Control-invariant set approximation
- [ ] Study how GNN architecture enables variable neighbor counts
- [ ] Compare centralized vs decentralized formulations
- [ ] Note differences from Borrmann approach

**Friday - Code Walkthrough**
- [ ] Trace `cbf_safety.py` line-by-line with paper equations
- [ ] Map `_inter_drone_cbf_constraint` to paper math
- [ ] Understand `filter_accelerations` QP solver setup
- [ ] Check `_obstacle_cbf_constraint` implementation
- [ ] Verify `check_safety` logic
- [ ] Test with simple 2-drone scenario

---

### **Phase 2: Core Implementation (Week 2-3)**

**Formation Manager (`formation_manager.py`)**
- [ ] Design pentagon formation geometry (ID-based positions)
- [ ] Implement moving target tracking logic
- [ ] Add formation scaling based on target velocity
- [ ] Handle formation rotation around target
- [ ] Test with stationary target first
- [ ] Validate with moving target (constant velocity)

**CBF Integration**
- [ ] Modify `control.py` to call CBF filter before attitude controller
- [ ] Insert CBF-QP at acceleration command level (after line 52)
- [ ] Pass neighbor states to CBF module
- [ ] Handle edge case: QP infeasibility (fallback behavior)
- [ ] Add telemetry logging for CBF activations
- [ ] Test: Does CBF preserve formation when far from obstacles?

**Environment Setup**
- [ ] Copy quad_dynamics.py, tello.py, control.py, environment.py to Formation5Drone/
- [ ] Create empty `multi_drone_sim.py` skeleton
- [ ] Set up test map with 2-3 static obstacles
- [ ] Verify single drone can fly with existing controller

---

### **Phase 3: Multi-Drone Simulator (Week 3-4)**

**Simulator Architecture (`multi_drone_sim.py`)**
- [ ] Extend simulator to handle N=5 drones simultaneously
- [ ] Create data structures: `states[5]`, `controls[5]`, `neighbors[5]`
- [ ] Implement neighbor detection (within sensing radius R)
- [ ] Build centralized state update loop
- [ ] Add visualization: 5 drones + formation lines + target
- [ ] Implement data logging for all drone trajectories

**Formation Execution Loop**
- [ ] Initialize 5 drones at spawn positions
- [ ] Formation manager computes desired positions
- [ ] Position controller → velocity → **CBF filter** → acceleration
- [ ] Integrate dynamics for all drones
- [ ] Update visualization every frame
- [ ] Check: formation maintained? collisions avoided?

**Testing Scenarios**
- [ ] Test 1: Static target, no obstacles (formation stability)
- [ ] Test 2: Moving target, no obstacles (tracking performance)
- [ ] Test 3: Static target, 3 obstacles (CBF safety check)
- [ ] Test 4: Moving target through obstacle field (stress test)
- [ ] Measure: collision rate, formation error, CBF activation frequency

---

### **Phase 4: Validation & Tuning (Week 5)**

**Parameter Tuning**
- [ ] Tune `alpha1`, `alpha2` CBF parameters for smooth behavior
- [ ] Adjust `safety_distance` for formation density
- [ ] Tune `max_acceleration` limits
- [ ] Optimize QP solver settings (OSQP tolerance)
- [ ] Find balance: safety vs formation tracking

**Failure Mode Analysis**
- [ ] Test: What happens if 2 drones head-on collision course?
- [ ] Test: Drones trapped between obstacles?
- [ ] Test: Target moves faster than drone max speed?
- [ ] Document failure modes and CBF behavior
- [ ] Implement fallback: emergency stop if QP fails repeatedly

**Metrics Collection**
- [ ] Log: min inter-drone distance over time
- [ ] Log: formation error (deviation from ideal pentagon)
- [ ] Log: target tracking error
- [ ] Count: number of CBF constraint activations
- [ ] Generate plots for report/presentation

---

### **Phase 5: Report & Deliverables (Week 6)**

**Code Deliverables**
- [ ] Clean up code: remove debug prints, add docstrings
- [ ] Write README.md with setup instructions
- [ ] Create requirements.txt with dependencies
- [ ] Add example run script: `python main.py --scenario 1`
- [ ] Push to Git with clear commit history

**Documentation**
- [ ] Write theory section: CBF formulation for formation control
- [ ] Explain QP integration point in control architecture
- [ ] Document parameter choices and tuning process
- [ ] Include simulation results (plots + videos)
- [ ] Discuss limitations and future work

**Presentation Materials**
- [ ] Create slides: problem → approach → results
- [ ] Make demo video: formation control with obstacle avoidance
- [ ] Prepare 5-min explanation of CBF theory
- [ ] Show comparison: with/without CBF safety layer
- [ ] Have backup slides on failure modes

---

### **Stretch Goals (If Time Permits)**

- [ ] Implement decentralized CBF (each drone runs own QP)
- [ ] Add dynamic obstacles (moving threats)
- [ ] Test with 10+ drones
- [ ] Compare to baseline: artificial potential fields
- [ ] Real hardware test on 2-3 Crazyflie drones

---

**Critical Path Items** (must complete):
1. Understand CBF theory from papers ✓ Week 1
2. Get CBF-QP working for 2 drones → Week 2
3. Formation manager for 5 drones → Week 3
4. Full simulation with obstacles → Week 4
5. Working demo + documentation → Week 6

Track completion in a spreadsheet or project board. Flag blockers immediately.