import numpy as np
import matplotlib.pyplot as plt

def verify_async_consensus_3_agents():
    """
    Verify the paper's math for 3-agent asynchronous consensus.
    Agent update order: X -> Y -> Z (with phase lags)
    """
    
    # Initial states
    X0 = np.array([100.0, 50.0, 20.0])  # Initial values for agents X, Y, Z
    alpha = 0.3  # Consensus parameter
    
    # Theoretical average (what they should converge to)
    avg = np.mean(X0)
    print(f"Theoretical consensus value: {avg:.2f}")
    
    # Setup matrices as in paper
    # A matrix: current state contributions
    A = np.array([
        [-2,  1,  1],
        [ 0, -2,  1],
        [ 0,  0, -2]
    ])
    
    # B matrix: next state contributions (phase lag structure)
    B = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])
    
    # Compute transformation matrix M = (I - αB)^(-1) * (I + αA)
    I = np.eye(3)
    M = np.linalg.inv(I - alpha * B) @ (I + alpha * A)
    
    print("\nTransformation matrix M:")
    print(M)
    
    # Eigenvalue analysis
    eigenvalues, eigenvectors = np.linalg.eig(M)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Magnitude of eigenvalues: {np.abs(eigenvalues)}")
    print(f"All eigenvalues in (0,1) except one at 1? {np.allclose(eigenvalues[0], 1.0) and np.all(np.abs(eigenvalues[1:]) < 1.0)}")
    
    # Simulate consensus
    X = X0.copy()
    history = [X.copy()]
    
    for n in range(100):
        X = M @ X
        history.append(X.copy())
    
    history = np.array(history)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history[:, 0], label='Agent X', linewidth=2)
    plt.plot(history[:, 1], label='Agent Y', linewidth=2)
    plt.plot(history[:, 2], label='Agent Z', linewidth=2)
    plt.axhline(avg, color='k', linestyle='--', label=f'True Average ({avg:.1f})', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('State Value', fontsize=12)
    plt.title(f'Asynchronous Consensus (α={alpha})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    final_value = history[-1, 0]
    drift = abs(final_value - avg)
    errors = np.abs(history - avg)
    plt.plot(errors[:, 0], label='Agent X error', linewidth=2)
    plt.plot(errors[:, 1], label='Agent Y error', linewidth=2)
    plt.plot(errors[:, 2], label='Agent Z error', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error from True Average', fontsize=12)
    plt.title(f'Convergence Error\nFinal Drift: {drift:.3f}', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('async_consensus_verification.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Plot saved to: async_consensus_verification.png")
    
    print(f"\nFinal converged values: {history[-1]}")
    print(f"Final drift from true average: {drift:.3f}")
    print(f"Drift percentage: {(drift/avg)*100:.2f}%")
    
    return M, eigenvalues, history


def test_different_alphas():
    """
    Reproduce Table II from the paper: test different alpha values
    """
    print("\n" + "="*60)
    print("TESTING DIFFERENT ALPHA VALUES (Table II Reproduction)")
    print("="*60)
    
    X0 = np.array([100.0, 50.0, 20.0])
    avg = np.mean(X0)
    alphas = [0.1, 0.3, 0.4, 0.5, 0.7]
    
    results = []
    
    for alpha in alphas:
        # Setup matrices
        A = np.array([[-2, 1, 1], [0, -2, 1], [0, 0, -2]])
        B = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        I = np.eye(3)
        
        try:
            M = np.linalg.inv(I - alpha * B) @ (I + alpha * A)
            eigenvalues = np.linalg.eig(M)[0]
            max_eig_magnitude = np.max(np.abs(eigenvalues[1:]))  # Ignore the eigenvalue at 1
            
            # Simulate
            X = X0.copy()
            for _ in range(200):
                X = M @ X
            
            final_value = X[0]
            drift = abs(final_value - avg)
            
            # Check for divergence
            if np.any(np.abs(X) > 1e6):
                status = "BLOWOUT"
            elif max_eig_magnitude >= 1.0:
                status = "NO CONSENSUS"
            else:
                status = f"{final_value:.1f}"
            
            results.append({
                'alpha': alpha,
                'async_value': status,
                'drift': drift,
                'eigenvalues': eigenvalues,
                'max_eig_mag': max_eig_magnitude
            })
            
            print(f"\nα = {alpha:.1f}:")
            print(f"  Final value: {status}")
            print(f"  Drift: {drift:.3f}")
            print(f"  Max eigenvalue magnitude (excluding 1): {max_eig_magnitude:.4f}")
            
        except Exception as e:
            print(f"\nα = {alpha:.1f}: ERROR - {e}")
            results.append({'alpha': alpha, 'async_value': 'ERROR', 'drift': np.inf})
    
    # Create comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE (like Table II)")
    print("="*60)
    print(f"{'Alpha':<10} {'Async Value':<15} {'Drift':<10} {'Status':<20}")
    print("-"*60)
    for r in results:
        status = "DIVERGES" if r['drift'] > 100 else "CONVERGES"
        print(f"{r['alpha']:<10.1f} {str(r['async_value']):<15} {r['drift']:<10.3f} {status:<20}")
    
    return results


# Run verification
print("PART 1: Math Verification")
print("="*60)
M, eigenvalues, history = verify_async_consensus_3_agents()

# Test different alphas
results = test_different_alphas()

plt.show()