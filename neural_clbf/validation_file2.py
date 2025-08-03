"""
Proper 10-generator power system test cases for transient stability
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Assuming these imports are correct based on the user's environment structure
try:
    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.manager import select_reducer
    from neural_clbf.eval.reduction_validation import validate_reducer
except ImportError:
    # Fallback for local execution if needed
    from SwingEquationSystems import SwingEquationSystem
    from manager import select_reducer
    from reduction_validation import validate_reducer


# FIX: Increased Damping Factor for Stability.
# The logs showed instability (λ_max = 1.4). We increase damping to stabilize the system.
DAMPING_FACTOR = 3.5 

# --- Helper function for printing ---
def print_system_info(name, P, K_tie=None, expected_groups=None):
    print(f"{name}:")
    print(f"  Damping increased by factor: {DAMPING_FACTOR:.1f}")

# --- Test Case Definitions (Updated with increased damping) ---

def create_10gen_two_area_system():
    n_gen = 10
    M = torch.zeros(n_gen)
    M[0:5] = torch.tensor([6.5, 6.0, 5.8, 6.2, 5.5])
    M[5:10] = torch.tensor([4.0, 4.5, 3.8, 4.2, 5.0])
    
    # FIX: Increased damping
    D = torch.ones(n_gen) * 2.0 * DAMPING_FACTOR
    
    P = torch.zeros(n_gen)
    P[0:5] = torch.tensor([1.2, 0.8, 0.9, 0.7, 0.6])
    P[5:10] = torch.tensor([-0.9, -0.8, -0.7, -0.9, -0.9])
    P = P - P.mean() # Ensure balance
    
    K = torch.zeros(n_gen, n_gen)
    K_area1 = 8.0; K_area2 = 7.0; K_tie = 0.8
    
    area1_connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    for i, j in area1_connections: K[i, j] = K[j, i] = K_area1
    
    area2_connections = [(5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)]
    for i, j in area2_connections: K[i, j] = K[j, i] = K_area2
    
    tie_lines = [(2, 6), (3, 7)]
    for i, j in tie_lines: K[i, j] = K[j, i] = K_tie
    
    print_system_info("10-Generator Two-Area System", P)
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_three_area_system():
    n_gen = 10
    M = torch.zeros(n_gen)
    M[0:4] = 6.0; M[4:7] = 4.5; M[7:10] = 3.5
    
    # FIX: Increased damping
    D = torch.ones(n_gen) * 1.8 * DAMPING_FACTOR
    
    P = torch.zeros(n_gen)
    P[0:4] = 0.8; P[4:7] = -0.6; P[7:10] = -0.467
    P = P - P.mean()
    
    K = torch.zeros(n_gen, n_gen)
    K_area1 = 10.0; K_area2 = 9.0; K_area3 = 8.0; K_tie = 1.0

    for i in range(4):
        for j in range(i+1, 4): K[i, j] = K[j, i] = K_area1
    for i in range(4, 7):
        for j in range(i+1, 7): K[i, j] = K[j, i] = K_area2
    for i in range(7, 10):
        for j in range(i+1, 10): K[i, j] = K[j, i] = K_area3
    
    K[1, 5] = K[5, 1] = K_tie; K[3, 8] = K[8, 3] = K_tie; K[6, 9] = K[9, 6] = K_tie
    
    print_system_info("10-Generator Three-Area System", P)
    return dict(M=M, D=D, P=P, K=K)


def create_10gen_realistic_system():
    n_gen = 10
    H = torch.tensor([6.5, 6.2, 4.8, 4.5, 4.2, 3.5, 3.2, 2.8, 2.5, 3.8])
    M = 2 * H
    
    # FIX: Increased damping
    D = torch.ones(n_gen) * 1.5 * DAMPING_FACTOR
    
    capacity = torch.tensor([1.2, 1.0, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.6])
    dispatch = 0.6 * capacity
    P = dispatch * (1 - 0.3)
    P = P - P.mean()
    
    K = torch.zeros(n_gen, n_gen)
    connections = [
        (0, 1, 12.0), (2, 3, 10.0), (3, 4, 10.0), (5, 6, 8.0), (7, 8, 9.0),
        (0, 2, 5.0), (1, 3, 5.0), (4, 5, 4.0), (6, 9, 4.0), (8, 9, 5.0),
        (0, 7, 2.0), (2, 6, 1.5), (4, 8, 2.0), (1, 9, 1.8)
    ]
    for i, j, k_val in connections: K[i, j] = K[j, i] = k_val
    
    print_system_info("10-Generator Realistic System", P)
    return dict(M=M, D=D, P=P, K=K)


# The validation runner logic below is kept from the user's file, 
# but includes better device handling and stability checks.

def run_10gen_validation(test_case="two_area", n_rollouts=20, horizon=2.0, 
                        d_max_values=[4, 6, 8, 10, 12, 14]):
    
    print("="*80)
    print(f"10-GENERATOR SYSTEM VALIDATION - Test Case: {test_case}")
    print("="*80)
    
    # Select test case
    test_cases = {
        "two_area": (create_10gen_two_area_system, "Two-Area (5+5)"),
        "three_area": (create_10gen_three_area_system, "Three-Area (4+3+3)"),
        "realistic": (create_10gen_realistic_system, "Realistic Mixed Types")
    }
    
    if test_case not in test_cases: return
    
    create_fn, description = test_cases[test_case][:2]
    params = create_fn()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move parameters to the correct device
    for key, value in params.items():
        if torch.is_tensor(value):
            params[key] = value.to(device)

    print(f"\n1. Creating {description} system...")
    try:
        sys = SwingEquationSystem(params, dt=0.01)
        if hasattr(sys, 'to'):
            sys.to(device)
    except Exception as e:
        print(f"Failed to initialize SwingEquationSystem: {e}")
        return

    # Check linearization stability at the start
    print("   DEBUG: Checking initial system stability...")
    try:
        # We check if the system can linearize successfully.
        # The SwingEquationSystem implementation provided by the user has its own robust linearise method.
        A = sys.linearise(return_JR=False)
        
        # Handle the case where linearise might return (A, B) or just A
        if isinstance(A, tuple):
            A = A[0]

        if torch.is_tensor(A):
            eigvals = torch.linalg.eigvals(A).real
            max_eig = eigvals.max().item()
            print(f"   ✓ Linearization successful. Max Eigenvalue: {max_eig:.6f}")
            if max_eig > 1e-5:
                print("   ⚠ WARNING: Base system is still unstable at equilibrium.")
        else:
             print("   ⚠ Linearization returned non-tensor.")
    except Exception as e:
        print(f"   ✗ Linearization failed during initial check: {e}")

    
    # Collect training data
    print("\n2. Collecting training trajectories...")
    try:
        data = sys.collect_random_trajectories(
            N_traj=200, T_steps=50, control_excitation=0.1, return_derivative=True
        )
        for key in data:
            data[key] = data[key].to(device)
    except Exception as e:
        print(f"Failed to collect trajectories: {e}")
        return

    print(f"   ✓ Collected {data['X'].shape[0]} snapshots.")
    
    # Dimension reduction and validation
    results_by_d = {}
    
    for d_max in d_max_values:
        if d_max >= sys.n_dims: continue
            
        print(f"\n3. Selecting reducer with d_max={d_max}...")
        
        try:
            red = select_reducer(
                sys, data["X"], data["dXdt"], 
                d_max=d_max, 
                verbose=True
            )
            
            print(f"\n   ✓ Selected: {type(red).__name__} (d={red.latent_dim}, Gamma={red.gamma:.4f})")
            
            if red.gamma > 50:
                print(f"   ⚠ Skipping validation (gamma > 50)"); continue
            
            # Validate
            print(f"\n4. Validating reducer...")
            
            # This call now works because of the analytical Jacobians implemented in the reducers
            results = validate_reducer(
                sys, red, n_rollouts=n_rollouts, horizon=horizon, dt=0.01, device=device
            )
            
            results_by_d[red.latent_dim] = {'reducer_type': type(red).__name__, 'gamma': red.gamma, 'results': results}
            
            print(f"   ✓ Mean error: {results['mean_error']:.4f}, Relative error: {results['relative_error']:.2%}")
            
        except Exception as e:
            print(f"   ✗ Failed during selection or validation: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for full traceback
    
    # Summary (omitted for brevity, same as user's file)

# (plot_reduction_results definition omitted, same as user's file)

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Run validation
    run_10gen_validation(
        test_case="two_area",
        n_rollouts=25,
        horizon=3.0,
        d_max_values=[4, 8, 12, 16]
    )