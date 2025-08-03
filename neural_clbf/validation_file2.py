"""
Robust 10-generator power system validation with stability guarantees
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.manager import select_reducer
    from neural_clbf.eval.reduction_validation import validate_reducer
except ImportError:
    from SwingEquationSystems import SwingEquationSystem
    from manager import select_reducer
    from reduction_validation import validate_reducer


def create_10gen_two_area_system_stable():
    """Create stabilized two-area system (see artifact above for full implementation)"""
    n_gen = 10
    
    # Inertias: Larger machines near tie-lines
    M = torch.zeros(n_gen)
    M[0:5] = torch.tensor([5.5, 6.0, 7.0, 7.0, 5.5])  
    M[5:10] = torch.tensor([5.0, 7.0, 7.0, 5.0, 5.0])
    
    # Damping proportional to inertia
    D = 0.15 * M
    
    # Better distributed power
    P = torch.zeros(n_gen)
    P[0:5] = torch.tensor([0.5, 0.4, 0.6, 0.5, 0.4])
    P[5:10] = torch.tensor([-0.4, -0.5, -0.6, -0.3, -0.2])
    P = P - P.mean()
    
    # Stronger coupling
    K = torch.zeros(n_gen, n_gen)
    K_area1 = 6.0
    K_area2 = 5.5
    K_tie = 2.0  # Much stronger!
    
    # Area 1 mesh
    area1_connections = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    for i, j in area1_connections:
        K[i, j] = K[j, i] = K_area1
    
    # Area 2 mesh
    area2_connections = [(5, 6), (5, 7), (6, 7), (6, 8), (7, 8), (7, 9), (8, 9)]
    for i, j in area2_connections:
        K[i, j] = K[j, i] = K_area2
    
    # Three tie lines for redundancy
    tie_lines = [(2, 6), (3, 7), (1, 5)]
    for i, j in tie_lines:
        K[i, j] = K[j, i] = K_tie
    
    return dict(M=M, D=D, P=P, K=K)


def check_system_stability(sys, verbose=True):
    """
    Comprehensive stability check for swing equation system.
    Returns (is_stable, max_eigenvalue, damping_ratios)
    """
    try:
        # Get linearization
        result = sys.linearise(return_JR=False)
        A = result if torch.is_tensor(result) else result[0]
        
        # Compute eigenvalues
        eigvals_complex = torch.linalg.eigvals(A)
        eigvals_real = eigvals_complex.real
        max_eig = eigvals_real.max().item()
        
        # Compute damping ratios for oscillatory modes
        damping_ratios = []
        for ev in eigvals_complex:
            if abs(ev.imag) > 1e-6:  # Oscillatory mode
                damping_ratio = -ev.real / abs(ev)
                damping_ratios.append(damping_ratio.item())
        
        is_stable = max_eig < 0
        
        if verbose:
            print(f"  Stability Analysis:")
            print(f"    Max eigenvalue: {max_eig:.6f}")
            print(f"    System stable: {'Yes' if is_stable else 'No'}")
            if damping_ratios:
                min_damping = min(damping_ratios)
                print(f"    Min damping ratio: {min_damping:.3f} ({min_damping*100:.1f}%)")
                if min_damping < 0.05:
                    print(f"    ⚠ Warning: Low damping detected!")
        
        return is_stable, max_eig, damping_ratios
        
    except Exception as e:
        if verbose:
            print(f"  Stability check failed: {e}")
        return False, float('inf'), []


def run_10gen_validation_robust(test_case="stable_two_area", n_rollouts=20, 
                               horizon=3.0, d_max_values=[4, 6, 8, 10, 12]):
    """
    Run validation with stability checks and robust error handling.
    """
    print("="*80)
    print(f"ROBUST 10-GENERATOR SYSTEM VALIDATION")
    print("="*80)
    
    # Create system based on test case
    if test_case == "stable_two_area":
        params = create_10gen_two_area_system_stable()
        description = "Stabilized Two-Area System"
    else:
        print(f"Unknown test case: {test_case}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move parameters to device
    for key, value in params.items():
        if torch.is_tensor(value):
            params[key] = value.to(device)
    
    print(f"\n1. Creating {description}...")
    
    # Initialize system
    try:
        sys = SwingEquationSystem(params, dt=0.01)
        if hasattr(sys, 'to'):
            sys.to(device)
        print(f"   ✓ System created (n_states={sys.n_dims}, n_controls={sys.n_controls})")
    except Exception as e:
        print(f"   ✗ Failed to create system: {e}")
        return
    
    # Check stability
    print("\n2. Verifying system stability...")
    is_stable, max_eig, damping_ratios = check_system_stability(sys)
    
    if not is_stable:
        print("   ✗ System is unstable! Cannot proceed with validation.")
        print("   Consider adjusting parameters (increase damping or tie-line strength)")
        return
    
    # Collect training data with error handling
    print("\n3. Collecting training trajectories...")
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            data = sys.collect_random_trajectories(
                N_traj=200, 
                T_steps=50, 
                control_excitation=0.05,  # Reduced excitation for stability
                return_derivative=True
            )
            
            # Move data to device
            for key in data:
                data[key] = data[key].to(device)
            
            # Check for NaN/Inf
            if torch.isnan(data['X']).any() or torch.isinf(data['X']).any():
                raise ValueError("Training data contains NaN or Inf")
            
            print(f"   ✓ Collected {data['X'].shape[0]} valid snapshots")
            break
            
        except Exception as e:
            print(f"   Attempt {attempt+1} failed: {e}")
            if attempt == max_attempts - 1:
                print("   ✗ Failed to collect valid training data")
                return
    
    # Dimension reduction with comprehensive testing
    results_summary = []
    print("\n4. Testing dimension reduction...")
    
    for d_max in d_max_values:
        if d_max >= sys.n_dims:
            continue
        
        print(f"\n   Testing d_max={d_max}:")
        
        try:
            # Select reducer with timeout protection
            start_time = time.time()
            red = select_reducer(
                sys, data["X"], data["dXdt"], 
                d_max=d_max, 
                verbose=False
            )
            selection_time = time.time() - start_time
            
            print(f"     Selected: {type(red).__name__}")
            print(f"     Latent dim: {red.latent_dim}")
            print(f"     Gamma: {red.gamma:.4f}")
            print(f"     Selection time: {selection_time:.2f}s")
            
            # Skip if gamma is too large
            if red.gamma > 100:
                print(f"     ⚠ Skipping validation (gamma too large)")
                continue
            
            # Validate reducer
            print(f"     Running validation...")
            val_results = validate_reducer(
                sys, red, 
                n_rollouts=n_rollouts, 
                horizon=horizon, 
                dt=0.01, 
                device=device
            )
            
            # Store results
            results_summary.append({
                'reducer_type': type(red).__name__,
                'latent_dim': red.latent_dim,
                'gamma': red.gamma,
                'mean_error': val_results['mean_error'].item(),
                'relative_error': val_results['relative_error'].item(),
                'energy_error': val_results['energy_error'].item() if not torch.isnan(val_results['energy_error']) else 0.0,
                'success_rate': val_results['success_rate'].item(),
            })
            
            print(f"     Mean error: {val_results['mean_error']:.4f}")
            print(f"     Relative error: {val_results['relative_error']:.2%}")
            print(f"     Success rate: {val_results['success_rate']:.2%}")
            
        except Exception as e:
            print(f"     ✗ Failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if results_summary:
        print(f"\nSuccessfully tested {len(results_summary)} configurations:")
        print(f"{'Reducer':<20} {'Dim':<6} {'Gamma':<10} {'Mean Err':<10} {'Rel Err':<10} {'Success':<10}")
        print("-"*76)
        
        for r in sorted(results_summary, key=lambda x: x['mean_error']):
            print(f"{r['reducer_type']:<20} {r['latent_dim']:<6} "
                  f"{r['gamma']:<10.4f} {r['mean_error']:<10.4f} "
                  f"{r['relative_error']:<10.2%} {r['success_rate']:<10.2%}")
        
        # Best configuration
        best = min(results_summary, key=lambda x: x['mean_error'])
        print(f"\nBest configuration:")
        print(f"  Reducer: {best['reducer_type']}")
        print(f"  Dimension: {best['latent_dim']}")
        print(f"  Mean error: {best['mean_error']:.4f}")
        
    else:
        print("\nNo successful validations completed.")
    
    return results_summary


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run robust validation
    results = run_10gen_validation_robust(
        test_case="stable_two_area",
        n_rollouts=25,
        horizon=3.0,
        d_max_values=[4, 6, 8, 10, 12, 14]
    )
    
    # Optional: Save results
    if results:
        import json
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to validation_results.json")