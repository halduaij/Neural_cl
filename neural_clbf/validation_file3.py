"""
Complete validation script using real network data with validated equilibrium
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer


def create_real_network_system_with_equilibrium():
    """
    Create the real network system with the validated equilibrium point.
    """
    # Real network parameters
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32)
    
    P = torch.tensor([-0.19983394, -0.25653884, -0.25191885, -0.10242008, -0.34510365,
                       0.23206371, 0.4404325, 0.5896664, 0.26257738, -0.36892462], 
                     dtype=torch.float32)
    
    K = torch.tensor([[18.217514, 1.1387165, 1.360604, 1.2791332, 0.572532,
                       1.2913872, 1.051677, 3.1750703, 1.7979614, 5.1949754],
                      [1.1387165, 13.809675, 2.2804017, 0.7847816, 0.3512633,
                       0.7922997, 0.6452313, 0.6032209, 0.5339053, 3.1851373],
                      [1.360604, 2.2804017, 14.325089, 1.0379374, 0.46457428,
                       1.0478808, 0.85337085, 0.7284146, 0.66612864, 3.2071328],
                      [1.2791332, 0.7847816, 1.0379374, 14.242994, 2.7465186,
                       2.0251245, 1.6492164, 0.7610195, 0.90347135, 1.6209806],
                      [0.572532, 0.3512633, 0.46457428, 2.7465186, 10.82826,
                       0.90643305, 0.7381789, 0.34062758, 0.4043881, 0.72554076],
                      [1.2913872, 0.7922997, 1.0478808, 2.0251245, 0.90643305,
                       16.046877, 4.1242394, 0.76831, 0.9121265, 1.6365094],
                      [1.051677, 0.6452313, 0.85337085, 1.6492164, 0.7381789,
                       4.1242394, 12.7116995, 0.62569463, 0.74281555, 1.3327368],
                      [3.1750703, 0.6032209, 0.7284146, 0.7610195, 0.34062758,
                       0.76831, 0.62569463, 11.512734, 1.4515272, 2.5534966],
                      [1.7979614, 0.5339053, 0.66612864, 0.90347135, 0.4043881,
                       0.9121265, 0.74281555, 1.4515272, 10.306445, 1.6531545],
                      [5.1949754, 3.1851373, 3.2071328, 1.6209806, 0.72554076,
                       1.6365094, 1.3327368, 2.5534966, 1.6531545, 25.04985]], 
                     dtype=torch.float32)
    
    # Validated equilibrium angles
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                               -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                              dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    
    return params, delta_star


def check_stability_simple(sys, device='cpu'):
    """
    Simple stability check that handles device issues.
    """
    try:
        # Make sure system is on CPU for linearization
        sys_cpu = SwingEquationSystem(sys.nominal_params, dt=sys.dt)
        sys_cpu.delta_star = sys.delta_star.cpu() if hasattr(sys, 'delta_star') else None
        
        # Linearize on CPU
        A = sys_cpu.linearise()
        if isinstance(A, tuple):
            A = A[0]
        
        # Get eigenvalues
        eigvals = torch.linalg.eigvals(A).real
        max_eig = eigvals.max().item()
        
        return max_eig, max_eig < 0
        
    except Exception as e:
        print(f"Stability check failed: {e}")
        return None, None


def run_validation_with_real_network():
    """
    Run complete validation using real network data and validated equilibrium.
    """
    print("="*80)
    print("DIMENSION REDUCTION VALIDATION - REAL POWER NETWORK")
    print("="*80)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Choose device - use CPU to avoid device issues
    device = torch.device("cpu")  # Force CPU for stability
    print(f"Using device: {device}")
    
    # Get parameters and equilibrium
    params, delta_star = create_real_network_system_with_equilibrium()
    
    # Move to device
    for key in params:
        params[key] = params[key].to(device)
    delta_star = delta_star.to(device)
    
    # Create system
    print("\n1. Creating system with validated equilibrium...")
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = delta_star  # Set the validated equilibrium
    sys.to(device)
    
    print(f"   ✓ System created (n_states={sys.n_dims}, n_controls={sys.n_controls})")
    print(f"   ✓ Using validated equilibrium point")
    
    # Check stability
    print("\n2. Checking stability at equilibrium...")
    max_eig, is_stable = check_stability_simple(sys, device)
    
    if max_eig is not None:
        print(f"   Max eigenvalue: {max_eig:.6f}")
        if is_stable:
            print("   ✓ System is STABLE")
        else:
            print("   ⚠ System appears unstable - may need parameter adjustment")
            # Try with increased damping
            print("   Trying with 2x damping...")
            params_2x = params.copy()
            params_2x['D'] = params['D'] * 2.0
            sys_2x = SwingEquationSystem(params_2x, dt=0.01)
            sys_2x.delta_star = delta_star
            max_eig_2x, is_stable_2x = check_stability_simple(sys_2x, device)
            if max_eig_2x is not None:
                print(f"   Max eigenvalue with 2x damping: {max_eig_2x:.6f}")
                if is_stable_2x:
                    print("   ✓ Using 2x damping for stability")
                    sys = sys_2x
                    params = params_2x
    
    # Collect training data
    print("\n3. Collecting training trajectories...")
    try:
        # Start from small perturbations around equilibrium
        n_traj = 100
        n_steps = 50
        
        # Create initial conditions near equilibrium
        x_eq = torch.zeros(sys.n_dims)
        # Convert delta_star to angle differences
        x_eq[:sys.N_NODES-1] = delta_star[0] - delta_star[1:]
        # Velocities are zero
        x_eq[sys.N_NODES-1:] = 0.0
        
        # Add small perturbations
        x_init = x_eq.unsqueeze(0).repeat(n_traj, 1)
        x_init += 0.05 * torch.randn_like(x_init)  # Small perturbations
        
        # Collect trajectories
        X_list = []
        Xdot_list = []
        
        for i in range(n_traj):
            x = x_init[i:i+1]
            for t in range(n_steps):
                # Zero control
                u = torch.zeros(1, sys.n_controls, device=device)
                
                # Get dynamics
                f = sys._f(x, params)
                g = sys._g(x, params)
                xdot = f.squeeze(-1)  # Remove last dimension
                
                # Store
                X_list.append(x.squeeze(0))
                Xdot_list.append(xdot.squeeze(0))
                
                # Euler step
                x = x + 0.01 * xdot
        
        data = {
            'X': torch.stack(X_list),
            'dXdt': torch.stack(Xdot_list)
        }
        
        print(f"   ✓ Collected {data['X'].shape[0]} snapshots")
        print(f"   State range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ Data collection failed: {e}")
        return
    
    # Test dimension reduction
    print("\n4. Testing dimension reduction...")
    
    d_max_values = [4, 6, 8, 10, 12, 14, 16]
    results = []
    
    for d_max in d_max_values:
        if d_max >= sys.n_dims:
            continue
        
        print(f"\n   Testing d_max={d_max}:")
        
        try:
            # Select reducer
            red = select_reducer(
                sys, data["X"], data["dXdt"],
                d_max=d_max,
                verbose=False
            )
            
            print(f"     Selected: {type(red).__name__}")
            print(f"     Latent dim: {red.latent_dim}")
            print(f"     Gamma: {red.gamma:.4f}")
            
            # Skip if gamma too large
            if red.gamma > 100:
                print(f"     Skipping validation (gamma too large)")
                continue
            
            # Quick validation
            print(f"     Running validation...")
            val_results = validate_reducer(
                sys, red,
                n_rollouts=10,  # Fewer rollouts for testing
                horizon=1.0,    # Shorter horizon
                dt=0.01,
                device=device
            )
            
            # Store results
            results.append({
                'reducer': type(red).__name__,
                'd': red.latent_dim,
                'gamma': red.gamma,
                'mean_err': val_results['mean_error'].item(),
                'rel_err': val_results['relative_error'].item(),
                'success': val_results['success_rate'].item()
            })
            
            print(f"     Mean error: {val_results['mean_error']:.5f}")
            print(f"     Relative error: {val_results['relative_error']:.3%}")
            
        except Exception as e:
            print(f"     Failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if results:
        print(f"\nSuccessfully tested {len(results)} configurations:")
        print(f"{'Reducer':<25} {'Dim':<5} {'Gamma':<8} {'Mean Err':<10} {'Rel Err':<8}")
        print("-"*66)
        
        for r in sorted(results, key=lambda x: x['mean_err']):
            print(f"{r['reducer']:<25} {r['d']:<5} {r['gamma']:<8.3f} "
                  f"{r['mean_err']:<10.5f} {r['rel_err']:<8.1%}")
        
        best = min(results, key=lambda x: x['mean_err'])
        print(f"\nBest configuration:")
        print(f"  {best['reducer']} with d={best['d']} (error={best['mean_err']:.5f})")
        
    else:
        print("\nNo successful validations. Consider:")
        print("  - Increasing damping further")
        print("  - Using smaller perturbations for training data")
        print("  - Adjusting d_max values")
    
    return results, sys


if __name__ == "__main__":
    # Run the validation
    results, sys = run_validation_with_real_network()