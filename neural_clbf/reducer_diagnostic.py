"""
Diagnostic script to identify why near-full dimension reduction is failing
=========================================================================

Run this after creating your reducers to isolate the issue.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def diagnose_reducer(reducer, reducer_name, sys, X_train, detailed=True):
    """Comprehensive diagnostic for a single reducer."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {reducer_name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Latent dimension: {reducer.latent_dim}")
    print(f"Full dimension: {getattr(reducer, 'full_dim', 'Not set')}")
    
    # Test 1: Pure projection/reconstruction (no dynamics)
    print("\n1. PURE PROJECTION TEST (no dynamics)")
    print("-" * 40)
    
    # Use different test sizes
    test_sizes = [1, 10, 100]
    for n_test in test_sizes:
        X_test = torch.randn(n_test, sys.n_dims)
        
        # Forward and inverse
        Z = reducer.forward(X_test)
        X_recon = reducer.inverse(Z)
        
        # Compute errors
        abs_errors = (X_recon - X_test).norm(dim=1)
        rel_errors = abs_errors / (X_test.norm(dim=1) + 1e-10)
        
        print(f"  Batch size {n_test}:")
        print(f"    Absolute error: mean={abs_errors.mean():.2e}, max={abs_errors.max():.2e}")
        print(f"    Relative error: mean={rel_errors.mean():.2%}, max={rel_errors.max():.2%}")
        
        # This should be ~0 for near-full dimension!
        if abs_errors.mean() > 1e-6:
            print("    ⚠️  WARNING: Large projection error for near-full dimension!")
    
    # Test 2: Check projection matrices
    print("\n2. PROJECTION MATRIX ANALYSIS")
    print("-" * 40)
    
    if hasattr(reducer, 'P'):  # LCR style
        P = reducer.P
        Pi = reducer.Pi
        print(f"  P shape: {P.shape}")
        print(f"  P condition number: {torch.linalg.cond(P):.2e}")
        print(f"  P rank: {torch.linalg.matrix_rank(P).item()}")
        
        # Check projection property
        P_proj = P @ Pi
        proj_error = torch.norm(P_proj @ P - P)
        print(f"  ||P @ Pi @ P - P||: {proj_error:.2e}")
        
        # Check if Pi is a proper pseudo-inverse
        pinv_error = torch.norm(Pi @ P @ Pi - Pi)
        print(f"  ||Pi @ P @ Pi - Pi||: {pinv_error:.2e}")
        
    elif hasattr(reducer, 'proj'):  # OpInf style
        proj = reducer.proj
        print(f"  proj shape: {proj.shape}")
        print(f"  proj condition number: {torch.linalg.cond(proj):.2e}")
        print(f"  proj rank: {torch.linalg.matrix_rank(proj).item()}")
        
        # For d=19, proj should be nearly square
        if proj.shape[1] == proj.shape[0]:
            I_error = torch.norm(proj @ proj.T - torch.eye(proj.shape[0]))
            print(f"  ||proj @ proj.T - I||: {I_error:.2e}")
            
    elif hasattr(reducer, 'T'):  # SPR style
        T = reducer.T
        Ti = reducer.Ti
        print(f"  T shape: {T.shape}")
        print(f"  T condition number: {torch.linalg.cond(T):.2e}")
        print(f"  T rank: {torch.linalg.matrix_rank(T).item()}")
        
        # Check projection property
        T_proj = T @ Ti
        proj_error = torch.norm(T_proj @ T - T)
        print(f"  ||T @ Ti @ T - T||: {proj_error:.2e}")
    
    # Test 3: State space coverage
    print("\n3. STATE SPACE COVERAGE TEST")
    print("-" * 40)
    
    # Use actual training data
    X_sample = X_train[:min(50, len(X_train))]
    Z_sample = reducer.forward(X_sample)
    X_recon_sample = reducer.inverse(Z_sample)
    
    recon_errors = (X_recon_sample - X_sample).norm(dim=1)
    print(f"  On training data reconstruction:")
    print(f"    Mean error: {recon_errors.mean():.2e}")
    print(f"    Max error: {recon_errors.max():.2e}")
    print(f"    Std error: {recon_errors.std():.2e}")
    
    # Test 4: Jacobian test (if implemented)
    print("\n4. JACOBIAN CONSISTENCY TEST")
    print("-" * 40)
    
    try:
        x_test = X_train[0:1]  # Single point
        J = reducer.jacobian(x_test)
        print(f"  Jacobian shape: {J.shape}")
        
        # Finite difference check
        eps = 1e-6
        z_base = reducer.forward(x_test)
        J_fd = torch.zeros_like(J)
        
        for i in range(x_test.shape[1]):
            x_pert = x_test.clone()
            x_pert[0, i] += eps
            z_pert = reducer.forward(x_pert)
            J_fd[0, :, i] = (z_pert - z_base)[0] / eps
        
        J_error = torch.norm(J - J_fd) / (torch.norm(J) + 1e-10)
        print(f"  Finite difference error: {J_error:.2%}")
        
    except Exception as e:
        print(f"  Jacobian test failed: {e}")
    
    # Test 5: Specific physics tests
    print("\n5. PHYSICS PRESERVATION TESTS")
    print("-" * 40)
    
    # Test equilibrium preservation
    if hasattr(sys, 'goal_point'):
        x_eq = sys.goal_point.squeeze()
        z_eq = reducer.forward(x_eq.unsqueeze(0))
        x_eq_recon = reducer.inverse(z_eq).squeeze()
        eq_error = torch.norm(x_eq_recon - x_eq)
        print(f"  Equilibrium reconstruction error: {eq_error:.2e}")
    
    # Test energy preservation
    if hasattr(sys, 'energy_function'):
        E_orig = sys.energy_function(X_sample)
        E_recon = sys.energy_function(X_recon_sample)
        E_errors = torch.abs(E_recon - E_orig)
        E_rel_errors = E_errors / (torch.abs(E_orig) + 1e-10)
        print(f"  Energy preservation:")
        print(f"    Absolute error: mean={E_errors.mean():.2e}, max={E_errors.max():.2e}")
        print(f"    Relative error: mean={E_rel_errors.mean():.2%}, max={E_rel_errors.max():.2%}")
    
    # Test 6: Dynamics test (if applicable)
    if hasattr(reducer, 'dyn') and reducer.dyn is not None:
        print("\n6. LEARNED DYNAMICS TEST")
        print("-" * 40)
        
        # Check eigenvalues of dynamics matrix
        if hasattr(reducer.dyn, 'A'):
            A = reducer.dyn.A
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            print(f"  Max eigenvalue real part: {max_real:.4f}")
            print(f"  Dynamics matrix norm: {A.norm():.2e}")
    
    return {
        'reducer_name': reducer_name,
        'latent_dim': reducer.latent_dim,
        'projection_error': abs_errors.mean().item(),
        'energy_error': E_errors.mean().item() if 'E_errors' in locals() else None
    }


def compare_reducers(reducers_dict, sys, X_train):
    """Compare all reducers side by side."""
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    
    results = []
    for name, reducer in reducers_dict.items():
        if reducer is not None:
            result = diagnose_reducer(reducer, name, sys, X_train, detailed=False)
            results.append(result)
    
    # Print comparison table
    print(f"\n{'Method':<12} {'Dim':<6} {'Proj Error':<12} {'Energy Error':<12}")
    print("-" * 48)
    
    for r in results:
        energy_str = f"{r['energy_error']:.2e}" if r['energy_error'] else "N/A"
        print(f"{r['reducer_name']:<12} {r['latent_dim']:<6} "
              f"{r['projection_error']:<12.2e} {energy_str:<12}")


def plot_projection_errors(reducers_dict, sys, n_points=100):
    """Visualize projection errors across state space."""
    fig, axes = plt.subplots(1, len(reducers_dict), figsize=(15, 5))
    if len(reducers_dict) == 1:
        axes = [axes]
    
    # Generate random test points
    X_test = torch.randn(n_points, sys.n_dims)
    
    for ax, (name, reducer) in zip(axes, reducers_dict.items()):
        if reducer is None:
            continue
            
        # Compute reconstruction errors
        Z = reducer.forward(X_test)
        X_recon = reducer.inverse(Z)
        errors = (X_recon - X_test).norm(dim=1).cpu().numpy()
        
        # Plot histogram
        ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} (d={reducer.latent_dim})')
        ax.set_yscale('log')
        
        # Add statistics
        ax.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'Mean: {errors.mean():.2e}')
        ax.axvline(np.median(errors), color='green', linestyle='--',
                   label=f'Median: {np.median(errors):.2e}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('reducer_projection_errors.png', dpi=150)
    print("\nProjection error histograms saved to 'reducer_projection_errors.png'")


def minimal_dynamics_test(reducer, sys, x0, dt=0.01, n_steps=10):
    """Test just the dynamics computation without full simulation."""
    print(f"\nMINIMAL DYNAMICS TEST")
    print("-" * 40)
    
    x = x0.clone()
    z = reducer.forward(x)
    
    errors = []
    for step in range(n_steps):
        # Full space dynamics
        f_full = sys._f(x, sys.nominal_params)
        x_next_full = x + dt * f_full.squeeze()
        
        # Reduced space dynamics (manual)
        x_recon = reducer.inverse(z)
        f_recon = sys._f(x_recon, sys.nominal_params)
        
        # Project dynamics
        if hasattr(reducer, 'jacobian'):
            J = reducer.jacobian(x_recon)
            z_dot = torch.bmm(J, f_recon).squeeze()
        else:
            # Finite difference fallback
            eps = 1e-8
            z_plus = reducer.forward(x_recon + eps * f_recon.squeeze())
            z_dot = (z_plus - z) / eps
        
        z_next = z + dt * z_dot
        x_next_reduced = reducer.inverse(z_next)
        
        # Compare
        error = torch.norm(x_next_reduced - x_next_full).item()
        errors.append(error)
        print(f"  Step {step}: error = {error:.2e}")
        
        # Update for next iteration
        x = x_next_full
        z = reducer.forward(x)
    
    print(f"  Mean error over {n_steps} steps: {np.mean(errors):.2e}")


# Main diagnostic function to call
def run_full_diagnostic(sys, spr=None, opinf=None, lcr=None, X_train=None):
    """Run complete diagnostic suite."""
    
    if X_train is None:
        print("Warning: No training data provided, using random data")
        X_train = torch.randn(100, sys.n_dims)
    
    # Collect available reducers
    reducers = {}
    if spr is not None:
        reducers['SPR-18'] = spr
    if opinf is not None:
        reducers['OpInf-19'] = opinf
    if lcr is not None:
        reducers['LCR-18'] = lcr
    
    if not reducers:
        print("Error: No reducers provided!")
        return
    
    # Run individual diagnostics
    for name, reducer in reducers.items():
        diagnose_reducer(reducer, name, sys, X_train)
    
    # Comparative analysis
    compare_reducers(reducers, sys, X_train)
    
    # Visual analysis
    plot_projection_errors(reducers, sys)
    
    # Test dynamics for one reducer
    print("\n" + "="*80)
    print("DYNAMICS INTEGRATION TEST")
    print("="*80)
    
    x0 = sys.goal_point.squeeze()
    for name, reducer in reducers.items():
        print(f"\nTesting {name}:")
        minimal_dynamics_test(reducer, sys, x0.unsqueeze(0))
        break  # Just test one
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nIf projection errors are large (>1e-6) for near-full dimension,")
    print("the issue is in the reducer construction.")
    print("If projection is good but dynamics fail, the issue is in rollout_rom.")


if __name__ == "__main__":
    # Example usage:
    run_full_diagnostic(sys, spr=spr, opinf=opinf, lcr=lcr, X_train=X_train)