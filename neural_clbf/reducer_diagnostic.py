"""
Diagnostic tool for debugging reducer issues
===========================================

Run this to diagnose problems with dimension reduction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def diagnose_reducer(reducer, X_test, system=None, name="Reducer"):
    """
    Comprehensive diagnostic for a dimension reducer.
    
    Args:
        reducer: The reducer to diagnose
        X_test: Test data (n_samples, n_dims)
        system: Optional system object for additional checks
        name: Name for display
    """
    print("\n" + "="*60)
    print(f"DIAGNOSING: {name}")
    print("="*60)
    
    print(f"Latent dimension: {reducer.latent_dim}")
    print(f"Full dimension: {getattr(reducer, 'full_dim', X_test.shape[1])}")
    
    # 1. Pure projection test
    print("\n1. PURE PROJECTION TEST (no dynamics)")
    print("-"*40)
    
    for batch_size in [1, 10, 100]:
        if batch_size > X_test.shape[0]:
            continue
            
        X_batch = X_test[:batch_size]
        Z_batch = reducer.forward(X_batch)
        X_recon = reducer.inverse(Z_batch)
        
        abs_error = (X_batch - X_recon).norm(dim=1)
        rel_error = abs_error / (X_batch.norm(dim=1) + 1e-8)
        
        print(f"  Batch size {batch_size}:")
        print(f"    Absolute error: mean={abs_error.mean():.2e}, max={abs_error.max():.2e}")
        print(f"    Relative error: mean={rel_error.mean():.2%}, max={rel_error.max():.2%}")
        
        if reducer.latent_dim >= 0.9 * X_batch.shape[1] and rel_error.max() > 1e-3:
            print(f"    ⚠️  WARNING: Large projection error for near-full dimension!")
    
    # 2. Projection matrix analysis
    print("\n2. PROJECTION MATRIX ANALYSIS")
    print("-"*40)
    
    if hasattr(reducer, 'proj'):  # OpInf style
        P = reducer.proj
        print(f"  proj shape: {P.shape}")
        print(f"  proj condition number: {torch.linalg.cond(P):.2e}")
        print(f"  proj rank: {torch.linalg.matrix_rank(P).item()}")
        
        # Check orthogonality
        ortho_error = torch.norm(P.T @ P - torch.eye(P.shape[1]))
        print(f"  ||proj.T @ proj - I||: {ortho_error:.2e}")
        
    elif hasattr(reducer, 'P'):  # LCR style
        P = reducer.P
        Pi = reducer.Pi
        print(f"  P shape: {P.shape}")
        print(f"  P condition number: {torch.linalg.cond(P):.2e}")
        print(f"  P rank: {torch.linalg.matrix_rank(P).item()}")
        
        # Check projection properties
        PPi = P @ Pi
        print(f"  ||P @ Pi @ P - P||: {torch.norm(PPi @ P - P):.2e}")
        print(f"  ||Pi @ P @ Pi - Pi||: {torch.norm(Pi @ PPi - Pi):.2e}")
        
    elif hasattr(reducer, 'T'):  # SPR style
        T = reducer.T
        Ti = reducer.Ti
        print(f"  T shape: {T.shape}")
        print(f"  T condition number: {torch.linalg.cond(T):.2e}")
        print(f"  T rank: {torch.linalg.matrix_rank(T).item()}")
        
        # Check properties
        print(f"  ||T @ Ti @ T - T||: {torch.norm(T @ Ti @ T - T):.2e}")
    
    # 3. State space coverage
    print("\n3. STATE SPACE COVERAGE TEST")
    print("-"*40)
    
    # Test on training data span
    X_mean = X_test.mean(0)
    X_centered = X_test - X_mean
    
    # Project and reconstruct centered data
    if hasattr(reducer, 'μ'):
        X_test_centered = X_test - reducer.μ
    else:
        X_test_centered = X_test
        
    Z_test = reducer.forward(X_test)
    X_recon_test = reducer.inverse(Z_test)
    
    recon_errors = (X_test - X_recon_test).norm(dim=1)
    print(f"  On training data reconstruction:")
    print(f"    Mean error: {recon_errors.mean():.2e}")
    print(f"    Max error: {recon_errors.max():.2e}")
    print(f"    Std error: {recon_errors.std():.2e}")
    
    # 4. Jacobian test
    print("\n4. JACOBIAN CONSISTENCY TEST")
    print("-"*40)
    
    x_test = X_test[0:1].clone().requires_grad_(True)
    
    # Analytical Jacobian
    J_analytical = reducer.jacobian(x_test)
    print(f"  Jacobian shape: {J_analytical.shape}")
    
    # Finite difference check
    eps = 1e-5
    z_base = reducer.forward(x_test)
    J_fd = torch.zeros_like(J_analytical)
    
    for i in range(x_test.shape[1]):
        x_pert = x_test.clone()
        x_pert[0, i] += eps
        z_pert = reducer.forward(x_pert)
        J_fd[0, :, i] = (z_pert - z_base)[0] / eps
    
    jac_error = (J_analytical - J_fd).norm() / (J_fd.norm() + 1e-8)
    print(f"  Finite difference error: {jac_error:.2%}")
    
    # 5. Physics preservation (if system available)
    if system is not None:
        print("\n5. PHYSICS PRESERVATION TESTS")
        print("-"*40)
        
        # Equilibrium preservation
        if hasattr(system, 'goal_point'):
            x_eq = system.goal_point.squeeze()
            z_eq = reducer.forward(x_eq.unsqueeze(0))
            x_eq_recon = reducer.inverse(z_eq).squeeze()
            eq_error = (x_eq - x_eq_recon).norm()
            print(f"  Equilibrium reconstruction error: {eq_error:.2e}")
        
        # Energy preservation
        if hasattr(system, 'energy_function'):
            E_orig = system.energy_function(X_test)
            E_recon = system.energy_function(X_recon_test)
            
            if E_orig.shape == E_recon.shape:
                E_abs_error = (E_orig - E_recon).abs()
                E_rel_error = E_abs_error / (E_orig.abs() + 1e-8)
                
                print(f"  Energy preservation:")
                print(f"    Absolute error: mean={E_abs_error.mean():.2e}, max={E_abs_error.max():.2e}")
                print(f"    Relative error: mean={E_rel_error.mean():.2%}, max={E_rel_error.max():.2%}")
    
    # 6. Dynamics test (if available)
    if hasattr(reducer, 'dyn') and reducer.dyn is not None:
        print("\n6. LEARNED DYNAMICS TEST")
        print("-"*40)
        
        # Check eigenvalues
        if hasattr(reducer.dyn, 'A'):
            A = reducer.dyn.A
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            print(f"  Max eigenvalue real part: {max_real:.4f}")
            
            # Check discrete stability
            dt = 0.01
            A_discrete = torch.eye(A.shape[0]) + dt * A
            rho = torch.linalg.eigvals(A_discrete).abs().max().item()
            print(f"  Discrete spectral radius (dt={dt}): {rho:.6f}")
            
            # Dynamics matrix properties
            print(f"  Dynamics matrix norm: {A.norm():.2e}")
    
    return {
        'name': name,
        'latent_dim': reducer.latent_dim,
        'projection_error': recon_errors.mean().item(),
        'max_projection_error': recon_errors.max().item(),
    }


def compare_reducers(reducers, X_test, system=None):
    """
    Compare multiple reducers side by side.
    
    Args:
        reducers: Dict of {name: reducer}
        X_test: Test data
        system: Optional system object
    """
    results = []
    
    for name, reducer in reducers.items():
        if reducer is not None:
            result = diagnose_reducer(reducer, X_test, system, name)
            results.append(result)
    
    # Summary table
    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY")
    print("="*60)
    
    print(f"{'Method':<15} {'Dim':<5} {'Proj Error':<12} {'Max Error':<12}")
    print("-"*50)
    
    for res in results:
        print(f"{res['name']:<15} {res['latent_dim']:<5} "
              f"{res['projection_error']:<12.2e} {res['max_projection_error']:<12.2e}")
    
    return results


def plot_projection_errors(reducers, X_test, save_path=None):
    """
    Plot histogram of projection errors for each reducer.
    """
    plt.figure(figsize=(12, 4))
    
    n_reducers = len([r for r in reducers.values() if r is not None])
    plot_idx = 1
    
    for name, reducer in reducers.items():
        if reducer is None:
            continue
            
        plt.subplot(1, n_reducers, plot_idx)
        
        # Compute errors
        Z = reducer.forward(X_test)
        X_recon = reducer.inverse(Z)
        errors = (X_test - X_recon).norm(dim=1).cpu().numpy()
        
        # Plot histogram
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.title(f'{name}\n(d={reducer.latent_dim})')
        plt.yscale('log')
        
        # Add statistics
        plt.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'Mean: {errors.mean():.2e}')
        plt.axvline(np.percentile(errors, 95), color='orange', linestyle='--',
                   label=f'95%: {np.percentile(errors, 95):.2e}')
        plt.legend()
        
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nProjection error histograms saved to '{save_path}'")
    
    plt.show()


def test_dynamics_stability(reducer, X_test, dt=0.01, n_steps=100):
    """
    Test the stability of learned dynamics through short rollouts.
    """
    print(f"\nTesting {type(reducer).__name__}:")
    
    if not hasattr(reducer, 'dyn') or reducer.dyn is None:
        print("  No learned dynamics to test")
        return
    
    print("\nMINIMAL DYNAMICS TEST")
    print("-"*40)
    
    # Pick a few test points
    n_tests = min(10, X_test.shape[0])
    
    stable_count = 0
    
    for i in range(n_tests):
        x0 = X_test[i:i+1]
        z = reducer.forward(x0)
        
        stable = True
        for t in range(n_steps):
            z_dot = reducer.dyn.forward(z)
            z_next = z + dt * z_dot
            
            if not torch.isfinite(z_next).all() or z_next.norm() > 1e5:
                stable = False
                break
                
            z = z_next
        
        if stable:
            stable_count += 1
            final_error = (reducer.inverse(z) - x0).norm().item()
            print(f"  Step {i}: error = {final_error:.2e}")
        else:
            print(f"  Step {i}: UNSTABLE at t={t}")
    
    print(f"  Stability rate: {stable_count}/{n_tests}")
    

def run_full_diagnostic(system, spr=None, opinf=None, lcr=None, X_train=None):
    """
    Run complete diagnostic on all reducers.
    """
    if X_train is None:
        # Generate test data
        print("Generating test data...")
        X_train = system.sample_state_space(1000)
    
    # Collect non-None reducers
    reducers = {}
    if opinf is not None:
        reducers['OpInf-19'] = opinf
    if lcr is not None:
        reducers['LCR-18'] = lcr
    if spr is not None:
        reducers['SPR-18'] = spr
    
    # Run comparative diagnostic
    results = compare_reducers(reducers, X_train, system)
    
    # Plot errors
    if len(reducers) > 0:
        plot_projection_errors(reducers, X_train, 'reducer_projection_errors.png')
    
    # Test dynamics
    print("\n" + "="*80)
    print("DYNAMICS INTEGRATION TEST")
    print("="*80)
    
    for name, reducer in reducers.items():
        if reducer is not None:
            test_dynamics_stability(reducer, X_train[:10])
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    print("\nIf projection errors are large (>1e-6) for near-full dimension,")
    print("the issue is in the reducer construction.")
    print("If projection is good but dynamics fail, the issue is in rollout_rom.")
    
    return results