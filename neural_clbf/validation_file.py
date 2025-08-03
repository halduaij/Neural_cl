import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.manager import select_reducer
from neural_clbf.eval.reduction_validation import validate_reducer
import time

def run_comprehensive_validation(n_nodes=10, n_rollouts=50, horizon=5.0):
    """
    Run a comprehensive validation of dimension reduction for swing equation system.
    
    Args:
        n_nodes: Number of nodes in the power system
        n_rollouts: Number of validation trajectories
        horizon: Time horizon for validation (seconds)
    """
    print("="*80)
    print(f"COMPREHENSIVE DIMENSION REDUCTION VALIDATION")
    print(f"System: {n_nodes}-node power system")
    print(f"Validation: {n_rollouts} rollouts, {horizon}s horizon")
    print("="*80)
    
    # Create system with realistic parameters
    print("\n1. Creating power system...")
    params = dict(
        M=torch.ones(n_nodes) * 2.0 + torch.randn(n_nodes) * 0.2,  # Inertia with variation
        D=torch.ones(n_nodes) * 0.1 + torch.randn(n_nodes) * 0.01,  # Damping
        P=torch.randn(n_nodes) * 0.1,  # Random power injections
        K=torch.ones(n_nodes, n_nodes) * 0.5,  # Coupling matrix
    )
    
    # Make K symmetric and zero diagonal
    params['K'] = (params['K'] + params['K'].T) / 2
    params['K'].fill_diagonal_(0)
    
    sys = SwingEquationSystem(params, dt=0.01)
    print(f"   ✓ System dimensions: {sys.n_dims} (full state)")
    print(f"   ✓ Number of controls: {sys.n_controls}")
    
    # Collect training data
    print("\n2. Collecting training trajectories...")
    start_time = time.time()
    data = sys.collect_random_trajectories(
        N_traj=100, 
        T_steps=50,
        control_excitation=0.1,
        return_derivative=True
    )
    collect_time = time.time() - start_time
    print(f"   ✓ Collected {data['X'].shape[0]} snapshots in {collect_time:.2f}s")
    print(f"   ✓ State range: [{data['X'].min():.3f}, {data['X'].max():.3f}]")
    
    # Test different maximum dimensions
    d_max_values = [4, 8, 12, 16]
    results_by_d = {}
    
    for d_max in d_max_values:
        print(f"\n3. Selecting reducer with d_max={d_max}...")
        start_time = time.time()
        try:
            red = select_reducer(sys, data["X"], data["dXdt"], d_max=d_max)
            select_time = time.time() - start_time
            
            print(f"   ✓ Selected: {type(red).__name__}")
            print(f"   ✓ Latent dimension: {red.latent_dim}")
            print(f"   ✓ Gamma (robustness): {red.gamma:.6f}")
            print(f"   ✓ Selection time: {select_time:.2f}s")
            
            # Validate the reducer
            print(f"\n4. Validating reducer (d={red.latent_dim})...")
            start_time = time.time()
            
            try:
                results = validate_reducer(
                    sys, 
                    red, 
                    n_rollouts=n_rollouts, 
                    horizon=horizon
                )
                validate_time = time.time() - start_time
                
                # Store results
                results_by_d[red.latent_dim] = {
                    'reducer_type': type(red).__name__,
                    'gamma': red.gamma,
                    'results': results,
                    'validate_time': validate_time
                }
                
                print(f"   ✓ Validation completed in {validate_time:.2f}s")
                
                # Display key metrics
                print("\n   PERFORMANCE METRICS:")
                print("   " + "-"*50)
                
                if 'mean_error' in results:
                    print(f"   Mean trajectory error: {results['mean_error'].item():.6f}")
                
                if 'max_error' in results:
                    print(f"   Max trajectory error:  {results['max_error'].item():.6f}")
                
                if 'relative_error' in results:
                    print(f"   Relative error:        {results['relative_error'].item():.2%}")
                
                if 'energy_error' in results:
                    print(f"   Energy conservation:   {results['energy_error'].item():.6f}")
                
                # Compute compression ratio
                compression = red.latent_dim / sys.n_dims
                print(f"   Compression ratio:     {compression:.2%} ({red.latent_dim}/{sys.n_dims})")
                
                # Success rate (trajectories that didn't diverge)
                if 'success_rate' in results:
                    print(f"   Success rate:          {results['success_rate'].item():.2%}")
                
            except Exception as e:
                print(f"   ✗ Validation failed: {e}")
                results_by_d[red.latent_dim] = {
                    'reducer_type': type(red).__name__,
                    'gamma': red.gamma,
                    'error': str(e)
                }
                
        except Exception as e:
            print(f"   ✗ Reducer selection failed: {e}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY: DIMENSION REDUCTION PERFORMANCE")
    print("="*80)
    
    print(f"\n{'Dim':<6} {'Type':<25} {'Gamma':<10} {'Mean Err':<12} {'Max Err':<12} {'Rel Err':<10}")
    print("-"*85)
    
    for d, info in sorted(results_by_d.items()):
        if 'error' in info:
            print(f"{d:<6} {info['reducer_type']:<25} {info['gamma']:<10.6f} {'FAILED':<12} {'FAILED':<12} {'FAILED':<10}")
        else:
            r = info['results']
            mean_err = r.get('mean_error', torch.tensor(float('nan'))).item()
            max_err = r.get('max_error', torch.tensor(float('nan'))).item()
            rel_err = r.get('relative_error', torch.tensor(float('nan'))).item()
            print(f"{d:<6} {info['reducer_type']:<25} {info['gamma']:<10.6f} {mean_err:<12.6f} {max_err:<12.6f} {rel_err:<10.2%}")
    
    # Plot results if we have them
    if len(results_by_d) > 1:
        plot_validation_results(results_by_d, sys.n_dims)
    
    return results_by_d


def plot_validation_results(results_by_d, full_dim):
    """Create visualization of validation results."""
    dims = sorted([d for d in results_by_d.keys() if 'results' in results_by_d[d]])
    
    if len(dims) < 2:
        print("\nNot enough successful validations to plot.")
        return
    
    # Extract metrics
    mean_errors = []
    max_errors = []
    rel_errors = []
    
    for d in dims:
        r = results_by_d[d]['results']
        mean_errors.append(r.get('mean_error', torch.tensor(float('nan'))).item())
        max_errors.append(r.get('max_error', torch.tensor(float('nan'))).item())
        rel_errors.append(r.get('relative_error', torch.tensor(float('nan'))).item())
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dimension Reduction Performance Analysis', fontsize=16)
    
    # Plot 1: Error vs dimension
    ax = axes[0, 0]
    ax.semilogy(dims, mean_errors, 'b-o', label='Mean Error', linewidth=2, markersize=8)
    ax.semilogy(dims, max_errors, 'r--s', label='Max Error', linewidth=2, markersize=8)
    ax.set_xlabel('Reduced Dimension')
    ax.set_ylabel('Trajectory Error')
    ax.set_title('Reconstruction Error vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Relative error and compression
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    line1 = ax.plot(dims, [100*e for e in rel_errors], 'g-^', label='Relative Error (%)', linewidth=2, markersize=8)
    ax.set_xlabel('Reduced Dimension')
    ax.set_ylabel('Relative Error (%)', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    
    compressions = [100 * d / full_dim for d in dims]
    line2 = ax2.plot(dims, compressions, 'm:d', label='Compression (%)', linewidth=2, markersize=8)
    ax2.set_ylabel('Compression Ratio (%)', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    
    ax.set_title('Relative Error vs Compression')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Reducer types
    ax = axes[1, 0]
    reducer_types = [results_by_d[d]['reducer_type'] for d in dims]
    unique_types = list(set(reducer_types))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    
    for i, rtype in enumerate(unique_types):
        dims_for_type = [d for d in dims if results_by_d[d]['reducer_type'] == rtype]
        errors_for_type = [mean_errors[dims.index(d)] for d in dims_for_type]
        ax.semilogy(dims_for_type, errors_for_type, 'o-', label=rtype, 
                   color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Reduced Dimension')
    ax.set_ylabel('Mean Error')
    ax.set_title('Performance by Reducer Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Time vs accuracy trade-off
    ax = axes[1, 1]
    val_times = [results_by_d[d]['validate_time'] for d in dims]
    
    # Create a scatter plot with point size proportional to dimension
    scatter = ax.scatter(val_times, mean_errors, s=[100*d/full_dim for d in dims], 
                        c=dims, cmap='viridis', alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Validation Time (s)')
    ax.set_ylabel('Mean Error')
    ax.set_title('Time vs Accuracy Trade-off')
    ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reduced Dimension')
    
    # Add annotations
    for i, d in enumerate(dims):
        ax.annotate(f'd={d}', (val_times[i], mean_errors[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('dimension_reduction_performance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Performance plots saved to 'dimension_reduction_performance.png'")
    plt.show()


def test_specific_reducer(sys, reducer_type='LyapCoherencyReducer', n_groups=3):
    """Test a specific reducer type with detailed diagnostics."""
    print(f"\nTesting {reducer_type} with {n_groups} groups...")
    
    # Collect data
    data = sys.collect_random_trajectories(50, return_derivative=True)
    
    # Create specific reducer
    if reducer_type == 'LyapCoherencyReducer':
        from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
        red = LyapCoherencyReducer(sys, n_groups, data["X"])
    else:
        red = select_reducer(sys, data["X"], data["dXdt"], d_max=2*n_groups)
    
    print(f"Reducer details:")
    print(f"  - Latent dimension: {red.latent_dim}")
    print(f"  - Full dimension: {sys.n_dims}")
    print(f"  - Compression: {red.latent_dim/sys.n_dims:.2%}")
    
    # Test forward/inverse mapping
    x_test = data["X"][:10]
    z = red.forward(x_test)
    x_recon = red.inverse(z)
    
    recon_error = (x_test - x_recon).norm(dim=1).mean()
    print(f"  - Reconstruction error: {recon_error:.6f}")
    
    # Test energy preservation
    E_orig = sys.energy_function(x_test)
    E_recon = sys.energy_function(x_recon)
    energy_error = (E_orig - E_recon).abs().mean()
    print(f"  - Energy error: {energy_error:.6f}")
    
    return red


if __name__ == "__main__":
    # Run the comprehensive validation
    results = run_comprehensive_validation(
        n_nodes=10,     # 10-node power system
        n_rollouts=20,  # 20 validation trajectories
        horizon=3.0     # 3 second horizon
    )
    
    # Optional: test specific reducer
    # params = dict(
    #     M=torch.ones(10) * 2.0,
    #     D=torch.ones(10) * 0.1,
    #     P=torch.zeros(10),
    #     K=torch.ones(10, 10) * 0.5,
    # )
    # sys = SwingEquationSystem(params)
    # test_specific_reducer(sys, 'LyapCoherencyReducer', n_groups=3)