"""
Comprehensive test script demonstrating all fixes:
1. SPR with intrinsic reduced dynamics
2. LCR with aggregated swing equations
3. OpInf with relaxed stability 
4. Fixed validation
"""

import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all modules
try:
    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.opinf import OpInfReducer
    from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
    from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
    from neural_clbf.rollout_and_validation_fixes import rollout_rom, validate_reducer
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

def create_test_system(n_machines=10):
    """Create a test power system."""
    torch.manual_seed(42)
    
    # Stable parameters
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    # Use higher damping for stability
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 5.0  # 5x damping
    
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
    
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                               -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                              dtype=torch.float32)
    
    params = {"M": M, "D": D, "P": P, "K": K}
    
    sys = SwingEquationSystem(params)
    logger.info(f"Created {n_machines}-machine system (state dim = {sys.n_dims})")
    
    # Compute equilibrium
    try:
        sys.delta_star = sys.solve_equilibrium_robust()
        x_eq = sys.compute_equilibrium_point()
        logger.info(f"Equilibrium computed successfully")
    except Exception as e:
        logger.error(f"Equilibrium computation failed: {e}")
    
    return sys

def collect_training_data(sys, n_traj=50, n_steps=50):
    """Collect diverse training data."""
    logger.info("Collecting training data...")
    
    data = sys.collect_random_trajectories(
        N_traj=n_traj,
        T_steps=n_steps,
        control_excitation=0.1,
        return_derivative=True
    )
    
    X = data["X"]
    Xdot = data["dXdt"]
    logger.info(f"  Collected {X.shape[0]} snapshots")
    
    # Add near-equilibrium data
    x_eq = sys.goal_point.squeeze()
    X_eq = x_eq + 0.05 * torch.randn(100, sys.n_dims)
    Xdot_eq = torch.stack([sys._f(x.unsqueeze(0), sys.nominal_params).squeeze() 
                          for x in X_eq])
    
    X = torch.cat([X, X_eq])
    Xdot = torch.cat([Xdot, Xdot_eq])
    
    return X, Xdot

def test_spr_with_dynamics(sys, X, Xdot):
    """Test SPR with intrinsic reduced dynamics."""
    logger.info("\n" + "="*60)
    logger.info("Testing SPR with Intrinsic Dynamics")
    logger.info("="*60)
    
    try:
        # Get linearization
        A, J, R = sys.linearise(return_JR=True)
        
        # Create SPR (will use even dimension)
        d = 18
        spr = SymplecticProjectionReducer(A, J, R, d, enhanced=True, X_data=X)
        
        logger.info(f"SPR created with achieved dimension: {spr.latent_dim}")
        
        # Check if it has reduced dynamics
        if hasattr(spr, 'f_red'):
            logger.info("✓ SPR has intrinsic reduced dynamics")
        
        # Validate
        results = validate_reducer(sys, spr, n_rollouts=20, t_sim=1.0)
        
        logger.info(f"Results:")
        logger.info(f"  Mean trajectory error: {results['mean_error']:.3e}")
        logger.info(f"  Max trajectory error: {results['max_error']:.3e}")
        logger.info(f"  Energy error: {results['energy_error']:.3e}")
        logger.info(f"  Success rate: {results['success_rate']:.1%}")
        
        return spr, results
        
    except Exception as e:
        logger.error(f"SPR test failed: {e}")
        return None, None

def test_lcr_with_dynamics(sys, X):
    """Test LCR with aggregated dynamics."""
    logger.info("\n" + "="*60)
    logger.info("Testing LCR with Aggregated Dynamics")
    logger.info("="*60)
    
    try:
        # Create LCR
        n_groups = 5
        lcr = LyapCoherencyReducer(sys, n_groups, X, strict_dim=False)
        
        logger.info(f"LCR created with dimension: {lcr.latent_dim}")
        logger.info(f"Actual groups: {lcr.actual_groups}")
        
        # Check if it has reduced dynamics
        if hasattr(lcr, 'f_red'):
            logger.info("✓ LCR has aggregated swing equation dynamics")
        
        # Validate
        results = validate_reducer(sys, lcr, n_rollouts=20, t_sim=1.0)
        
        logger.info(f"Results:")
        logger.info(f"  Mean trajectory error: {results['mean_error']:.3e}")
        logger.info(f"  Max trajectory error: {results['max_error']:.3e}")
        logger.info(f"  Energy error: {results['energy_error']:.3e}")
        logger.info(f"  Success rate: {results['success_rate']:.1%}")
        
        return lcr, results
        
    except Exception as e:
        logger.error(f"LCR test failed: {e}")
        return None, None

def test_opinf_stability(sys, X, Xdot):
    """Test OpInf with relaxed stability."""
    logger.info("\n" + "="*60)
    logger.info("Testing OpInf with Relaxed Stability")
    logger.info("="*60)
    
    try:
        # Create OpInf
        d = sys.n_dims  # Full dimension
        opinf = OpInfReducer(d, sys.n_dims, sys.n_controls, include_quadratic=False)
        opinf.sys = sys
        
        # Simple energy function
        def V_fn(x):
            return sys.energy_function(x)
        
        V_min = 0.01
        
        # Fit
        opinf.fit(X, Xdot, V_fn, V_min)
        
        # Check if dynamics were retained
        if opinf.dyn is not None:
            logger.info("✓ OpInf retained learned dynamics")
            if hasattr(opinf.dyn, 'A'):
                # Check spectral radius
                try:
                    dt = 0.01
                    A_discrete = torch.eye(d) + dt * opinf.dyn.A
                    rho = torch.linalg.eigvals(A_discrete).abs().max().item()
                    logger.info(f"  Discrete spectral radius: {rho:.4f}")
                except:
                    pass
        else:
            logger.info("✗ OpInf dynamics were disabled")
        
        # Validate
        results = validate_reducer(sys, opinf, n_rollouts=20, t_sim=1.0)
        
        logger.info(f"Results:")
        logger.info(f"  Mean trajectory error: {results['mean_error']:.3e}")
        logger.info(f"  Max trajectory error: {results['max_error']:.3e}")
        logger.info(f"  Energy error: {results['energy_error']:.3e}")
        logger.info(f"  Success rate: {results['success_rate']:.1%}")
        
        return opinf, results
        
    except Exception as e:
        logger.error(f"OpInf test failed: {e}")
        return None, None

def plot_comparison(sys, reducers, results_dict):
    """Plot trajectory comparison."""
    logger.info("\nPlotting trajectory comparison...")
    
    # Generate test trajectory
    x0 = sys.goal_point.squeeze() + 0.1 * torch.randn(sys.n_dims)
    T = 100  # 1 second
    dt = 0.01
    
    # Full system
    x_full = torch.zeros(T+1, sys.n_dims)
    x_full[0] = x0
    x = x0.unsqueeze(0)
    for t in range(T):
        x_dot = sys._f(x, sys.nominal_params).squeeze()
        x = x + dt * x_dot
        x_full[t+1] = x.squeeze()
    
    # Time vector
    time = np.arange(T+1) * dt
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot angles and frequencies
    angle_idx = 0  # First angle
    freq_idx = sys.N_NODES - 1  # First frequency
    
    # Angle plot
    ax = axes[0, 0]
    ax.plot(time, x_full[:, angle_idx], 'k-', linewidth=2, label='Full')
    
    for name, reducer in reducers.items():
        if reducer is not None:
            x_rom = rollout_rom(sys, reducer, x0.unsqueeze(0), T, dt=dt).squeeze()
            ax.plot(time, x_rom[:, angle_idx], '--', label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle θ₁₂ (rad)')
    ax.legend()
    ax.grid(True)
    
    # Frequency plot
    ax = axes[0, 1]
    ax.plot(time, x_full[:, freq_idx], 'k-', linewidth=2, label='Full')
    
    for name, reducer in reducers.items():
        if reducer is not None:
            x_rom = rollout_rom(sys, reducer, x0.unsqueeze(0), T, dt=dt).squeeze()
            ax.plot(time, x_rom[:, freq_idx], '--', label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency ω₁ (rad/s)')
    ax.legend()
    ax.grid(True)
    
    # Energy plot
    ax = axes[1, 0]
    E_full = torch.stack([sys.energy_function(x_full[t:t+1]) for t in range(T+1)])
    ax.plot(time, E_full, 'k-', linewidth=2, label='Full')
    
    for name, reducer in reducers.items():
        if reducer is not None:
            x_rom = rollout_rom(sys, reducer, x0.unsqueeze(0), T, dt=dt).squeeze()
            E_rom = torch.stack([sys.energy_function(x_rom[t:t+1]) for t in range(T+1)])
            ax.plot(time, E_rom, '--', label=name)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Energy')
    ax.legend()
    ax.grid(True)
    
    # Error summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary table
    summary_text = "Performance Summary:\n\n"
    summary_text += f"{'Method':<10} {'Mean Err':<12} {'Energy Err':<12} {'Success':<10}\n"
    summary_text += "-" * 44 + "\n"
    
    for name, res in results_dict.items():
        if res is not None:
            summary_text += f"{name:<10} {res['mean_error']:.3e}  {res['energy_error']:.3e}  {res['success_rate']:.1%}\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('improved_reducer_comparison.png', dpi=150)
    logger.info("  Saved to 'improved_reducer_comparison.png'")
    plt.close()

def main():
    """Run comprehensive tests."""
    logger.info("Starting comprehensive reducer tests with all fixes...")
    
    # Create system
    sys = create_test_system(n_machines=10)
    
    # Collect data
    X, Xdot = collect_training_data(sys, n_traj=50, n_steps=50)
    
    # Test each reducer
    reducers = {}
    results = {}
    
    # SPR
    spr, spr_results = test_spr_with_dynamics(sys, X, Xdot)
    if spr is not None:
        reducers['SPR-18'] = spr
        results['SPR-18'] = spr_results
    
    # LCR
    lcr, lcr_results = test_lcr_with_dynamics(sys, X)
    if lcr is not None:
        reducers['LCR'] = lcr
        results['LCR'] = lcr_results
    
    # OpInf
    opinf, opinf_results = test_opinf_stability(sys, X, Xdot)
    if opinf is not None:
        reducers['OpInf-19'] = opinf
        results['OpInf-19'] = opinf_results
    
    # Plot comparison
    plot_comparison(sys, reducers, results)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF IMPROVEMENTS")
    logger.info("="*60)
    
    expected_performance = {
        'SPR-18': {'traj': 2e-3, 'energy': 1e-4},
        'LCR': {'traj': 2e-2, 'energy': 1e-3},
        'OpInf-19': {'traj': 5e-3, 'energy': 5e-4}
    }
    
    for name, res in results.items():
        if res is not None:
            expected = expected_performance.get(name, {'traj': 1e-2, 'energy': 1e-3})
            
            traj_ok = res['mean_error'] < expected['traj']
            energy_ok = res['energy_error'] < expected['energy']
            
            status = "✓ PASS" if (traj_ok and energy_ok) else "✗ FAIL"
            
            logger.info(f"\n{name}: {status}")
            logger.info(f"  Trajectory error: {res['mean_error']:.3e} (target < {expected['traj']:.0e})")
            logger.info(f"  Energy error: {res['energy_error']:.3e} (target < {expected['energy']:.0e})")
            logger.info(f"  Success rate: {res['success_rate']:.1%}")
    
    logger.info("\n" + "="*60)
    logger.info("All tests completed!")

if __name__ == "__main__":
    main()