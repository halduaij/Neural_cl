"""
Example Usage: Improved Power System Dimension Reduction
========================================================

Demonstrates how to use the improved reducers and validation framework
for power system transient stability analysis.
"""

import torch
import numpy as np
from neural_clbf.systems import SwingEquationSystem

# Import improved reducers - adjust paths based on where you saved the files
# If you saved them in neural_clbf/dimension_reduction/:
from neural_clbf.dimension_reduction.improved_spr import ImprovedSymplecticProjectionReducer
from neural_clbf.dimension_reduction.improved_lcr import ImprovedLyapCoherencyReducer
from neural_clbf.dimension_reduction.improved_opinf import ImprovedOpInfReducer

# If transient_stability_validation.py is in the same folder as this script:


def create_test_power_system():
    """Create a test 10-machine power system."""
    # System parameters (from your working example)
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    # Higher damping for stability
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 3.0
    
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
    
    params = dict(M=M, D=D, P=P, K=K)
    
    # Create system
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = delta_star
    
    # Add corrected dynamics (as in your file)
    original_f = sys._f
    def _f_fixed(x, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, 19, 1))
        
        theta = x[:, :9]
        omega = x[:, 9:]
        
        delta_eq = delta_star
        theta_eq = delta_eq[0] - delta_eq[1:]
        
        delta = torch.zeros(batch_size, 10)
        delta[:, 0] = delta_eq[0]
        delta[:, 1:] = delta_eq[1:] + (theta_eq - theta)
        
        # Angle derivatives
        for i in range(1, 10):
            f[:, i-1, 0] = omega[:, 0] - omega[:, i]
        
        # Frequency derivatives
        for i in range(10):
            omega_dot = P[i] / M[i] - (D[i] / M[i]) * omega[:, i]
            for j in range(10):
                if i != j:
                    omega_dot -= (K[i, j] / M[i]) * torch.sin(delta[:, i] - delta[:, j])
            f[:, 9+i, 0] = omega_dot
        
        return f
    
    sys._f = _f_fixed
    
    return sys


def collect_training_data(sys, n_samples=2000, perturbation_level=0.02):
    """Collect diverse training data for reducer fitting."""
    print("Collecting training data...")
    
    # Get equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    # Generate diverse initial conditions
    X = []
    Xdot = []
    
    # 1. Small perturbations around equilibrium
    for i in range(n_samples // 2):
        x = x_eq + perturbation_level * torch.randn_like(x_eq)
        X.append(x)
        
        # Compute derivative
        f = sys._f(x.unsqueeze(0), sys.nominal_params)
        Xdot.append(f.squeeze())
    
    # 2. Larger perturbations for robustness
    for i in range(n_samples // 4):
        x = x_eq + 2 * perturbation_level * torch.randn_like(x_eq)
        X.append(x)
        
        f = sys._f(x.unsqueeze(0), sys.nominal_params)
        Xdot.append(f.squeeze())
    
    # 3. Trajectory data
    dt = 0.01
    for i in range(n_samples // 4):
        # Random initial condition
        x = x_eq + perturbation_level * torch.randn_like(x_eq)
        
        # Short trajectory
        for _ in range(10):
            X.append(x)
            f = sys._f(x.unsqueeze(0), sys.nominal_params)
            Xdot.append(f.squeeze())
            x = x + dt * f.squeeze()
    
    X = torch.stack(X)
    Xdot = torch.stack(Xdot)
    
    print(f"  Collected {X.shape[0]} snapshots")
    print(f"  State range: [{X.min():.3f}, {X.max():.3f}]")
    
    return X, Xdot


def create_improved_reducers(sys, X, Xdot, target_dim=6):
    """Create all three improved reducers."""
    print("\nCreating improved reducers...")
    
    reducers = {}
    
    # Energy function for OpInf
    def energy_fn(x):
        return sys.energy_function(x)
    
    V_min = energy_fn(X).min().item()
    
    # 1. Improved Symplectic Projection
    try:
        print("\n1. Creating Improved Symplectic Projection Reducer...")
        spr = ImprovedSymplecticProjectionReducer(
            sys=sys,
            latent_dim=target_dim,
            enhanced=True,
            n_linearization_points=5,
            nonlinear_weight=0.15
        )
        spr.full_dim = sys.n_dims
        reducers['SPR-Enhanced'] = spr
        print(f"   Created SPR with d={spr.latent_dim}")
    except Exception as e:
        print(f"   SPR failed: {e}")
    
    # 2. Improved Lyapunov Coherency
    try:
        print("\n2. Creating Improved Lyapunov Coherency Reducer...")
        lcr = ImprovedLyapCoherencyReducer(
            sys=sys,
            n_groups=target_dim // 2,  # 2 states per group
            snaps=X,
            λ=0.7,
            adaptive=True,
            fuzzy_sigma=0.5,
            energy_weight=0.3
        )
        lcr.full_dim = sys.n_dims
        lcr.gamma = lcr.compute_gamma(V_min)
        reducers['LCR-Enhanced'] = lcr
        print(f"   Created LCR with d={lcr.latent_dim}, γ={lcr.gamma:.3f}")
    except Exception as e:
        print(f"   LCR failed: {e}")
    
    # 3. Improved OpInf
    try:
        print("\n3. Creating Improved Operator Inference Reducer...")
        opinf = ImprovedOpInfReducer(
            latent_dim=target_dim,
            n_full=sys.n_dims,
            n_controls=sys.n_controls,
            physics_informed=True,
            stability_enforced=True,
            multi_fidelity=True,
            energy_weight=0.3,
            freq_weight=0.3,
            stability_margin=0.5
        )
        opinf.sys = sys
        opinf.fit(X, Xdot, energy_fn, V_min)
        reducers['OpInf-Enhanced'] = opinf
        print(f"   Created OpInf with d={opinf.latent_dim}, γ={opinf.gamma:.3f}")
    except Exception as e:
        print(f"   OpInf failed: {e}")
    
    return reducers


def quick_performance_test(sys, reducers, X_test):
    """Quick performance test of reducers."""
    print("\nQuick Performance Test:")
    print("-" * 60)
    
    for name, reducer in reducers.items():
        # Test reconstruction
        Z = reducer.forward(X_test)
        X_recon = reducer.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1).mean()
        
        # Test dynamics preservation
        f_full = sys._f(X_test, sys.nominal_params).squeeze(-1)
        f_recon = sys._f(X_recon, sys.nominal_params).squeeze(-1)
        
        dynamics_error = (f_recon - f_full).norm(dim=1).mean()
        
        # Test energy preservation
        E_full = sys.energy_function(X_test)
        E_recon = sys.energy_function(X_recon)
        energy_error = ((E_recon - E_full) / E_full).abs().mean()
        
        print(f"\n{name}:")
        print(f"  Reconstruction error: {recon_error:.4f}")
        print(f"  Dynamics error: {dynamics_error:.4f}")
        print(f"  Energy error: {energy_error:.2%}")


def main():
    """Main example workflow."""
    print("="*80)
    print("IMPROVED POWER SYSTEM DIMENSION REDUCTION - EXAMPLE")
    print("="*80)
    
    # 1. Create power system
    sys = create_test_power_system()
    print(f"\nCreated {sys.n_machines}-machine power system")
    print(f"State dimension: {sys.n_dims}")
    
    # 2. Collect training data
    X, Xdot = collect_training_data(sys, n_samples=2000)
    
    # 3. Create improved reducers
    reducers = create_improved_reducers(sys, X, Xdot, target_dim=6)
    
    # 4. Quick performance test
    X_test = X[:100]  # Test on subset
    quick_performance_test(sys, reducers, X_test)
    
    # 5. Comprehensive validation
    print("\n" + "="*80)
    print("COMPREHENSIVE TRANSIENT STABILITY VALIDATION")
    print("="*80)
    
    # Run validation with fewer scenarios for demo
    validator = TransientStabilityValidator(sys)
    
    # Custom thresholds for this system (can be adjusted)
    custom_thresholds = {
        'cct_error': 0.05,  # 5% CCT error
        'stability_accuracy': 0.95,  # 95% accuracy (relaxed for demo)
        'angle_rmse': 2.0,  # 2 degrees (relaxed)
        'frequency_rmse': 0.2,  # 0.2 Hz (relaxed)
        'energy_error': 0.1,  # 10% (relaxed)
        'min_perturbation': 0.3,  # 30% (relaxed)
        'max_computation_time': 0.2,  # 200ms (relaxed)
    }
    validator.thresholds.update(custom_thresholds)
    
    # Run validation (with fewer scenarios for speed)
    results = run_comprehensive_validation(sys, reducers, n_scenarios=5)
    
    # 6. Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_reducer = min(results.items(), key=lambda x: x[1].score)
    print(f"\nBest performer: {best_reducer[0]} with score {best_reducer[1].score:.2f}")
    
    print("\nKey findings:")
    for name, metrics in results.items():
        print(f"\n{name}:")
        if metrics.passed:
            print("  ✓ Suitable for transient stability analysis")
        else:
            print("  ✗ Needs improvement:")
            if metrics.cct_accuracy > validator.thresholds['cct_error']:
                print(f"    - CCT error too high ({metrics.cct_accuracy:.1%})")
            if metrics.energy_error > validator.thresholds['energy_error']:
                print(f"    - Energy conservation poor ({metrics.energy_error:.1%})")
            if metrics.perturbation_preservation < validator.thresholds['min_perturbation']:
                print(f"    - Poor perturbation sensitivity ({metrics.perturbation_preservation:.1%})")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run example
    main()