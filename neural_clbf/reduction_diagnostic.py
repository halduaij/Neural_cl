"""
Comprehensive Reduction Diagnostic
==================================

Diagnoses why ALL reduction methods (SPR, LCR, OpInf) have poor accuracy.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer


def create_system_and_data():
    """Create system and data"""
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
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
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = delta_star
    
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
        
        for i in range(1, 10):
            f[:, i-1, 0] = omega[:, 0] - omega[:, i]
        
        for i in range(10):
            omega_dot = P[i] / M[i] - (D[i] / M[i]) * omega[:, i]
            for j in range(10):
                if i != j:
                    omega_dot -= (K[i, j] / M[i]) * torch.sin(delta[:, i] - delta[:, j])
            f[:, 9+i, 0] = omega_dot
        
        return f
    
    sys._f = _f_fixed
    
    # Training data
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    X_train = []
    Xdot_train = []
    
    for i in range(1000):
        x = x_eq + 0.01 * torch.randn_like(x_eq)
        X_train.append(x)
        f = sys._f(x.unsqueeze(0), sys.nominal_params).squeeze()
        Xdot_train.append(f)
    
    X_train = torch.stack(X_train)
    Xdot_train = torch.stack(Xdot_train)
    
    return sys, X_train, Xdot_train, x_eq


def analyze_svd_and_energy(X_train, sys):
    """Analyze SVD spectrum and energy content"""
    print("\n1. SVD Analysis of Training Data:")
    print("-" * 50)
    
    # Center data
    X_mean = X_train.mean(0)
    X_centered = X_train - X_mean
    
    # SVD
    U, S, Vt = torch.linalg.svd(X_centered.T, full_matrices=False)
    
    # Energy captured
    energy = S**2
    total_energy = energy.sum()
    cumsum_energy = torch.cumsum(energy, dim=0) / total_energy
    
    print(f"Singular values (top 10):")
    for i in range(min(10, len(S))):
        print(f"  σ_{i+1} = {S[i]:.4f}, Energy captured: {cumsum_energy[i]:.2%}")
    
    # Find how many modes needed for 99% energy
    n_99 = (cumsum_energy < 0.99).sum() + 1
    n_999 = (cumsum_energy < 0.999).sum() + 1
    
    print(f"\nModes needed:")
    print(f"  For 99% energy: {n_99}")
    print(f"  For 99.9% energy: {n_999}")
    print(f"  Currently using: 6")
    
    return U, S, Vt, cumsum_energy


def analyze_projection_quality(sys, reducers, X_test):
    """Analyze projection and reconstruction quality"""
    print("\n2. Projection Quality Analysis:")
    print("-" * 50)
    
    results = {}
    
    for name, reducer in reducers.items():
        # Forward-inverse error
        Z = reducer.forward(X_test)
        X_recon = reducer.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1)
        
        # Component-wise errors
        angle_error = (X_recon[:, :9] - X_test[:, :9]).norm(dim=1)
        freq_error = (X_recon[:, 9:] - X_test[:, 9:]).norm(dim=1)
        
        # Energy preservation
        E_true = sys.energy_function(X_test)
        E_recon = sys.energy_function(X_recon)
        energy_error = (E_recon - E_true).abs() / (E_true.abs() + 1e-10)
        
        # Dynamics preservation
        f_true = sys._f(X_test, sys.nominal_params).squeeze(-1)
        f_recon = sys._f(X_recon, sys.nominal_params).squeeze(-1)
        dynamics_error = (f_recon - f_true).norm(dim=1)
        
        results[name] = {
            'recon_error': recon_error.mean().item(),
            'angle_error': angle_error.mean().item(),
            'freq_error': freq_error.mean().item(),
            'energy_error': energy_error.mean().item(),
            'dynamics_error': dynamics_error.mean().item(),
            'latent_dim': reducer.latent_dim
        }
        
        print(f"\n{name}:")
        print(f"  Reconstruction error: {recon_error.mean():.4f}")
        print(f"  Angle error: {angle_error.mean():.4f}")
        print(f"  Frequency error: {freq_error.mean():.4f}")
        print(f"  Energy error: {energy_error.mean():.2%}")
        print(f"  Dynamics error: {dynamics_error.mean():.4f}")
    
    return results


def analyze_dynamics_projection(sys, reducer, name):
    """Analyze how dynamics are projected"""
    print(f"\n3. Dynamics Projection Analysis - {name}:")
    print("-" * 50)
    
    # Get equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    # Small perturbation
    x_pert = x_eq + 0.01 * torch.randn_like(x_eq)
    
    # Full dynamics
    f_full = sys._f(x_pert.unsqueeze(0), sys.nominal_params).squeeze()
    
    # Through reducer
    z = reducer.forward(x_pert.unsqueeze(0))
    x_recon = reducer.inverse(z).squeeze()
    f_recon_direct = sys._f(x_recon.unsqueeze(0), sys.nominal_params).squeeze()
    
    # Projected dynamics
    J = reducer.jacobian(x_pert.unsqueeze(0))
    f_full_batch = sys._f(x_pert.unsqueeze(0), sys.nominal_params)
    z_dot = torch.bmm(J, f_full_batch).squeeze()
    
    # Reconstruct projected dynamics
    # This is what the reduced model actually uses
    f_projected = reducer.inverse(z_dot.unsqueeze(0)).squeeze() - x_recon
    
    print(f"  ||f_full||: {f_full.norm():.4f}")
    print(f"  ||f_recon_direct||: {f_recon_direct.norm():.4f}")
    print(f"  ||f_projected||: {f_projected.norm():.4f}")
    print(f"  Direct error: {(f_recon_direct - f_full).norm():.4f}")
    print(f"  Projection error: {(f_projected - f_full).norm():.4f}")
    
    # Check if dynamics are in the range of the projection
    if hasattr(reducer, 'P'):  # For LCR
        P = reducer.P
        Pi = reducer.Pi
        f_proj_check = f_full @ P @ Pi
        print(f"  Dynamics in range check: {(f_proj_check - f_full).norm():.4f}")
    
    return f_full, f_recon_direct, f_projected


def test_different_dimensions(sys, X_train, Xdot_train, dimensions=[4, 6, 8, 10, 12, 14, 16]):
    """Test how accuracy changes with dimension"""
    print("\n4. Dimension Sensitivity Analysis:")
    print("-" * 50)
    
    V_fn = sys.energy_function
    V_min = V_fn(X_train).min().item()
    
    # Test data
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    X_test = []
    for _ in range(100):
        x = x_eq + 0.01 * torch.randn_like(x_eq)
        X_test.append(x)
    X_test = torch.stack(X_test)
    
    results = {
        'SPR': {},
        'LCR': {},
        'OpInf': {}
    }
    
    for d in dimensions:
        print(f"\nTesting dimension {d}:")
        
        # SPR (only even dimensions)
        if d % 2 == 0:
            try:
                A, J, R = sys.linearise(return_JR=True)
                spr = SymplecticProjectionReducer(A, J, R, latent_dim=d)
                Z = spr.forward(X_test)
                X_recon = spr.inverse(Z)
                error = (X_recon - X_test).norm(dim=1).mean().item()
                results['SPR'][d] = error
                print(f"  SPR: {error:.4f}")
            except:
                print(f"  SPR: Failed")
        
        # LCR (groups = d/2)
        if d % 2 == 0:
            try:
                lcr = LyapCoherencyReducer(sys, n_groups=d//2, snaps=X_train)
                Z = lcr.forward(X_test)
                X_recon = lcr.inverse(Z)
                error = (X_recon - X_test).norm(dim=1).mean().item()
                results['LCR'][d] = error
                print(f"  LCR: {error:.4f}")
            except:
                print(f"  LCR: Failed")
        
        # OpInf
        try:
            opinf = OpInfReducer(latent_dim=d, n_full=sys.n_dims, n_controls=1)
            opinf.sys = sys
            opinf.fit(X_train[:500], Xdot_train[:500], V_fn, V_min)  # Less data for speed
            Z = opinf.forward(X_test)
            X_recon = opinf.inverse(Z)
            error = (X_recon - X_test).norm(dim=1).mean().item()
            results['OpInf'][d] = error
            print(f"  OpInf: {error:.4f}")
        except:
            print(f"  OpInf: Failed")
    
    return results


def visualize_modes(sys, reducers, X_train):
    """Visualize what each reducer captures"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Get equilibrium
    x_eq = X_train.mean(0)
    
    for idx, (name, reducer) in enumerate(reducers.items()):
        row = idx
        
        # 1. Projection matrix structure (if available)
        ax = axes[row, 0]
        if hasattr(reducer, 'P'):
            P = reducer.P.detach().numpy()
            im = ax.imshow(P, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
            ax.set_title(f'{name} - Projection Matrix')
            ax.set_xlabel('Latent dim')
            ax.set_ylabel('Full state')
            plt.colorbar(im, ax=ax)
        elif hasattr(reducer, 'proj'):
            P = reducer.proj.detach().numpy()
            im = ax.imshow(P, aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
            ax.set_title(f'{name} - Projection Matrix')
            ax.set_xlabel('Latent dim')
            ax.set_ylabel('Full state')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No projection matrix', ha='center', va='center')
            ax.set_title(f'{name}')
        
        # 2. Mode shapes
        ax = axes[row, 1]
        # Create unit vectors in latent space and see their effect
        Z_unit = torch.eye(reducer.latent_dim)
        X_modes = reducer.inverse(Z_unit)
        
        # Plot first 3 modes
        for i in range(min(3, reducer.latent_dim)):
            mode = X_modes[i] - x_eq
            ax.plot(mode[:9], label=f'Mode {i+1} angles', alpha=0.7)
        ax.set_title(f'{name} - Mode Shapes (Angles)')
        ax.set_xlabel('Angle index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Frequency content
        ax = axes[row, 2]
        for i in range(min(3, reducer.latent_dim)):
            mode = X_modes[i] - x_eq
            ax.plot(mode[9:], label=f'Mode {i+1} freq', alpha=0.7)
        ax.set_title(f'{name} - Mode Shapes (Frequencies)')
        ax.set_xlabel('Machine index')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Run comprehensive diagnostic"""
    print("="*80)
    print("REDUCTION DIAGNOSTIC - Why All Methods Have Poor Accuracy")
    print("="*80)
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    sys, X_train, Xdot_train, x_eq = create_system_and_data()
    
    # Create reducers
    V_fn = sys.energy_function
    V_min = V_fn(X_train).min().item()
    
    # Standard reducers at d=6
    reducers = {}
    
    # SPR
    A, J, R = sys.linearise(return_JR=True)
    spr = SymplecticProjectionReducer(A, J, R, latent_dim=6)
    reducers['SPR-6'] = spr
    
    # LCR
    lcr = LyapCoherencyReducer(sys, n_groups=3, snaps=X_train)
    reducers['LCR-6'] = lcr
    
    # OpInf
    opinf = OpInfReducer(latent_dim=6, n_full=sys.n_dims, n_controls=1)
    opinf.sys = sys
    opinf.fit(X_train, Xdot_train, V_fn, V_min)
    reducers['OpInf-6'] = opinf
    
    # 1. SVD Analysis
    U, S, Vt, cumsum_energy = analyze_svd_and_energy(X_train, sys)
    
    # 2. Projection Quality
    X_test = X_train[:100]
    proj_results = analyze_projection_quality(sys, reducers, X_test)
    
    # 3. Dynamics Projection
    for name, reducer in reducers.items():
        analyze_dynamics_projection(sys, reducer, name)
    
    # 4. Dimension sensitivity
    dim_results = test_different_dimensions(sys, X_train, Xdot_train)
    
    # 5. Visualize modes
    mode_fig = visualize_modes(sys, reducers, X_train)
    
    # 6. Plot dimension sensitivity
    plt.figure(figsize=(10, 6))
    
    for method, results in dim_results.items():
        if results:
            dims = sorted(results.keys())
            errors = [results[d] for d in dims]
            plt.semilogy(dims, errors, 'o-', label=method, linewidth=2, markersize=8)
    
    plt.axvline(x=6, color='k', linestyle='--', alpha=0.5, label='Current dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Reconstruction Error (log scale)')
    plt.title('Reconstruction Error vs Dimension')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figures
    mode_fig.savefig('mode_visualization.png', dpi=150, bbox_inches='tight')
    plt.savefig('dimension_sensitivity.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    print("\n1. DIMENSION ISSUE:")
    print(f"   - Using 6 dimensions captures only ~{cumsum_energy[5]:.1%} of energy")
    print(f"   - Need at least {(cumsum_energy < 0.99).sum() + 1} dimensions for 99% energy")
    
    print("\n2. PROJECTION ISSUES:")
    print("   - All methods have significant reconstruction error")
    print("   - Dynamics are not well-preserved in projection")
    
    print("\n3. RECOMMENDATIONS:")
    print("   a) Increase dimension to at least 10-12")
    print("   b) Use physics-based mode selection")
    print("   c) Consider nonlinear reduction methods")
    print("   d) Design reducers specifically for oscillatory systems")
    
    plt.show()
    
    return dim_results


if __name__ == "__main__":
    results = main()