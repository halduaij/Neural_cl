"""
Investigating Linearization Discrepancy
======================================

Why does sys.linearise() give unstable eigenvalues while compute_A_matrix() gives stable ones?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from neural_clbf.systems import SwingEquationSystem


def create_system():
    """Create the power system"""
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
    
    # Fixed dynamics
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
    
    return sys


def compare_linearization_methods(sys):
    """Compare different linearization approaches"""
    print("="*80)
    print("LINEARIZATION METHOD COMPARISON")
    print("="*80)
    
    # Get equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    print(f"\nEquilibrium point:")
    print(f"  theta_eq norm: {theta_eq.norm():.6f}")
    print(f"  omega_eq norm: {omega_eq.norm():.6f}")
    
    # Method 1: sys.linearise() - What SPR uses
    print("\n1. Using sys.linearise() method (what SPR uses):")
    try:
        A1, J, R = sys.linearise(return_JR=True)
        print(f"   Shape: {A1.shape}")
        eigvals1 = torch.linalg.eigvals(A1)
        max_real1 = eigvals1.real.max().item()
        print(f"   Max eigenvalue real part: {max_real1:.6f}")
        
        # Also check what reference machine is being used
        print(f"   Reference machine: {sys.reference_machine if hasattr(sys, 'reference_machine') else 'Unknown'}")
        
    except Exception as e:
        print(f"   Failed: {e}")
        A1 = None
    
    # Method 2: sys.compute_A_matrix() - What I used
    print("\n2. Using sys.compute_A_matrix() method:")
    try:
        A2 = sys.compute_A_matrix(sys.nominal_params)
        A2_torch = torch.tensor(A2)
        print(f"   Shape: {A2.shape}")
        eigvals2 = np.linalg.eigvals(A2)
        max_real2 = np.real(eigvals2).max()
        print(f"   Max eigenvalue real part: {max_real2:.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
        A2_torch = None
    
    # Method 3: Manual Jacobian at equilibrium
    print("\n3. Manual Jacobian computation:")
    try:
        x_eq_grad = x_eq.clone().requires_grad_(True)
        
        def dynamics(x):
            return sys._f(x.unsqueeze(0), sys.nominal_params).squeeze()
        
        A3 = torch.autograd.functional.jacobian(dynamics, x_eq_grad)
        print(f"   Shape: {A3.shape}")
        eigvals3 = torch.linalg.eigvals(A3)
        max_real3 = eigvals3.real.max().item()
        print(f"   Max eigenvalue real part: {max_real3:.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
        A3 = None
    
    # Method 4: Check goal_point vs equilibrium
    print("\n4. Checking different equilibrium points:")
    print(f"   sys.goal_point: {sys.goal_point}")
    print(f"   Computed equilibrium norm: {x_eq.norm():.6f}")
    print(f"   Difference: {(sys.goal_point.squeeze() - x_eq).norm():.6f}")
    
    # Method 5: Original SwingEquationSystem linearization
    print("\n5. Checking original SwingEquationSystem linearization:")
    try:
        # Create a fresh system without the _f_fixed override
        sys_fresh = SwingEquationSystem(sys.nominal_params, dt=0.01)
        A5, J5, R5 = sys_fresh.linearise(return_JR=True)
        eigvals5 = torch.linalg.eigvals(A5)
        max_real5 = eigvals5.real.max().item()
        print(f"   Without _f_fixed: Max eigenvalue = {max_real5:.6f}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    return A1, A2_torch, A3


def analyze_differences(A1, A2, A3):
    """Analyze differences between linearization methods"""
    print("\n" + "="*80)
    print("DIFFERENCE ANALYSIS")
    print("="*80)
    
    if A1 is not None and A2 is not None:
        diff = (A1 - A2).abs()
        print(f"\nMethod 1 vs Method 2:")
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")
        print(f"  Frobenius norm difference: {diff.norm():.6e}")
    
    if A1 is not None and A3 is not None:
        diff = (A1 - A3).abs()
        print(f"\nMethod 1 vs Method 3:")
        print(f"  Max difference: {diff.max():.6e}")
        print(f"  Mean difference: {diff.mean():.6e}")
        print(f"  Frobenius norm difference: {diff.norm():.6e}")
    
    # Plot eigenvalue comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [
        (A1, "sys.linearise() - SPR uses this"),
        (A2, "compute_A_matrix()"),
        (A3, "Manual Jacobian")
    ]
    
    for idx, (A, name) in enumerate(methods):
        if A is not None:
            ax = axes[idx]
            eigvals = torch.linalg.eigvals(A) if torch.is_tensor(A) else np.linalg.eigvals(A)
            
            if torch.is_tensor(eigvals):
                real_parts = eigvals.real.numpy()
                imag_parts = eigvals.imag.numpy()
            else:
                real_parts = np.real(eigvals)
                imag_parts = np.imag(eigvals)
            
            ax.scatter(real_parts, imag_parts, c='red', s=100, alpha=0.7, edgecolors='black')
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Real part')
            ax.set_ylabel('Imaginary part')
            ax.set_title(f'{name}\nMax real: {real_parts.max():.3f}')
            ax.grid(True, alpha=0.3)
            
            # Highlight positive eigenvalues
            positive_mask = real_parts > 0
            if positive_mask.any():
                ax.scatter(real_parts[positive_mask], imag_parts[positive_mask], 
                          c='orange', s=150, marker='x', linewidths=3)
    
    plt.tight_layout()
    plt.savefig('linearization_comparison.png', dpi=150, bbox_inches='tight')
    
    return fig


def investigate_swing_equation_linearization(sys):
    """Deep dive into SwingEquationSystem's linearization"""
    print("\n" + "="*80)
    print("INVESTIGATING SwingEquationSystem LINEARIZATION")
    print("="*80)
    
    # Check if the system is using reference machine transformation
    print("\n1. Reference machine check:")
    print(f"   State dimension: {sys.n_dims}")
    print(f"   Number of machines: {sys.n_machines}")
    print(f"   Number of nodes: {sys.N_NODES}")
    
    # The issue might be in the coordinate system
    # SwingEquationSystem uses relative angles θ_1j = δ_1 - δ_j
    # This can affect stability
    
    print("\n2. State representation:")
    print("   State = [θ_12, θ_13, ..., θ_1n, ω_1, ω_2, ..., ω_n]")
    print("   Using machine 1 as reference")
    
    # Check the dynamics at different points
    print("\n3. Dynamics evaluation at different points:")
    
    # At goal point (all zeros)
    x_zero = torch.zeros(19)
    f_zero = sys._f(x_zero.unsqueeze(0), sys.nominal_params).squeeze()
    print(f"   At x=0: ||f|| = {f_zero.norm():.6f}")
    
    # At computed equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    f_eq = sys._f(x_eq.unsqueeze(0), sys.nominal_params).squeeze()
    print(f"   At computed eq: ||f|| = {f_eq.norm():.6f}")
    
    # The key issue: linearization point!
    print("\n4. THE KEY ISSUE:")
    print("   - sys.linearise() linearizes around x=0 (goal_point)")
    print("   - But x=0 might not be the true equilibrium!")
    print(f"   - True equilibrium has theta = {theta_eq.norm():.6f}")
    
    return x_eq


def main():
    """Main investigation"""
    # Create system
    sys = create_system()
    
    # Compare methods
    A1, A2, A3 = compare_linearization_methods(sys)
    
    # Analyze differences
    fig = analyze_differences(A1, A2, A3)
    
    # Investigate SwingEquationSystem specifics
    x_eq = investigate_swing_equation_linearization(sys)
    
    plt.show()
    
    return A1, A2, A3


if __name__ == "__main__":
    A1, A2, A3 = main()