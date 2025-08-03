"""
Test the fixed reducer implementations
======================================
"""
import torch
import numpy as np
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer


def create_test_system():
    """Create test system."""
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 5.0
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
    sys = SwingEquationSystem(params, dt=0.001)
    sys.delta_star = delta_star
    
    return sys, params


def test_lcr_fix():
    """Test if LCR fix works."""
    print("\n" + "="*70)
    print("TESTING LCR FIX")
    print("="*70)
    
    sys, params = create_test_system()
    
    # Create training data
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    X_train = x_eq.unsqueeze(0).repeat(100, 1)
    X_train += 0.01 * torch.randn_like(X_train)
    
    # Test LCR with 9 groups (should work now)
    lcr = LyapCoherencyReducer(sys, 9, X_train)
    
    # Check projection properties
    P_Pi = lcr.P @ lcr.Pi
    identity_error = (P_Pi @ lcr.P - lcr.P).norm()
    
    print(f"LCR with 9 groups:")
    print(f"  P shape: {lcr.P.shape}")
    print(f"  Latent dim: {lcr.latent_dim}")
    print(f"  ||P @ Pi @ P - P||: {identity_error:.6e} (should be ~0)")
    
    # Check for empty groups
    for g in range(lcr.n_groups):
        count = (lcr.labels == g).sum().item()
        print(f"  Group {g}: {count} machines")
    
    # Test reconstruction
    x_test = X_train[:5]
    z = lcr.forward(x_test)
    x_recon = lcr.inverse(z)
    errors = (x_recon - x_test).norm(dim=1)
    
    print(f"\nReconstruction test:")
    print(f"  Mean error: {errors.mean():.6e}")
    print(f"  Max error: {errors.max():.6e}")
    
    return identity_error < 1e-6


def test_opinf_fix():
    """Test if OpInf fix works."""
    print("\n" + "="*70)
    print("TESTING OPINF FIX")
    print("="*70)
    
    sys, params = create_test_system()
    
    # Create training data
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    X_train = x_eq.unsqueeze(0).repeat(100, 1)
    X_train += 0.01 * torch.randn_like(X_train)
    
    Xdot_train = []
    for i in range(100):
        f = sys._f(X_train[i:i+1], params)
        Xdot_train.append(f.squeeze())
    Xdot_train = torch.stack(Xdot_train)
    
    V_fn = sys.energy_function
    V_min = V_fn(X_train).min().item()
    
    # Test OpInf with d=19
    opinf = OpInfReducer(19, 19, sys.n_controls)
    opinf.sys = sys
    opinf.fit(X_train, Xdot_train, V_fn, V_min)
    
    # Check dynamics stability
    A_eigs = torch.linalg.eigvals(opinf.dyn.A).real
    max_eig = A_eigs.max().item()
    
    print(f"OpInf with d=19:")
    print(f"  Max eigenvalue: {max_eig:.6f} (should be < 0)")
    print(f"  Regularization: {opinf.dyn.reg.item()}")
    print(f"  Gamma: {opinf.gamma:.6f}")
    
    # Test short trajectory
    x0 = X_train[0]
    dt = 0.001
    trajectory_stable = True
    
    x = x0
    for step in range(3000):  # 1 second
        z = opinf.forward(x.unsqueeze(0))
        z_dot = opinf.dyn.forward(z, torch.zeros(1, sys.n_controls)).squeeze()
        z_next = z.squeeze() + dt * z_dot
        x = opinf.inverse(z_next)
        
        if not torch.isfinite(x).all() or x.norm() > 1e3:
            trajectory_stable = False
            print(f"  Trajectory diverged at step {step}")
            break
    
    if trajectory_stable:
        print(f"  Trajectory stable for 100 steps")
    
    return max_eig < 0 and trajectory_stable


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING FIXED REDUCER IMPLEMENTATIONS")
    print("="*70)
    
    lcr_pass = test_lcr_fix()
    opinf_pass = test_opinf_fix()
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"LCR Fix: {'PASS' if lcr_pass else 'FAIL'}")
    print(f"OpInf Fix: {'PASS' if opinf_pass else 'FAIL'}")
    
    if lcr_pass and opinf_pass:
        print("\nAll fixes working! You can now:")
        print("1. Replace neural_clbf/dimension_reduction/lyap_coherency.py with the fixed version")
        print("2. Replace neural_clbf/dimension_reduction/opinf.py with the fixed version")
        print("3. Re-run your trajectory preservation tests")
    else:
        print("\nSome fixes still need work.")


if __name__ == "__main__":
    main()