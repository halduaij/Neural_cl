"""
Validation with d=19 (full dimension) to ensure 100% information retention
"""
import torch
import numpy as np
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer


def create_stable_system():
    """Create a stable system with reasonable parameters."""
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
    
    params = dict(M=M, D=D, P=P, K=K)
    
    sys = SwingEquationSystem(params, dt=0.001)
    sys.delta_star = delta_star
    
    return sys, params


def collect_test_data(sys, params, n_samples=100):
    """Collect test data around equilibrium."""
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    # Small perturbations
    X = x_eq.unsqueeze(0).repeat(n_samples, 1)
    X += 0.01 * torch.randn_like(X)
    
    # Compute derivatives
    Xdot = []
    for i in range(n_samples):
        f = sys._f(X[i:i+1], params)
        Xdot.append(f.squeeze())
    
    return {
        'X': X,
        'dXdt': torch.stack(Xdot)
    }


def test_full_dimension_preservation():
    """Test all reducers with d=19 (full dimension) for perfect reconstruction."""
    print("="*80)
    print("FULL DIMENSION TEST (d=19) - EXPECTING NEAR-ZERO ERRORS")
    print("="*80)
    
    # Create system
    sys, params = create_stable_system()
    data = collect_test_data(sys, params)
    
    # Energy function
    V_fn = sys.energy_function if hasattr(sys, 'energy_function') else lambda x: 0.5 * (x ** 2).sum(dim=1)
    V_min = V_fn(data['X']).min().item()
    
    print(f"\nSystem: 10-generator network")
    print(f"Full dimension: 19")
    print(f"Test data: {data['X'].shape[0]} samples")
    
    results = {}
    
    # Test 1: Symplectic Projection (d=18 since must be even)
    print("\n1. Testing Symplectic Projection (d=18, closest even dimension):")
    try:
        A, J, R = sys.linearise(return_JR=True)
        spr = SymplecticProjectionReducer(A, J, R, 18)  # Must be even
        spr.full_dim = 19
        
        # Test reconstruction
        X_test = data['X'][:10]  # Test on 10 samples
        Z = spr.forward(X_test)
        X_recon = spr.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1).mean().item()
        
        # Test projection matrix properties
        P = spr.T  # Projection matrix
        PT_P = P.T @ P
        I_expected = torch.eye(18, device=P.device)
        orthogonality_error = (PT_P - I_expected).norm().item()
        
        results['SPR-18'] = {
            'recon_error': recon_error,
            'orthogonality_error': orthogonality_error,
            'gamma': spr.gamma
        }
        
        print(f"   Reconstruction error: {recon_error:.6e}")
        print(f"   Orthogonality error: {orthogonality_error:.6e}")
        print(f"   Gamma: {spr.gamma}")
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['SPR-18'] = {'error': str(e)}
    
    # Test 2: OpInf with d=19
    print("\n2. Testing OpInf (d=19, full dimension):")
    try:
        opinf = OpInfReducer(19, 19, sys.n_controls)
        opinf.sys = sys
        opinf.fit(data['X'], data['dXdt'], V_fn, V_min)
        opinf.full_dim = 19
        
        # Test reconstruction
        X_test = data['X'][:10]
        Z = opinf.forward(X_test)
        X_recon = opinf.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1).mean().item()
        
        # Check projection matrix
        P = opinf.proj
        if P.shape[1] == 19:  # Square matrix
            # Should be close to identity (after centering)
            X_centered = X_test - opinf.μ
            Z_manual = X_centered @ P
            proj_error = (Z_manual - Z).norm().item()
        else:
            proj_error = float('nan')
        
        results['OpInf-19'] = {
            'recon_error': recon_error,
            'projection_error': proj_error,
            'gamma': opinf.gamma
        }
        
        print(f"   Reconstruction error: {recon_error:.6e}")
        print(f"   Projection error: {proj_error:.6e}")
        print(f"   Gamma: {opinf.gamma}")
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['OpInf-19'] = {'error': str(e)}
    
    # Test 3: Lyapunov Coherency with maximum groups
    print("\n3. Testing Lyapunov Coherency (k=9 groups → d=18):")
    try:
        # Maximum is 9 groups (not 10, since reference machine is fixed)
        lcr = LyapCoherencyReducer(sys, 9, data['X'])
        lcr.full_dim = 19
        lcr.gamma = lcr.compute_gamma(V_min)
        
        # Test reconstruction
        X_test = data['X'][:10]
        Z = lcr.forward(X_test)
        X_recon = lcr.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1).mean().item()
        
        # Check if P @ Pi ≈ I
        P_Pi = lcr.P @ lcr.Pi
        I_expected = torch.eye(P_Pi.shape[0], P_Pi.shape[1], device=P_Pi.device)
        reconstruction_property = (P_Pi - I_expected).norm().item()
        
        results['LCR-18'] = {
            'recon_error': recon_error,
            'reconstruction_property': reconstruction_property,
            'gamma': lcr.gamma
        }
        
        print(f"   Reconstruction error: {recon_error:.6e}")
        print(f"   P @ Pi - I error: {reconstruction_property:.6e}")
        print(f"   Gamma: {lcr.gamma}")
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['LCR-18'] = {'error': str(e)}
    
    # Test 4: Identity reducer (baseline)
    print("\n4. Testing Identity Reducer (baseline):")
    try:
        class IdentityReducer:
            def __init__(self):
                self.latent_dim = 19
                self.full_dim = 19
                self.gamma = 0.0
            
            def forward(self, x):
                return x
            
            def inverse(self, z):
                return z
            
            def jacobian(self, x):
                B = x.shape[0] if x.dim() > 1 else 1
                return torch.eye(19, device=x.device).unsqueeze(0).expand(B, -1, -1)
        
        identity = IdentityReducer()
        
        X_test = data['X'][:10]
        Z = identity.forward(X_test)
        X_recon = identity.inverse(Z)
        
        recon_error = (X_recon - X_test).norm(dim=1).mean().item()
        
        results['Identity'] = {
            'recon_error': recon_error,
            'gamma': identity.gamma
        }
        
        print(f"   Reconstruction error: {recon_error:.6e} (should be exactly 0)")
        
    except Exception as e:
        print(f"   Failed: {e}")
        results['Identity'] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - FULL DIMENSION TEST")
    print("="*80)
    print("\nExpected behavior:")
    print("- Reconstruction errors should be < 1e-10 (numerical precision)")
    print("- Orthogonality/projection errors should be < 1e-10")
    print("- Gamma should be small (< 1.0) for full dimension")
    
    print(f"\n{'Method':<15} {'Recon Error':<15} {'Other Error':<15} {'Gamma':<10} {'Status':<10}")
    print("-"*65)
    
    for method, res in results.items():
        if 'error' in res:
            print(f"{method:<15} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'FAILED':<10}")
        else:
            recon = f"{res['recon_error']:.2e}"
            other = f"{res.get('orthogonality_error', res.get('projection_error', res.get('reconstruction_property', 0))):.2e}"
            gamma = f"{res['gamma']:.3f}"
            status = "PASS" if res['recon_error'] < 1e-6 else "FAIL"
            print(f"{method:<15} {recon:<15} {other:<15} {gamma:<10} {status:<10}")
    
    # Additional dynamics test
    print("\n" + "="*80)
    print("DYNAMICS PRESERVATION TEST")
    print("="*80)
    
    for method_name, reducer in [
        ('OpInf-19', opinf if 'opinf' in locals() else None),
        ('SPR-18', spr if 'spr' in locals() else None),
        ('LCR-18', lcr if 'lcr' in locals() else None)
    ]:
        if reducer is None:
            continue
            
        print(f"\n{method_name}:")
        try:
            # Test if dynamics are preserved
            x0 = data['X'][0:1]
            z0 = reducer.forward(x0)
            
            # Full dynamics
            f_full = sys._f(x0, params).squeeze()
            
            # Reduced dynamics (project → dynamics → reconstruct)
            J = reducer.jacobian(x0)
            f_reduced = J @ f_full.unsqueeze(-1)
            f_reconstructed = reducer.inverse(f_reduced.squeeze().unsqueeze(0)).squeeze()
            
            dynamics_error = (f_reconstructed - f_full).norm().item()
            print(f"   Dynamics preservation error: {dynamics_error:.6e}")
            
        except Exception as e:
            print(f"   Dynamics test failed: {e}")
    
    return results


if __name__ == "__main__":
    results = test_full_dimension_preservation()