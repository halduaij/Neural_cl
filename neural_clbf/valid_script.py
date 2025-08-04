"""
Verify that all fixes have been applied correctly
"""
import torch
import numpy as np
from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer

def check_spr_fixes():
    """Check if SPR fixes were applied"""
    print("\n=== CHECKING SPR FIXES ===")
    
    # Check if the _build_standard_basis method has the fix
    import inspect
    source = inspect.getsource(SymplecticProjectionReducer._build_standard_basis)
    has_pair_fix = "basis.extend([v, Jv])" in source
    print(f"✓ Symplectic pair fix: {'FOUND' if has_pair_fix else 'NOT FOUND'}")
    
    # Check if _build_reduced_dynamics has stabilization
    source = inspect.getsource(SymplecticProjectionReducer._build_reduced_dynamics)
    has_stabilization = "max_real > -0.02" in source or "shift * torch.eye" in source
    print(f"✓ A_r stabilization fix: {'FOUND' if has_stabilization else 'NOT FOUND'}")
    
    return has_pair_fix and has_stabilization

def check_lcr_fixes():
    """Check if LCR fixes were applied"""
    print("\n=== CHECKING LCR FIXES ===")
    
    import inspect
    source = inspect.getsource(LyapCoherencyReducer._build_projection_matrix)
    
    # Check for inertia weighting
    has_inertia_weight = "sqrt(self.sys.M[i] / M_total)" in source or "sqrt(M_i / M_group)" in source
    print(f"✓ Inertia weighting fix: {'FOUND' if has_inertia_weight else 'NOT FOUND'}")
    
    # Check for sign fix
    source_dynamics = inspect.getsource(LyapCoherencyReducer._build_reduced_dynamics)
    has_sign_fix = "-self.sys.P_mechanical" in source_dynamics
    print(f"✓ P_eq sign fix: {'FOUND' if has_sign_fix else 'NOT FOUND'}")
    
    # Check for omega slice fix
    has_omega_fix = "z[:, 1::2]" in source_dynamics or "omega = z[:, 1::2]" in source_dynamics
    print(f"✓ Omega slice fix: {'FOUND' if has_omega_fix else 'NOT FOUND'}")
    
    return has_inertia_weight and has_sign_fix and has_omega_fix

def check_opinf_fixes():
    """Check if OpInf fixes were applied"""
    print("\n=== CHECKING OPINF FIXES ===")
    
    import inspect
    source = inspect.getsource(OpInfReducer._check_and_disable_unstable_dynamics)
    
    # Check for batch fix
    has_batch_fix = "batch_size = 5" in source or "U_zero = torch.zeros(batch_size" in source
    print(f"✓ Batch dimension fix: {'FOUND' if has_batch_fix else 'NOT FOUND'}")
    
    # Check GPOpInfDynamics
    try:
        from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
        source_forward = inspect.getsource(GPOpInfDynamics.forward)
        has_shape_check = "u.shape[0] != z.shape[0]" in source_forward
        print(f"✓ GPOpInf shape check: {'FOUND' if has_shape_check else 'NOT FOUND'}")
    except:
        print("✗ Could not check GPOpInfDynamics")
        has_shape_check = False
    
    return has_batch_fix and has_shape_check

def test_minimal_example():
    """Test with a minimal example to isolate the issue"""
    print("\n=== MINIMAL TEST ===")
    
    # Create minimal system
    M = torch.ones(10) 
    D = torch.ones(10) * 0.1
    P = torch.zeros(10)
    K = torch.eye(10) * 2.0
    
    params = dict(M=M, D=D, P=P, K=K)
    sys = SwingEquationSystem(params, dt=0.001)
    
    # Create simple data
    X = torch.randn(100, 19) * 0.1
    Xdot = torch.randn(100, 19) * 0.01
    
    print("\nTesting OpInf with minimal data:")
    try:
        opinf = OpInfReducer(19, 19, sys.n_controls)
        opinf.sys = sys
        opinf.fit(X, Xdot, lambda x: x.norm(dim=1), 0.1)
        
        # Test forward pass
        z_test = torch.randn(5, 19)
        u_test = torch.zeros(5, sys.n_controls)
        
        if opinf.dyn is not None:
            print(f"  OpInf dynamics exists")
            print(f"  dyn.d = {opinf.dyn.d}, dyn.m = {opinf.dyn.m}")
            print(f"  Testing forward pass...")
            z_dot = opinf.dyn.forward(z_test, u_test)
            print(f"  SUCCESS! z_dot shape: {z_dot.shape}")
        else:
            print("  WARNING: OpInf dynamics is None")
            
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("VERIFYING FIXES...")
    
    spr_ok = check_spr_fixes()
    lcr_ok = check_lcr_fixes()
    opinf_ok = check_opinf_fixes()
    
    print("\n=== SUMMARY ===")
    print(f"SPR fixes applied: {'YES' if spr_ok else 'NO'}")
    print(f"LCR fixes applied: {'YES' if lcr_ok else 'NO'}")
    print(f"OpInf fixes applied: {'YES' if opinf_ok else 'NO'}")
    
    if not all([spr_ok, lcr_ok, opinf_ok]):
        print("\n⚠️  Not all fixes have been applied!")
    else:
        print("\n✓ All fixes appear to be applied")
        print("\nRunning minimal test to check for other issues...")
        test_minimal_example()