from __future__ import annotations
import torch
from .symplectic_projection import SymplecticProjectionReducer as SPR
from .opinf import OpInfReducer
from .lyap_coherency import LyapCoherencyReducer as LCR


def _score(r, w):
    """Compute score for reducer selection (lower is better)."""
    return r.gamma + w * r.latent_dim / r.full_dim


def select_reducer(sys, X, Xdot, d_max=12, w_dim=0.02, verbose=False):
    """
    Select the best dimension reducer for the given system and data.
    
    Args:
        sys: The dynamical system
        X: (N, n) state snapshots
        Xdot: (N, n) state derivatives  
        d_max: Maximum latent dimension to consider
        w_dim: Weight for dimension penalty in scoring
        verbose: Print debug information
        
    Returns:
        The best reducer according to the scoring function
    """
    n = X.shape[1]
    device = X.device
    
    # Compute energy function values for gamma calculation
    V = sys.energy_function
    V_vals = V(X)
    V_min = V_vals.min().item()
    
    # Ensure V_min is not too small to avoid numerical issues
    V_min = max(V_min, 1e-3)
    
    if verbose:
        print(f"Selecting reducer: n={n}, d_max={d_max}, V_min={V_min:.6f}")
    
    # Ensure X requires grad for Lyapunov computations
    if not X.requires_grad:
        X = X.clone().requires_grad_(True)
        
    candidates = []
    
    # 1. Try Symplectic Projection (if system supports linearization)
    if hasattr(sys, "linearise"):
        try:
            if verbose:
                print("Trying Symplectic Projection...")
            A, J, R = sys.linearise(return_JR=True)
            # Try different even dimensions
            for d in range(2, min(d_max + 1, n), 2):
                try:
                    spr = SPR(A, J, R, d)
                    spr.full_dim = n
                    candidates.append(spr)
                    if verbose:
                        print(f"  SPR(d={d}): gamma={spr.gamma}")
                except Exception as e:
                    if verbose:
                        print(f"  SPR(d={d}) failed: {e}")
        except Exception as e:
            if verbose:
                print(f"Symplectic Projection failed: {e}")
    
    # 2. Try Lyapunov Coherency Reducer
    if hasattr(sys, 'n_machines') or hasattr(sys, 'N_NODES'):
        try:
            if verbose:
                print("Trying Lyapunov Coherency...")
            # Try different numbers of groups (2, 3, 4)
            for k in (2, 3, 4):
                if 2 * k <= min(d_max, n - 1):  # Each group needs 2 dimensions
                    try:
                        lcr = LCR(sys, k, X)
                        lcr.full_dim = n
                        lcr.gamma = lcr.compute_gamma(V_min)
                        # Only add if gamma is reasonable
                        if lcr.gamma < 100:
                            candidates.append(lcr)
                            if verbose:
                                print(f"  LCR(k={k}): d={lcr.latent_dim}, gamma={lcr.gamma:.6f}")
                        elif verbose:
                            print(f"  LCR(k={k}): d={lcr.latent_dim}, gamma={lcr.gamma:.6f} (too high)")
                    except Exception as e:
                        if verbose:
                            print(f"  LCR(k={k}) failed: {e}")
        except Exception as e:
            if verbose:
                print(f"Lyapunov Coherency failed: {e}")
    
    # 3. Try Operator Inference Reducer
    try:
        if verbose:
            print("Trying Operator Inference...")
        # Try different dimensions
        for d in range(4, min(d_max + 1, n)):
            try:
                # Get actual number of controls from system
                n_controls = getattr(sys, 'n_controls', 1)
                
                opinf = OpInfReducer(d, n, n_controls)
                # Pass system reference if needed
                opinf.sys = sys
                opinf.fit(X, Xdot, V, V_min)
                opinf.full_dim = n
                
                # Only add if gamma is reasonable
                if opinf.gamma < 100:
                    candidates.append(opinf)
                    if verbose:
                        print(f"  OpInf(d={d}): gamma={opinf.gamma:.6f}")
                elif verbose:
                    print(f"  OpInf(d={d}): gamma={opinf.gamma:.6f} (too high)")
                    
            except Exception as e:
                if verbose:
                    print(f"  OpInf(d={d}) failed: {e}")
    except Exception as e:
        if verbose:
            print(f"Operator Inference failed: {e}")
    
    # Select best candidate
    if not candidates:
        # If no candidates, try to create at least one fallback
        print("Warning: No valid reducers found with reasonable gamma, creating fallback")
        d_fallback = min(d_max, n - 1)
        opinf = OpInfReducer(d_fallback, n, getattr(sys, 'n_controls', 1))
        opinf.sys = sys
        try:
            opinf.fit(X, Xdot, V, V_min)
        except:
            # Even simpler fallback
            opinf.μ = X.mean(0)
            opinf.proj = torch.eye(n, d_fallback, device=device)[:, :d_fallback]
            opinf.gamma = 10.0
        opinf.full_dim = n
        candidates = [opinf]
    
    best = min(candidates, key=lambda r: _score(r, w_dim))
    
    if verbose:
        print(f"\nSelected: {type(best).__name__} with d={best.latent_dim}, "
              f"gamma={best.gamma:.6f}, score={_score(best, w_dim):.6f}")
    
    # Ensure the reducer is on the correct device
    best = best.to(device)
    
    return best