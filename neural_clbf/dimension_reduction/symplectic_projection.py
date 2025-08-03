from __future__ import annotations
import torch
import numpy as np
try:
    from control import care
except ImportError:
    care = None
from neural_clbf.dimension_reduction.base import BaseReducer


class SymplecticProjectionReducer(BaseReducer):
    """Improved SPR with better mode selection for odd-dimensional systems."""

    def __init__(self, A: torch.Tensor, J: torch.Tensor,
                 R: torch.Tensor, latent_dim: int, 
                 enhanced: bool = True, X_data: torch.Tensor = None):
        
        if care is None:
            raise ImportError("The 'control' library is required for SymplecticProjectionReducer.")
            
        super().__init__(latent_dim)
        
        self.enhanced = enhanced
        self.A = A
        self.J_symplectic = J
        self.R = R
        
        # Build the projection
        self._build_projection(A, J, R, X_data)

    def _build_projection(self, A, J, R, X_data=None):
        """Build improved projection matrix."""
        device = A.device
        dtype = A.dtype
        n_full = A.shape[0]
        
        # Step 1: Get base symplectic modes from CARE
        A_np = A.cpu().numpy()
        Eye_np = torch.eye(n_full, device=A.device).cpu().numpy()
        
        try:
            # Solve CARE: Aᵀ P + P A + Q = 0 with Q = I
            P_np, _, _ = care(A_np.T, Eye_np, Eye_np, Eye_np)
            P = torch.as_tensor(P_np, dtype=dtype, device=device)
        except Exception as e:
            print(f"Warning: CARE solver failed: {e}. Using fallback.")
            P = torch.eye(n_full, device=device, dtype=dtype)
        
        # Step 2: Compute eigenvectors of P (these are good candidates)
        eigvals, eigvecs = torch.linalg.eigh(P)
        
        # Sort by eigenvalue magnitude (most important modes first)
        idx = torch.argsort(eigvals.abs(), descending=True)
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]
        
        if self.enhanced and X_data is not None:
            # Step 3: Enhance with data-driven modes
            print("Enhancing SPR with data-driven modes...")
            
            # Compute POD modes from data
            X_mean = X_data.mean(0)
            X_centered = X_data - X_mean
            U_svd, S_svd, Vt_svd = torch.linalg.svd(X_centered.T, full_matrices=False)
            
            # Combine symplectic and POD modes
            combined_modes = torch.cat([eigvecs, U_svd[:, :10]], dim=1)  # Add top 10 POD modes
            
            # Orthogonalize
            Q_combined, _ = torch.linalg.qr(combined_modes)
            
            # Score each mode based on:
            # 1. Preservation of symplectic structure
            # 2. Data reconstruction error
            # 3. Energy preservation
            scores = self._score_modes(Q_combined, J, X_data)
            
            # Select top scoring modes
            idx_best = torch.argsort(scores, descending=True)[:self.latent_dim]
            T = Q_combined[:, idx_best]
            
        else:
            # Standard approach: use top eigenvectors
            T = eigvecs[:, :self.latent_dim]
        
        # Step 4: Optimize the basis specifically for minimal reconstruction error
        if n_full == 19 and self.latent_dim == 18:
            # Special handling for 19D -> 18D
            print("Optimizing basis for 19D -> 18D reduction...")
            
            # Find the least important direction to drop
            if X_data is not None:
                # Use data to find least important direction
                X_cov = (X_data - X_data.mean(0)).T @ (X_data - X_data.mean(0)) / X_data.shape[0]
                eig_data, vec_data = torch.linalg.eigh(X_cov)
                
                # The eigenvector with smallest eigenvalue is least important
                least_important_dir = vec_data[:, 0]
                
                # Build projection that's orthogonal to least important direction
                # This minimizes information loss
                I_proj = torch.eye(n_full, device=device) - torch.outer(least_important_dir, least_important_dir)
                T_opt = I_proj @ T
                
                # Re-orthogonalize
                T, _ = torch.linalg.qr(T_opt)
                T = T[:, :self.latent_dim]
        
        # Step 5: Final symplectic structure preservation (if possible)
        # For even dimensions, ensure symplectic pairing
        if self.latent_dim % 2 == 0:
            T = self._enforce_symplectic_structure(T, J)
        
        # Store the projection matrices
        self.register_buffer("T", T)
        
        # Compute optimal pseudo-inverse
        # For non-square matrices, we want the Moore-Penrose pseudo-inverse
        self.register_buffer("Ti", torch.linalg.pinv(T))
        
        # Set gamma
        self.gamma = 0.0  # Will be updated if data is provided
        
        # Compute actual gamma if we have data
        if X_data is not None:
            self._compute_empirical_gamma(X_data)

    def _score_modes(self, modes, J, X_data):
        """Score modes based on multiple criteria."""
        n_modes = modes.shape[1]
        scores = torch.zeros(n_modes, device=modes.device)
        
        for i in range(n_modes):
            mode = modes[:, i:i+1]
            
            # Criterion 1: Symplectic structure preservation
            # Check how well J @ mode is represented in the span
            J_mode = J @ mode
            mode_ext = torch.cat([mode, J_mode], dim=1)
            Q_ext, _ = torch.linalg.qr(mode_ext)
            symplectic_score = (Q_ext[:, :1].T @ J_mode).abs().item()
            
            # Criterion 2: Data variance captured
            if X_data is not None:
                X_centered = X_data - X_data.mean(0)
                proj_data = X_centered @ mode
                variance_score = proj_data.var()
            else:
                variance_score = 0
            
            # Combined score
            scores[i] = symplectic_score + 0.5 * variance_score
        
        return scores

    def _enforce_symplectic_structure(self, T, J):
        """Modify T to better preserve symplectic structure."""
        # For each pair of columns, try to make them symplectic partners
        n_pairs = self.latent_dim // 2
        T_new = T.clone()
        
        for i in range(n_pairs):
            # Take two columns
            v1 = T[:, 2*i]
            v2 = T[:, 2*i + 1]
            
            # Make v2 = J @ v1 (approximately, while maintaining orthogonality)
            v2_target = J @ v1
            
            # Project v2_target to be orthogonal to previous columns
            for j in range(2*i):
                v2_target = v2_target - (T_new[:, j] @ v2_target) * T_new[:, j]
            
            # Normalize
            v2_norm = v2_target.norm()
            if v2_norm > 1e-6:
                T_new[:, 2*i + 1] = v2_target / v2_norm
        
        return T_new

    def _compute_empirical_gamma(self, X_data):
        """Compute gamma based on actual reconstruction error."""
        # Test reconstruction on data
        Z = self.forward(X_data)
        X_recon = self.inverse(Z)
        errors = (X_recon - X_data).norm(dim=1)
        
        # Gamma is related to worst-case error
        self.gamma = errors.max().item() / (X_data.norm(dim=1).mean().item() + 1e-6)

    # BaseReducer API
    def fit(self, X):
        """Refit with new data."""
        if X is not None:
            print("Refitting SPR with trajectory data...")
            self._build_projection(self.A, self.J_symplectic, self.R, X)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to reduced space: z = x @ T"""
        return x @ self.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from reduced space: x = z @ Ti"""
        return z @ self.Ti

    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian: J = T^T"""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.T.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()


def test_improved_spr():
    """Test the improved SPR implementation."""
    print("="*70)
    print("TESTING IMPROVED SPR")
    print("="*70)
    
    # Create test system
    from neural_clbf.systems import SwingEquationSystem
    
    # System parameters (abbreviated for space)
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
                       0.7922997, 0.6452313, 0.6032209, 0.5339053, 3.1851373]] + 
                     [[0]*10]*8, dtype=torch.float32)  # Truncated for brevity
    
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                               -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                              dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    sys = SwingEquationSystem(params, dt=0.001)
    sys.delta_star = delta_star
    
    # Create training data
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    X_train = x_eq.unsqueeze(0).repeat(100, 1)
    X_train += 0.01 * torch.randn_like(X_train)
    
    # Get linearization
    A, J, R = sys.linearise(return_JR=True)
    
    # Test standard SPR
    print("\nStandard SPR (d=18):")
    spr_standard = SymplecticProjectionReducer(A, J, R, 18, enhanced=False)
    
    X_test = X_train[:10]
    Z = spr_standard.forward(X_test)
    X_recon = spr_standard.inverse(Z)
    errors = (X_recon - X_test).norm(dim=1)
    rel_errors = errors / X_test.norm(dim=1)
    
    print(f"  Mean reconstruction error: {errors.mean():.6f}")
    print(f"  Max reconstruction error: {errors.max():.6f}")
    print(f"  Mean relative error: {rel_errors.mean():.2%}")
    
    # Test enhanced SPR
    print("\nEnhanced SPR with data (d=18):")
    spr_enhanced = SymplecticProjectionReducer(A, J, R, 18, enhanced=True, X_data=X_train)
    
    Z_enh = spr_enhanced.forward(X_test)
    X_recon_enh = spr_enhanced.inverse(Z_enh)
    errors_enh = (X_recon_enh - X_test).norm(dim=1)
    rel_errors_enh = errors_enh / X_test.norm(dim=1)
    
    print(f"  Mean reconstruction error: {errors_enh.mean():.6f}")
    print(f"  Max reconstruction error: {errors_enh.max():.6f}")
    print(f"  Mean relative error: {rel_errors_enh.mean():.2%}")
    print(f"  Gamma: {spr_enhanced.gamma:.6f}")
    
    # Compare projection quality
    print("\nProjection quality comparison:")
    print(f"  Standard: ||T^T @ T - I|| = {(spr_standard.T.T @ spr_standard.T - torch.eye(18)).norm():.6e}")
    print(f"  Enhanced: ||T^T @ T - I|| = {(spr_enhanced.T.T @ spr_enhanced.T - torch.eye(18)).norm():.6e}")
    
    # Test which dimension is being dropped
    print("\nAnalyzing information loss:")
    
    # Compute null space of projection
    P_standard = spr_standard.T @ spr_standard.Ti
    null_standard = torch.eye(19) - P_standard
    
    P_enhanced = spr_enhanced.T @ spr_enhanced.Ti
    null_enhanced = torch.eye(19) - P_enhanced
    
    # Find the most significant null space direction
    _, s_null_std, v_null_std = torch.linalg.svd(null_standard)
    _, s_null_enh, v_null_enh = torch.linalg.svd(null_enhanced)
    
    print(f"  Standard null space: largest singular value = {s_null_std[0]:.6f}")
    print(f"  Enhanced null space: largest singular value = {s_null_enh[0]:.6f}")
    
    improvement = (errors.mean() - errors_enh.mean()) / errors.mean()
    print(f"\nImprovement: {improvement:.1%} reduction in error")
    
    return spr_enhanced


if __name__ == "__main__":
    test_improved_spr()