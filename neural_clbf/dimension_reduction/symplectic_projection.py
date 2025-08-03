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
            # Use better Q matrix - weight important states more
            Q_np = Eye_np.copy()
            # Weight angle states higher (first n-1 states)
            for i in range(min(9, n_full)):  # Assuming 9 angle states
                Q_np[i, i] = 10.0
                
            P_np, _, _ = care(A_np.T, Eye_np, Q_np, Eye_np)
            P = torch.as_tensor(P_np, dtype=dtype, device=device)
        except Exception as e:
            print(f"Warning: CARE solver failed: {e}. Using fallback.")
            P = torch.eye(n_full, device=device, dtype=dtype)
        
        # Step 2: Compute eigenvectors of P
        eigvals, eigvecs = torch.linalg.eigh(P)
        
        # Step 3: If we have training data, use it to select best modes
        if X_data is not None and X_data.shape[0] > 0:
            print("  Using data-driven mode selection...")
            
            # Compute POD modes from data
            X_centered = X_data - X_data.mean(0)
            U_pod, S_pod, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
            
            # Combine symplectic eigenvectors and POD modes
            combined_modes = torch.cat([eigvecs, U_pod[:, :10]], dim=1)
            
            # Orthogonalize
            Q_combined, _ = torch.linalg.qr(combined_modes)
            
            # Score each mode
            scores = []
            x_eq = X_data[0]  # Use first point as reference
            
            for i in range(Q_combined.shape[1]):
                mode = Q_combined[:, i:i+1]
                
                # Reconstruction score on training data
                proj_data = X_centered @ mode @ mode.T
                recon_error = (X_centered - proj_data).norm() / X_centered.norm()
                
                # Energy preservation score

                if x_eq.dim() == 1:
                    x_eq_batch = x_eq.unsqueeze(0)
                else:
                    x_eq_batch = x_eq
                x_proj = x_eq_batch @ self.T  # Project first
                x_recon = x_proj @ self.Ti     # Then reconstruct
                error = (x_recon - x_eq_batch).norm()

                energy_error = x_proj.norm() / x_eq.norm()
                
                # Combined score (lower is better)
                score = recon_error + 0.5 * energy_error
                scores.append(score.item())
            
            # Select best modes
            scores = torch.tensor(scores)
            best_indices = torch.argsort(scores)[:self.latent_dim]
            best_indices = torch.sort(best_indices).values
            
            self.T = Q_combined[:, best_indices]
            
        else:
            # No data - use standard approach
            if n_full == 19 and self.latent_dim == 18:
                print("  Optimizing 19D -> 18D reduction...")
                idx = torch.argsort(eigvals.abs(), descending=True)[:18]
                self.T = eigvecs[:, idx]
            else:
                idx = torch.argsort(eigvals.abs(), descending=True)
                self.T = eigvecs[:, idx[:self.latent_dim]]
        
        # Step 4: Improve conditioning
        self.T = self.T + 1e-8 * torch.randn_like(self.T)
        self.T, _ = torch.linalg.qr(self.T)
        
        # Step 5: Compute pseudo-inverse with regularization
        regularization = 1e-10
        self.Ti = self.T.T @ torch.linalg.inv(
            self.T @ self.T.T + regularization * torch.eye(self.latent_dim, device=device)
        )
        
        # Register buffers
        self.register_buffer("T", self.T)
        self.register_buffer("Ti", self.Ti)
        
        # Set gamma
        self.gamma = 0.0
        
        # Print diagnostics
        print(f"  SPR projection matrix condition number: {torch.linalg.cond(self.T):.2e}")
        proj_error = torch.norm(self.T @ self.Ti @ self.T - self.T)
        print(f"  Projection property error: {proj_error:.2e}")

    

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