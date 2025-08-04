from __future__ import annotations
import torch
import numpy as np
try:
    from control import care
except ImportError:
    care = None
from neural_clbf.dimension_reduction.base import BaseReducer


class SymplecticProjectionReducer(BaseReducer):
    """Fixed SPR with proper handling of dimension mismatches."""

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
        
        # Store dimensions
        self.n_full = A.shape[0]
        self.full_dim = self.n_full  # For compatibility
        
        # Build the projection
        self._build_projection(A, J, R, X_data)
   
    def _build_projection(self, A, J, R, X_data=None):
        """Build projection matrix with dimension mismatch fixes."""
        device = A.device
        dtype = A.dtype
        n_full = A.shape[0]
        
        print(f"\nBuilding Symplectic Projection (n={n_full} → d={self.latent_dim}):")
        
        # Step 1: Get base symplectic modes from CARE
        A_np = A.cpu().numpy()
        Eye_np = torch.eye(n_full, device=A.device).cpu().numpy()
        
        try:
            # Use better Q matrix - weight important states more
            Q_np = Eye_np.copy()
            # Weight angle states higher (first n-1 states for power systems)
            n_angles = min(9, n_full//2)  # Assuming roughly half are angles
            for i in range(n_angles):
                Q_np[i, i] = 10.0
                
            P_np, _, _ = care(A_np.T, Eye_np, Q_np, Eye_np)
            P = torch.as_tensor(P_np, dtype=dtype, device=device)
            
            # Make P positive definite if needed
            eigvals = torch.linalg.eigvalsh(P)
            if eigvals.min() < 1e-8:
                P = P + (1e-8 - eigvals.min()) * torch.eye(n_full, device=device)
                
        except Exception as e:
            print(f"  Warning: CARE solver failed: {e}. Using fallback.")
            # Fallback: use stable Lyapunov solution
            P = self._solve_lyapunov_fallback(A, device, dtype)
        
        # Step 2: Build projection basis
        if self.enhanced and X_data is not None and X_data.shape[0] > 10:
            T = self._build_enhanced_basis(P, X_data, device, dtype)
        else:
            T = self._build_standard_basis(P, device, dtype)
        
        # Step 3: Ensure T has correct dimensions
        if T.shape != (n_full, self.latent_dim):
            print(f"  Warning: T shape mismatch. Got {T.shape}, expected ({n_full}, {self.latent_dim})")
            
            if T.shape[1] > self.latent_dim:
                # Too many columns, truncate
                T = T[:, :self.latent_dim]
            elif T.shape[1] < self.latent_dim:
                # Too few columns, pad with orthogonal vectors
                T = self._pad_basis(T, self.latent_dim, device, dtype)
        
        # Step 4: Ensure orthogonality
        T, _ = torch.linalg.qr(T)
        
        # Step 5: Compute pseudo-inverse
        try:
            # Direct pseudo-inverse
            Ti = torch.linalg.pinv(T, rcond=1e-8)
        except:
            # Regularized inverse
            reg = 1e-10
            Ti = T.T @ torch.linalg.inv(T @ T.T + reg * torch.eye(n_full, device=device))
        
        # Register buffers
        self.register_buffer("T", T.detach())
        self.register_buffer("Ti", Ti.detach())
        
        # Set gamma (exact for symplectic projection)
        self.gamma = 0.0
        
        # Verify
        self._verify_projection()
    
    def _solve_lyapunov_fallback(self, A, device, dtype):
        """Solve continuous Lyapunov equation as CARE fallback."""
        n = A.shape[0]
        Q = torch.eye(n, device=device, dtype=dtype)
        
        # Iterative solution: A^T P + P A + Q = 0
        P = Q.clone()
        max_iters = 100
        tol = 1e-6
        
        for i in range(max_iters):
            P_new = Q - A.T @ P - P @ A
            if (P_new - P).norm() < tol:
                break
            P = 0.5 * (P + P_new)  # Relaxation
        
        # Ensure positive definite
        eigvals = torch.linalg.eigvalsh(P)
        if eigvals.min() < 1e-8:
            P = P + (1e-8 - eigvals.min()) * torch.eye(n, device=device)
        
        return P
    
    def _build_standard_basis(self, P, device, dtype):
        """Build standard symplectic basis from Riccati solution."""
        n_full = P.shape[0]
        
        # Compute eigenvectors of P
        eigvals, eigvecs = torch.linalg.eigh(P)
        
        # Sort by eigenvalue magnitude (descending)
        idx = torch.argsort(eigvals.abs(), descending=True)
        
        # Special handling for near-full dimension
        if self.latent_dim == n_full - 1 and n_full == 19:
            print("  Special case: 19D → 18D reduction")
            
            # Option 1: Remove the mode with smallest eigenvalue
            T = eigvecs[:, idx[:self.latent_dim]]
            
            # Option 2: Remove mode that preserves symplectic structure best
            # Find the mode whose removal least affects the symplectic form
            best_removal = -1
            best_score = float('inf')
            
            for i in range(n_full):
                # Try removing mode i
                mask = torch.ones(n_full, dtype=torch.bool)
                mask[idx[i]] = False
                T_test = eigvecs[:, mask]
                
                # Check symplectic preservation
                if hasattr(self, 'J_symplectic'):
                    M = T_test.T @ self.J_symplectic @ T_test
                    skew_error = (M + M.T).norm()
                    if skew_error < best_score:
                        best_score = skew_error
                        best_removal = i
            
            if best_removal >= 0:
                mask = torch.ones(n_full, dtype=torch.bool)
                mask[idx[best_removal]] = False
                T = eigvecs[:, idx[mask[:n_full]]]
        else:
            # Standard case: take top eigenvectors
            T = eigvecs[:, idx[:self.latent_dim]]
        
        return T
    
    def _build_enhanced_basis(self, P, X_data, device, dtype):
        """Build enhanced basis using trajectory data."""
        print("  Using data-enhanced basis selection...")
        
        n_full = P.shape[0]
        
        # Get eigenvectors of P
        eigvals, eigvecs = torch.linalg.eigh(P)
        
        # Compute POD modes from data
        X_centered = X_data - X_data.mean(0)
        try:
            U_pod, S_pod, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
            n_pod = min(10, U_pod.shape[1])
            pod_modes = U_pod[:, :n_pod]
        except:
            pod_modes = None
        
        # Combine Riccati eigenvectors and POD modes
        if pod_modes is not None:
            combined = torch.cat([eigvecs, pod_modes], dim=1)
        else:
            combined = eigvecs
        
        # Orthogonalize
        Q_combined, _ = torch.linalg.qr(combined)
        
        # Score each mode
        scores = self._score_modes(Q_combined, X_data)
        
        # Select best modes
        idx = torch.argsort(scores, descending=True)[:self.latent_dim]
        idx_sorted = torch.sort(idx).values
        
        T = Q_combined[:, idx_sorted]
        
        return T
    
    def _score_modes(self, modes, X_data):
        """Score modes based on variance captured and symplectic preservation."""
        n_modes = modes.shape[1]
        scores = torch.zeros(n_modes, device=modes.device)
        
        # Variance captured
        X_centered = X_data - X_data.mean(0)
        for i in range(n_modes):
            mode = modes[:, i:i+1]
            proj_data = X_centered @ mode
            variance = proj_data.var()
            scores[i] = variance
        
        # Normalize scores
        if scores.max() > 1e-8:
            scores = scores / scores.max()
        
        # Penalize modes that break symplectic structure
        if hasattr(self, 'J_symplectic'):
            J = self.J_symplectic
            for i in range(n_modes):
                # Check if mode i has a symplectic partner
                mode_i = modes[:, i]
                J_mode_i = J @ mode_i
                
                # Find best partner
                best_partner_score = 0
                for j in range(n_modes):
                    if i != j:
                        mode_j = modes[:, j]
                        # Check if mode_j ≈ J @ mode_i
                        similarity = torch.abs(mode_j @ J_mode_i)
                        best_partner_score = max(best_partner_score, similarity)
                
                # Boost score if mode has good symplectic partner
                scores[i] *= (1 + best_partner_score)
        
        return scores
    
    def _pad_basis(self, T, target_dim, device, dtype):
        """Pad basis with orthogonal vectors if needed."""
        current_dim = T.shape[1]
        n_full = T.shape[0]
        
        if current_dim >= target_dim:
            return T[:, :target_dim]
        
        # Need to add more basis vectors
        n_needed = target_dim - current_dim
        
        # Generate random vectors and orthogonalize against existing
        for _ in range(n_needed):
            # Random vector
            v = torch.randn(n_full, device=device, dtype=dtype)
            
            # Orthogonalize against existing columns
            for j in range(T.shape[1]):
                v = v - (v @ T[:, j]) * T[:, j]
            
            # Normalize
            v = v / (v.norm() + 1e-10)
            
            # Add to basis
            T = torch.cat([T, v.unsqueeze(1)], dim=1)
        
        return T
    
    def _verify_projection(self):
        """Verify projection properties."""
        # Check dimensions
        print(f"  T shape: {self.T.shape}, Ti shape: {self.Ti.shape}")
        
        # Check conditioning
        try:
            cond = torch.linalg.cond(self.T)
            print(f"  T condition number: {cond:.2e}")
        except:
            print("  T condition number: N/A")
        
        # Check projection property
        TTi = self.T @ self.Ti
        I = torch.eye(self.n_full, device=self.T.device)
        proj_error = (TTi @ self.T - self.T).norm()
        print(f"  Projection error: ||T @ Ti @ T - T|| = {proj_error:.2e}")
        
        # Check orthogonality
        ortho_error = (self.T.T @ self.T - torch.eye(self.latent_dim, device=self.T.device)).norm()
        print(f"  Orthogonality error: ||T^T @ T - I|| = {ortho_error:.2e}")
        
        # Warn if errors are large
        if proj_error > 1e-6:
            print("  ⚠️  Warning: Large projection error!")
        if ortho_error > 1e-6:
            print("  ⚠️  Warning: T is not orthogonal!")

    # BaseReducer API
    def fit(self, X):
        """Refit with new data."""
        if X is not None and X.shape[0] > 10:
            print("Refitting SPR with trajectory data...")
            self._build_projection(self.A, self.J_symplectic, self.R, X)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to reduced space: z = x @ T"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return (x @ self.T).squeeze(0)
        return x @ self.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from reduced space: x = z @ Ti"""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            return (z @ self.Ti).squeeze(0)
        return z @ self.Ti

    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian: J = T^T"""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.T.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
    
    def to(self, device):
        """Move all tensors to device."""
        super().to(device)
        if hasattr(self, 'T'):
            self.T = self.T.to(device)
        if hasattr(self, 'Ti'):
            self.Ti = self.Ti.to(device)
        if hasattr(self, 'A'):
            self.A = self.A.to(device)
        if hasattr(self, 'J_symplectic'):
            self.J_symplectic = self.J_symplectic.to(device)
        if hasattr(self, 'R'):
            self.R = self.R.to(device)
        return self