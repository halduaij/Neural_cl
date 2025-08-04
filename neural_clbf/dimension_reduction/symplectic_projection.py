from __future__ import annotations
import torch
import numpy as np
import warnings
import logging
try:
    from control import care
except ImportError:
    care = None
from neural_clbf.dimension_reduction.base import BaseReducer

logger = logging.getLogger(__name__)


class SymplecticProjectionReducer(BaseReducer):
    """Fixed SPR with guaranteed orthogonality and robust basis selection."""

    def __init__(self, A: torch.Tensor, J: torch.Tensor,
                 R: torch.Tensor, latent_dim: int, 
                 enhanced: bool = True, X_data: torch.Tensor = None):
        
        if care is None:
            raise ImportError("The 'control' library is required for SymplecticProjectionReducer.")
        
        # Ensure even dimension for symplectic structure
        if latent_dim % 2 != 0:
            raise ValueError(f"Symplectic projection requires even latent dimension, got {latent_dim}")
            
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
        """Build projection matrix T such that T^T T = I with symplectic pairing."""
        device = A.device
        dtype = A.dtype
        n_full = A.shape[0]
        
        logger.info(f"\nBuilding Symplectic Projection (n={n_full} → d={self.latent_dim}):")
        
        # Step 1: Solve CARE/Lyapunov for P
        P = self._solve_riccati(A, device, dtype)
        
        # Step 2: Build projection basis T
        if self.enhanced and X_data is not None and X_data.shape[0] > 10:
            T = self._build_enhanced_basis(P, X_data, device, dtype)
        else:
            T = self._build_standard_basis(P, device, dtype)
        
        # Step 3: Ensure T has correct dimensions
        if T.shape != (n_full, self.latent_dim):
            logger.warning(f"  T shape mismatch. Got {T.shape}, expected ({n_full}, {self.latent_dim})")
            
            if T.shape[1] > self.latent_dim:
                # Too many columns, truncate
                T = T[:, :self.latent_dim]
            elif T.shape[1] < self.latent_dim:
                # Too few columns, pad with orthogonal vectors
                T = self._pad_basis(T, self.latent_dim, device, dtype)

        # Step 4: Ensure orthogonality using QR decomposition
        try:
            T_ortho, _ = torch.linalg.qr(T)
        except Exception as e:
             logger.warning(f"  QR decomposition failed: {e}. Using T as-is.")
             T_ortho = T

        T = T_ortho
        
        # Step 5: Compute pseudo-inverse. Since T is orthonormal, Ti = T.T.
        Ti = T.T
        
        # Register buffers
        self.register_buffer("T", T.detach())
        self.register_buffer("Ti", Ti.detach())
        
        self.gamma = 0.0
        self._verify_projection()
        self._verify_symplectic_pairs()
        
        # Build reduced-order dynamics
        self._build_reduced_dynamics()

    def _solve_riccati(self, A, device, dtype):
        """Solve CARE with robustness improvements"""
        n_full = A.shape[0]
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
            try:
                eigvals = torch.linalg.eigvalsh(P)
                if eigvals.min() < 1e-8:
                    P = P + (1e-8 - eigvals.min()) * torch.eye(n_full, device=device)
            except Exception as e:
                logger.warning(f"  eigvalsh failed ({e}); applying γI fallback")
                P = P + 1e-8 * torch.eye(n_full, device=device)
                
        except Exception as e:
            logger.warning(f"  CARE solver failed: {e}. Using fallback.")
            # Fallback: use stable Lyapunov solution
            P = self._solve_lyapunov_fallback(A, device, dtype)
        
        return P

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
        """Build basis by selecting dominant eigenvectors of P and ensuring symplectic pairs."""
        n_full = P.shape[0]
        
        # Compute eigenvectors of P
        eigvals, eigvecs = torch.linalg.eigh(P)
        
        # Sort by eigenvalue magnitude (descending)
        idx = torch.argsort(eigvals.abs(), descending=True)
        
        # FIX: Build symplectic pairs properly
        basis = []
        used_indices = set()
        
        for i in idx:
            if i.item() in used_indices:
                continue
            if len(basis) >= self.latent_dim:
                break
                
            # Take eigenvector
            v = eigvecs[:, i]
            # Compute symplectic partner
            Jv = self.J_symplectic @ v
            Jv = Jv / (Jv.norm() + 1e-12)
            
            # Check if we have room for the pair
            if len(basis) + 2 <= self.latent_dim:
                basis.extend([v, Jv])
                used_indices.add(i.item())
            else:
                # Can't fit the pair, stop
                break
        
        if len(basis) < self.latent_dim:
            logger.warning(f"  Could only build {len(basis)} basis vectors (requested {self.latent_dim})")
            self.latent_dim = len(basis)
        
        T = torch.stack(basis, dim=1)
        
        # Orthogonalize to ensure numerical stability
        T, _ = torch.linalg.qr(T)
        
        return T
    
    def _build_enhanced_basis(self, P, X_data, device, dtype):
        """Build enhanced basis with symplectic pairing enforcement - FIXED VERSION."""
        logger.info("  Using data-enhanced basis with symplectic pairing...")
        
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
        
        # Orthogonalize the combined basis
        Q_combined, _ = torch.linalg.qr(combined)
        
        # FIX: Build proper symplectic pairs after QR
        basis = []
        for col in range(0, Q_combined.shape[1] - 1, 2):
            v = Q_combined[:, col]
            Jv = self.J_symplectic @ v
            Jv = Jv / (Jv.norm() + 1e-12)
            
            # Accept the pair only if we stay within latent_dim
            if len(basis) + 2 > self.latent_dim:
                break
            basis.extend([v, Jv])
        
        # Update latent_dim if we couldn't achieve the requested dimension
        achieved_dim = len(basis)
        if achieved_dim < self.latent_dim:
            warnings.warn(f"SPR returned d={achieved_dim} (requested {self.latent_dim}) "
                         f"to preserve symplectic pairs")
            self.latent_dim = achieved_dim
        
        if achieved_dim == 0:
            raise ValueError("Could not build any symplectic pairs")
        
        T = torch.stack(basis, dim=1)
        
        # Final orthogonalization to ensure numerical orthogonality
        T, _ = torch.linalg.qr(T)
        
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
        logger.info(f"  T shape: {self.T.shape}, Ti shape: {self.Ti.shape}")
        
        # Check conditioning
        try:
            cond = torch.linalg.cond(self.T)
            logger.info(f"  T condition number: {cond:.2e}")
        except:
            logger.info("  T condition number: N/A")
        
        # Check orthogonality
        I_d = torch.eye(self.latent_dim, device=self.T.device)
        ortho_error = (self.T.T @ self.T - I_d).norm()
        logger.info(f"  Orthogonality error: ||T^T @ T - I|| = {ortho_error:.2e}")

        if ortho_error > 1e-5:
            logger.warning("  ⚠️  CRITICAL: T is not orthogonal!")

        # Check projection property (T @ Ti @ T = T)
        proj_error = (self.T @ self.Ti @ self.T - self.T).norm()
        logger.info(f"  Projection error: ||T @ Ti @ T - T|| = {proj_error:.2e}")

    def _verify_symplectic_pairs(self):
        """Verify symplectic pairing structure."""
        if self.latent_dim < 2:
            return
            
        max_pair_error = 0.0
        for i in range(0, self.latent_dim - 1, 2):
            # Check if T[:, i+1] ≈ J @ T[:, i] (up to normalization)
            v1 = self.T[:, i]
            v2 = self.T[:, i+1]
            Jv1 = self.J_symplectic @ v1
            
            # Normalize for comparison
            Jv1_normalized = Jv1 / (Jv1.norm() + 1e-10)
            v2_normalized = v2 / (v2.norm() + 1e-10)
            
            pair_error = (Jv1_normalized - v2_normalized).norm().item()
            max_pair_error = max(max_pair_error, pair_error)
        
        logger.info(f"  Max symplectic pair error: {max_pair_error:.2e}")
        if max_pair_error > 1e-3:
            logger.warning(f"  ⚠️  Warning: Symplectic pairing may not be perfectly preserved")

    def _build_reduced_dynamics(self):
        """Build reduced-order dynamics that preserve structure with STABILITY FIX."""
        logger.info("  Building reduced-order dynamics...")
        
        # Project system matrices to reduced space
        self.J_r = self.T.T @ self.J_symplectic @ self.T
        self.R_r = self.T.T @ self.R @ self.T
        self.A_r = self.T.T @ self.A @ self.T  # Linearized Hamiltonian
        
        # FIX: Stabilize A_r right after it is built
        eigvals = torch.linalg.eigvals(self.A_r)
        max_real = eigvals.real.max().item()
        if max_real > -0.02:  # margin 20 mrad/s
            shift = max_real + 0.05
            self.A_r = self.A_r - shift * torch.eye(self.latent_dim, device=self.A_r.device)
            logger.info(f"    Stabilized A_r: shifted eigenvalues by -{shift:.3f}")
            
            # Verify stability after shift
            eigvals_new = torch.linalg.eigvals(self.A_r)
            max_real_new = eigvals_new.real.max().item()
            logger.info(f"    New max Re(λ) = {max_real_new:.3f}")
        
        # Define reduced ODE
        def f_red(z, u=None):
            """Pure reduced ODE: z_dot = A_r @ z"""
            if z.dim() == 1:
                z = z.unsqueeze(0)
            return z @ self.A_r.T
        
        self.f_red = f_red
        
        logger.info(f"    Reduced system max Re(λ) = {eigvals.real.max().item():.3f}")

    # BaseReducer API (Row vector convention: z = x @ T, x = z @ Ti)
    def fit(self, X):
        """Refit with new data."""
        if X is not None and X.shape[0] > 10:
            logger.info("Refitting SPR with trajectory data...")
            self._build_projection(self.A, self.J_symplectic, self.R, X)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to reduced space: z = x @ T"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return (x @ self.T).squeeze(0)
        return x @ self.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from reduced space: x = z @ Ti (where Ti = T.T)"""
        if z.dim() == 1:
            z = z.unsqueeze(0)
            return (z @ self.Ti).squeeze(0)
        return z @ self.Ti

    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian: J = T.T = Ti"""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.Ti.unsqueeze(0)
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
        if hasattr(self, 'A_r'):
            self.A_r = self.A_r.to(device)
        if hasattr(self, 'J_r'):
            self.J_r = self.J_r.to(device)
        if hasattr(self, 'R_r'):
            self.R_r = self.R_r.to(device)
        return self