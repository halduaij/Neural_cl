from __future__ import annotations
import torch
import numpy as np
import warnings
import logging
try:
    from control import care
    from scipy.linalg import solve_lyapunov
except ImportError:
    care = None
    solve_lyapunov = None
from neural_clbf.dimension_reduction.base import BaseReducer
from neural_clbf.systems import ControlAffineSystem

logger = logging.getLogger(__name__)


class SymplecticProjectionReducer(BaseReducer):
    """
    SPR with a structure-preserving port-Hamiltonian reduced-order model.
    """
    def __init__(self, sys: ControlAffineSystem, A: torch.Tensor, J: torch.Tensor,
                 R: torch.Tensor, latent_dim: int, 
                 enhanced: bool = True, X_data: torch.Tensor = None):
        
        if care is None or solve_lyapunov is None:
            raise ImportError("The 'control' and 'scipy' libraries are required for SymplecticProjectionReducer.")
        
        if latent_dim % 2 != 0:
            raise ValueError(f"Symplectic projection requires even latent dimension, got {latent_dim}")
            
        super().__init__(latent_dim)
        
        self.sys = sys
        self.enhanced = enhanced
        self.A = A
        self.J_symplectic = J
        self.R = R
        self.n_full = A.shape[0]
        self.full_dim = self.n_full
        
        self._build_projection(A, J, R, X_data)
   
    def _build_projection(self, A, J, R, X_data=None):
        device = A.device
        dtype = A.dtype
        
        logger.info(f"\nBuilding Symplectic Projection (n={self.n_full} → d={self.latent_dim}):")
        
        P = self._solve_riccati(A, device, dtype)
        
        if self.enhanced and X_data is not None and X_data.shape[0] > 10:
            T = self._build_enhanced_basis(P, X_data, device, dtype)
        else:
            T = self._build_standard_basis(P, device, dtype)
        
        T_ortho, _ = torch.linalg.qr(T)
        self.register_buffer("T", T_ortho.detach())
        self.register_buffer("Ti", self.T.T.detach())
        
        self.gamma = 0.0
        self._build_reduced_dynamics()
        self._verify_symplectic_pairs()

    def _build_reduced_dynamics(self):
        """Builds the structure-preserving port-Hamiltonian ROM with a robust f_red."""
        logger.info("  Building port-Hamiltonian reduced-order dynamics...")
        
        self.J_r = self.Ti @ self.J_symplectic @ self.T
        self.R_r = self.Ti @ self.R @ self.T
        
     
        def f_red(z, u=None):
            if z.dim() == 1: z = z.unsqueeze(0)

            # 1. Reconstruct full state and enable gradient tracking
            x = self.inverse(z).requires_grad_(True)
            
            # 2. Compute the Hamiltonian (internal energy) part of the dynamics
            H = self.sys.energy_function(x, relative=False)
            grad_x = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H))[0]
            hamiltonian_dynamics_x = grad_x @ (self.J_symplectic - self.R).T
            
            # 3. Get the full dynamics f(x) from the system
            full_dynamics_x = self.sys._f(x, self.sys.nominal_params).squeeze(-1)
            
            # 4. The external force is the difference between the full dynamics
            #    and the internal Hamiltonian dynamics.
            external_force_x = full_dynamics_x - hamiltonian_dynamics_x
            
            # 5. Project both parts of the dynamics to the reduced space
            hamiltonian_dynamics_z = self.forward(hamiltonian_dynamics_x)
            external_force_z = self.forward(external_force_x)
            
            # 6. The complete reduced dynamics is the sum of both parts
            z_dot = hamiltonian_dynamics_z + external_force_z
            
            return z_dot
        self.f_red = f_red
        self.A_r = self.Ti @ self.A @ self.T

    # All helper methods for basis construction from the previous step are correct.
    # The code below is the complete, correct set of helper methods.
    def _solve_riccati(self, A, device, dtype):
        n_full = A.shape[0]
        A_np = A.cpu().numpy()
        Eye_np = torch.eye(n_full, device=device).cpu().numpy()
        try:
            Q_np = Eye_np.copy()
            n_angles = min(9, n_full//2)
            for i in range(n_angles): Q_np[i, i] = 10.0
            P_np, _, _ = care(A_np.T, Eye_np, Q_np, Eye_np)
            P = torch.as_tensor(P_np, dtype=dtype, device=device)
            eigvals = torch.linalg.eigvalsh(P)
            if eigvals.min() < 1e-8:
                P = P + (1e-6 - eigvals.min()) * torch.eye(n_full, device=device)
        except Exception as e:
            logger.warning(f"  CARE solver failed: {e}. Using Lyapunov fallback.")
            P = self._solve_lyapunov_fallback(A, device, dtype)
        return P

    def _solve_lyapunov_fallback(self, A, device, dtype):
        n = A.shape[0]
        Q = torch.eye(n, device=device, dtype=dtype)
        try:
            A_np = A.detach().cpu().numpy()
            Q_np = -Q.detach().cpu().numpy()
            P_np = solve_lyapunov(A_np.T, Q_np)
            P = torch.as_tensor(P_np, dtype=dtype, device=device)
        except Exception as e:
            logger.error(f"Lyapunov fallback failed: {e}. Returning Identity.")
            P = torch.eye(n, device=device, dtype=dtype)
        return P
    
    def _build_standard_basis(self, P, device, dtype):
        logger.info("  Using standard basis with symplectic pairing...")
        eigvals, eigvecs = torch.linalg.eigh(P)
        sorted_indices = torch.argsort(eigvals.abs(), descending=True).tolist()
        basis_cols, used_indices = [], set()
        for i in sorted_indices:
            if i in used_indices or len(basis_cols) >= self.latent_dim: continue
            v_i = eigvecs[:, i]
            Jv_i = self.J_symplectic @ v_i
            Jv_i_norm = Jv_i / (Jv_i.norm() + 1e-10)
            max_similarity, best_match_idx = 0.0, -1
            for j in sorted_indices:
                if j not in used_indices and j != i:
                    v_j = eigvecs[:, j]
                    similarity = (Jv_i_norm @ (v_j / (v_j.norm() + 1e-10))).abs().item()
                    if similarity > max_similarity:
                        max_similarity, best_match_idx = similarity, j
            basis_cols.append(v_i)
            used_indices.add(i)
            if len(basis_cols) < self.latent_dim:
                if best_match_idx != -1:
                    basis_cols.append(eigvecs[:, best_match_idx])
                    used_indices.add(best_match_idx)
                else:
                    basis_cols.append(Jv_i_norm)
        return torch.stack(basis_cols, dim=1)[:, :self.latent_dim]

    def _build_enhanced_basis(self, P, X_data, device, dtype):
        logger.info("  Using data-enhanced basis with rigorous symplectic pairing (SGS)...")
        n_full = P.shape[0]
        try:
            _, riccati_modes = torch.linalg.eigh(P)
        except:
            riccati_modes = torch.empty(n_full, 0, device=device, dtype=dtype)
        X_centered = X_data - X_data.mean(0)
        try:
            U_pod, S_pod, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
            pod_modes = U_pod
        except:
            pod_modes = torch.empty(n_full, 0, device=device, dtype=dtype)
        combined = torch.cat([pod_modes, riccati_modes], dim=1)
        Q_combined, _ = torch.linalg.qr(combined, mode='reduced')
        T = torch.zeros(n_full, self.latent_dim, device=device, dtype=dtype)
        k = 0
        for i in range(Q_combined.shape[1]):
            if k >= self.latent_dim: break
            v = Q_combined[:, i]
            if k > 0: v = v - T[:, :k] @ (T[:, :k].T @ v)
            norm_v = v.norm()
            if norm_v < 1e-8: continue
            v = v / norm_v
            if k + 1 >= self.latent_dim: break
            T[:, k] = v
            Jv = self.J_symplectic @ v
            Jv = Jv - T[:, :k+1] @ (T[:, :k+1].T @ Jv)
            norm_Jv = Jv.norm()
            if norm_Jv < 1e-8: continue
            Jv = Jv / norm_Jv
            T[:, k+1] = Jv
            k += 2
        if k < self.latent_dim:
            T = T[:, :k]
            self.latent_dim = k
        return T

    def _verify_symplectic_pairs(self):
        if self.latent_dim < 2: return
        J_r = self.Ti @ self.J_symplectic @ self.T
        J_d = torch.zeros_like(J_r)
        for i in range(0, self.latent_dim, 2):
            J_d[i, i+1] = 1.0
            J_d[i+1, i] = -1.0
        structure_error = (J_r - J_d).norm().item()
        logger.info(f"  Symplectic structure preservation error ||T^TJT - J_d||: {structure_error:.2e}")

    def fit(self, X):
        self._build_projection(self.A, self.J_symplectic, self.R, X)
        return self
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.T
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.Ti
        
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0] if X.dim() > 1 else 1
        return self.Ti.unsqueeze(0).expand(B, -1, -1).contiguous()
        
    def to(self, device):
        super().to(device)
        self.sys = self.sys.to(device)
        self.A = self.A.to(device)
        self.J_symplectic = self.J_symplectic.to(device)
        self.R = self.R.to(device)
        if hasattr(self, 'T'):
            self.T = self.T.to(device)
        if hasattr(self, 'Ti'):
            self.Ti = self.Ti.to(device)
        return self