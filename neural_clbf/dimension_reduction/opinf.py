from __future__ import annotations
import torch
import numpy as np
from neural_clbf.dimension_reduction.base import BaseReducer
# Assuming the import path from the context provided
try:
    from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
except ImportError:
    # Fallback for direct execution if needed
    from gp_opinf_dynamics import GPOpInfDynamics


class OpInfReducer(BaseReducer):
    """PCA encoder + latent ODE, provides γ margin."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.n_controls = n_controls
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        self.dyn = None
        self.sys = None # Placeholder for system reference

    def fit(self, X, Xdot, V_fn, V_min):
            device = X.device
            
            # 1. PCA (Robust SVD and Energy-based dimension selection)
            self.μ = X.mean(0)
            X_centered = X - self.μ
            
            try:
                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
                
                # Determine effective rank (99.9% energy)
                total_energy = (S ** 2).sum()
                if total_energy > 1e-9:
                    cumsum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                    n_effective = (cumsum_energy < 0.999).sum().item() + 1
                else:
                    n_effective = 1
                
                n_keep = min(self.latent_dim, n_effective, X.shape[1])
                
                if n_keep < self.latent_dim:
                    print(f"OpInf: Reducing latent dim from {self.latent_dim} to {n_keep} (99.9% energy)")
                    self.latent_dim = n_keep

                self.proj = Vt[:self.latent_dim].T.contiguous()
                
            except Exception as e:
                print(f"OpInf SVD failed: {e}. Using QR fallback.")
                try:
                    Q, _ = torch.linalg.qr(X_centered.T)
                    self.proj = Q[:, :self.latent_dim]
                except:
                    self.proj = torch.eye(self.full_dim, self.latent_dim, device=device)
            
            # 2. Project Data
            Z = self.forward(X)
            dZdt = Xdot @ self.proj
            
            # 3. Fit Dynamics
            actual_n_controls = getattr(self.sys, 'n_controls', self.n_controls) if self.sys else self.n_controls

            self.dyn = GPOpInfDynamics(self.latent_dim, actual_n_controls).to(device)
            
            # FIX: Set strong regularization (crucial for stability)
            self.dyn.reg = torch.tensor(0.1, device=device)

            U = torch.zeros(Z.shape[0], actual_n_controls, device=device)
            
            # This call uses the aggressively stabilized fitting procedure
            self.dyn.fit(Z, U, dZdt)
            
            # 4. Compute Gamma
            self.compute_gamma(X, V_fn, V_min)

            self.to(device)
            return self

    def compute_gamma(self, X, V_fn, V_min):
        """Computes the robustness margin gamma robustly."""
        try:
            eps = float(self.dyn.residual.item())
            
            # Estimate L_V (Lipschitz constant of V)
            n_samples = min(X.shape[0], 1000)
            # FIX: Ensure gradients can be computed for V_fn
            X_subset = X[:n_samples].clone().detach().requires_grad_(True)
            
            if not callable(V_fn):
                 L_V = 10.0
            else:
                V_vals = V_fn(X_subset)
                # Check if V_fn is differentiable
                if V_vals.ndim == 0 or not V_vals.requires_grad:
                     L_V = 10.0
                else:
                    gradV = torch.autograd.grad(V_vals.sum(), X_subset, create_graph=False)[0]
                    gradV_norms = gradV.norm(dim=1)
                    finite_norms = gradV_norms[torch.isfinite(gradV_norms)]
                    if finite_norms.numel() > 0:
                        L_V = torch.quantile(finite_norms, 0.95).item()
                        L_V = max(1e-3, min(L_V, 100.0))
                    else:
                        L_V = 10.0

            V_min_safe = max(V_min, 1e-4)
            self.gamma = eps * L_V / V_min_safe
            
            if self.gamma > 50.0 or not np.isfinite(self.gamma):
                self.gamma = 50.0

        except Exception as e:
            print(f"Gamma computation failed: {e}. Setting default gamma=10.0.")
            self.gamma = 10.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return (x - self.μ) @ self.proj
        return (x - self.μ) @ self.proj
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
             return z @ self.proj.T + self.μ
        return z @ self.proj.T + self.μ
        
    # FIX: Implement Analytical Jacobian to avoid Autograd error.
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). J = P.T"""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.proj.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
        
    def to(self, device):
        super().to(device)
        self.μ = self.μ.to(device)
        self.proj = self.proj.to(device)
        if self.dyn is not None:
            self.dyn = self.dyn.to(device)
        return self