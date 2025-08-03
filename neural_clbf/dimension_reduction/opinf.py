from __future__ import annotations
import torch
from neural_clbf.dimension_reduction.base import BaseReducer
from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics


class OpInfReducer(BaseReducer):
    """PCA encoder + quadratic latent ODE, provides γ margin."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.n_controls = n_controls
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        self.dyn = GPOpInfDynamics(latent_dim, n_controls)

    def fit(self, X, Xdot, V_fn, V_min):
        # Store device
        device = X.device
        
        # Compute mean and PCA
        self.μ = X.mean(0)
        X_centered = X - self.μ
        _, _, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        self.proj = Vt[: self.latent_dim].T

        # Project to latent space
        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        
        # Create dummy control input with correct shape
        U = torch.zeros(Z.shape[0], self.n_controls, device=device)
        
        # Fit dynamics
        self.dyn.fit(Z, U, dZdt)

        # Compute gamma
        eps = float(self.dyn.residual.item())
        gradV = torch.autograd.grad(V_fn(X).sum(), X, create_graph=False)[0]
        L_V = gradV.norm(dim=1).max().item()
        self.gamma = eps * L_V / float(V_min) if V_min > 0 else float('inf')
        
        # Move everything to same device
        self.to(device)
        
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space"""
        return (x - self.μ) @ self.proj
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space"""
        return z @ self.proj.T + self.μ
        
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n)"""
        B = X.shape[0]
        # proj.T has shape (d, n), expand to (B, d, n)
        return self.proj.T.unsqueeze(0).expand(B, -1, -1)
        
    def to(self, device):
        """Move reducer to device"""
        self.μ = self.μ.to(device)
        self.proj = self.proj.to(device)
        self.dyn = self.dyn.to(device)
        return self