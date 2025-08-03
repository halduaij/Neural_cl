from __future__ import annotations
import torch
from .base import BaseReducer
from ..rom.gp_opinf_dynamics import GPOpInfDynamics


class OpInfReducer(BaseReducer):
    """PCA encoder + quadratic latent ODE, provides γ margin."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        self.dyn = GPOpInfDynamics(latent_dim, n_controls)

    def fit(self, X, Xdot, V_fn, V_min):
        self.μ.copy_(X.mean(0))
        _, _, Vt = torch.linalg.svd(X - self.μ, full_matrices=False)
        self.proj.copy_(Vt[: self.latent_dim].T)

        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        U = torch.zeros(Z.shape[0], 1, device=Z.device)
        self.dyn.fit(Z, U, dZdt)

        eps = float(self.dyn.residual.item())
        gradV = torch.autograd.grad(V_fn(X).sum(), X, create_graph=False)[0]
        L_V = gradV.norm(dim=1).max().item()
        self.gamma = eps * L_V / float(V_min)
        return self

    # BaseReducer interface
    forward = lambda self, x: (x - self.μ) @ self.proj
    inverse = lambda self, z: z @ self.proj.T + self.μ
    def jacobian(self, X):
        B = X.shape[0]
        return self.proj.T.unsqueeze(0).expand(B, *self.proj.T.shape)
