from __future__ import annotations
import torch, itertools
from torch import nn


class GPOpInfDynamics(nn.Module):
    """Quadratic operator inference latent ODE."""

    def __init__(self, d: int, m: int = 1):
        super().__init__()
        self.d = d
        self.m = m
        self.A = nn.Parameter(torch.zeros(d, d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(d, m), requires_grad=False)
        comb = list(itertools.combinations_with_replacement(range(d), 2))
        self.register_buffer("_comb", torch.tensor(comb, dtype=torch.long))
        self.H = nn.Parameter(torch.zeros(d, len(comb)), requires_grad=False)
        self.register_buffer("residual", torch.tensor(0.0))

    def _q(self, z):
        """Compute quadratic terms"""
        z1 = z[..., self._comb[:, 0]]
        z2 = z[..., self._comb[:, 1]]
        return z1 * z2

    def forward(self, z, u=None):
        """
        Compute latent dynamics dz/dt = Az + H*q(z) + Bu
        
        Args:
            z: (B, d) latent state
            u: (B, m) control input (optional)
        
        Returns:
            dz/dt: (B, d) latent dynamics
        """
        # Ensure z has batch dimension
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        # Linear term
        out = z @ self.A.T
        
        # Quadratic term
        out = out + self._q(z) @ self.H.T
        
        # Control term
        if u is not None:
            # Ensure u has correct shape
            if u.dim() == 1:
                u = u.unsqueeze(0)  # (m,) -> (1, m)
            if u.shape[0] == 1 and z.shape[0] > 1:
                u = u.expand(z.shape[0], -1)  # (1, m) -> (B, m)
            out = out + u @ self.B.T
            
        return out

    @torch.no_grad()
    def fit(self, Z, U, dZdt):
        """
        Fit dynamics using least squares.
        
        Args:
            Z: (N, d) latent states
            U: (N, m) control inputs
            dZdt: (N, d) latent derivatives
        """
        # Build data matrix
        Q = self._q(Z)  # (N, n_quad)
        
        # Stack features [Z, Q, U]
        if U.shape[1] > 0:
            Φ = torch.cat([Z, Q, U], dim=1)
        else:
            Φ = torch.cat([Z, Q], dim=1)
        
        # Least squares solve
        θ = torch.linalg.lstsq(Φ, dZdt).solution
        
        # Extract parameters
        d = self.d
        self.A.copy_(θ[:d].T)
        
        n_quad = Q.shape[1]
        self.H.copy_(θ[d:d+n_quad].T)
        
        if U.shape[1] > 0:
            self.B.copy_(θ[d+n_quad:].T)
        
        # Compute residual
        self.residual = (dZdt - Φ @ θ).norm(dim=1).max()
        
    def to(self, device):
        """Move module to device"""
        super().to(device)
        return self