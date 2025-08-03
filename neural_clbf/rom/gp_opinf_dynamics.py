from __future__ import annotations
import torch, itertools
from torch import nn


class GPOpInfDynamics(nn.Module):
    """Quadratic operator inference latent ODE."""

    def __init__(self, d: int, m: int = 1):
        super().__init__()
        self.d = d
        self.A = nn.Parameter(torch.zeros(d, d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(d, m), requires_grad=False)
        comb = list(itertools.combinations_with_replacement(range(d), 2))
        self.register_buffer("_comb", torch.tensor(comb))
        self.H = nn.Parameter(torch.zeros(d, len(comb)), requires_grad=False)

    def _q(self, z):
        z1, z2 = z[..., self._comb[:, 0]], z[..., self._comb[:, 1]]
        return z1 * z2

    def forward(self, z, u=None):
        out = z @ self.A.T + self._q(z) @ self.H.T
        if u is not None:
            out = out + u @ self.B.T
        return out

    @torch.no_grad()
    def fit(self, Z, U, dZdt):
        Q = self._q(Z)
        Φ = torch.cat([Z, Q, U], 1)
        θ = torch.linalg.lstsq(Φ, dZdt).solution
        d = self.d
        self.A.copy_(θ[:d].T)
        self.H.copy_(θ[d:-U.shape[1]].T)
        self.B.copy_(θ[-U.shape[1]:].T)
        self.register_buffer("residual",
                             (dZdt - Φ @ θ).norm(dim=1).max().unsqueeze(0))
