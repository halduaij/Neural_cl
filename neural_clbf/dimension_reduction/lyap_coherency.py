from __future__ import annotations
import torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from .base import BaseReducer


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency with explicit Lyapunov energy bound."""

    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, λ: float = 0.7):
        super().__init__(latent_dim=2 * n_groups)
        self.sys, self.n_groups = sys, n_groups
        self._build(snaps, λ)

    def _build(self, X, λ):
        N = self.sys.n_machines
        M = torch.as_tensor(self.sys.M, dtype=X.dtype)

        kin = 0.5 * M * X[:, N:2*N] ** 2
        pot = self.sys.potential_energy_per_machine(X[:, :N])
        E = kin + pot                                    # (T,N)

        corr = torch.corrcoef(E.T).cpu().numpy()
        gap = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        gap = (gap / gap.max()).numpy()
        D = λ * (1 - np.abs(corr)) + (1 - λ) * gap

        Z = linkage(D[np.triu_indices(N, 1)], "average")
        labels = fcluster(Z, t=self.n_groups, criterion="maxclust") - 1
        labels = torch.as_tensor(labels, dtype=torch.long)

        P = torch.zeros(2*N, 2*self.n_groups, dtype=X.dtype)
        for g in range(self.n_groups):
            idx = torch.where(labels == g)[0]
            P[idx, 2*g] = 1/len(idx)
            P[idx+N, 2*g+1] = 1/len(idx)

        self.register_buffer("P", P)
        self.register_buffer("Pi", P.T)
        self.register_buffer("labels", labels)

        ΔV = (self.sys.energy_function(X) -
              self.sys.energy_function(self.inverse(self.forward(X)))).abs()
        self.register_buffer("deltaV_max", ΔV.max().unsqueeze(0))

    fit = lambda self, x: self
    forward = lambda self, x: x @ self.P
    inverse = lambda self, z: z @ self.Pi
    def jacobian(self, X):
        B = X.shape[0]
        return self.P.T.unsqueeze(0).expand(B, *self.P.T.shape)

    def gamma(self, V_min):
        return float(self.deltaV_max.item() / V_min)
