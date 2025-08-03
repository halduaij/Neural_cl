from __future__ import annotations
import torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from neural_clbf.dimension_reduction.base import BaseReducer


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency with explicit Lyapunov energy bound."""

    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, λ: float = 0.7):
        super().__init__(latent_dim=2 * n_groups)
        self.sys, self.n_groups = sys, n_groups
        self._build(snaps, λ)

    def _build(self, X, λ):
        N = self.sys.n_machines
        M = torch.as_tensor(self.sys.M, dtype=X.dtype, device=X.device)

        # Debug prints
        print(f"[DEBUG LyapCoherency] X.shape = {X.shape}, N = {N}")
        print(f"[DEBUG LyapCoherency] sys.N_NODES = {self.sys.N_NODES}")
        
        # Extract angles and velocities from state vector
        # State is [θ_12, ..., θ_1n, ω_1, ..., ω_n] with dimension 2N-1
        delta_abs = self.sys.state_to_absolute_angles(X)  # Convert to absolute angles
        omega = X[:, self.sys.N_NODES - 1:]  # Extract angular velocities
        
        print(f"[DEBUG LyapCoherency] delta_abs.shape = {delta_abs.shape}, omega.shape = {omega.shape}")

        # Compute kinetic and potential energies
        kin = 0.5 * M * omega ** 2
        pot = self.sys.potential_energy_per_machine(delta_abs)

        E = kin + pot                                    # (T,N)
        
        print(f"[DEBUG LyapCoherency] E.shape = {E.shape}")

        # Compute correlation and gap matrices
        corr = torch.corrcoef(E.T).detach().cpu().numpy()
        gap = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        gap = (gap / gap.max()).detach().cpu().numpy()
        D = λ * (1 - np.abs(corr)) + (1 - λ) * gap

        # Hierarchical clustering
        Z = linkage(D[np.triu_indices(N, 1)], "average")
        labels = fcluster(Z, t=self.n_groups, criterion="maxclust") - 1
        labels = torch.as_tensor(labels, dtype=torch.long, device=X.device)

        # Build projection matrix P
        # Since state is [θ_12, ..., θ_1n, ω_1, ..., ω_n], we need dimension 2N-1
        state_dim = 2 * N - 1
        P = torch.zeros(state_dim, 2 * self.n_groups, dtype=X.dtype, device=X.device)
        
        # Map angles (first N-1 components)
        for g in range(self.n_groups):
            idx = torch.where(labels == g)[0]
            if len(idx) > 0:
                # For angles: we need to handle the fact that we have N-1 angle differences
                # but N machines. We'll map based on which machines are in the group.
                # Skip machine 0 (reference) for angle differences
                for i in idx[idx > 0]:  # Skip reference machine 0
                    P[i - 1, 2 * g] = 1 / len(idx)
                
                # For velocities: map all machines in the group
                for i in idx:
                    P[N - 1 + i, 2 * g + 1] = 1 / len(idx)

        self.register_buffer("P", P.detach())
        # Compute pseudo-inverse
        self.register_buffer("Pi", torch.linalg.pinv(P).detach())
        self.register_buffer("labels", labels)

        # Compute maximum energy deviation
        X_reconstructed = self.inverse(self.forward(X))
        ΔV = (self.sys.energy_function(X) - 
              self.sys.energy_function(X_reconstructed)).abs()
        self.register_buffer("deltaV_max", ΔV.detach().max().unsqueeze(0))
        
        print(f"[DEBUG LyapCoherency] P.shape = {P.shape}, Pi.shape = {self.Pi.shape}")
        print(f"[DEBUG LyapCoherency] deltaV_max = {self.deltaV_max.item()}")

    def fit(self, x):
        """No-op for this reducer"""
        return self
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space"""
        return x @ self.P
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space"""
        return z @ self.Pi
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). Analytical: J = P.T"""
        B = X.shape[0]
        # P.T has shape (d, n), expand to (B, d, n)
        return self.P.T.unsqueeze(0).expand(B, -1, -1).contiguous()
    def compute_gamma(self, V_min: float) -> float:
        """Compute the Lyapunov robustness margin gamma."""
        return float(self.deltaV_max.item() / V_min) if V_min > 0 else float('inf')
        
    def to(self, device):
        """Move reducer to device"""
        self.P = self.P.to(device)
        self.Pi = self.Pi.to(device)
        self.labels = self.labels.to(device)
        self.deltaV_max = self.deltaV_max.to(device)
        return self