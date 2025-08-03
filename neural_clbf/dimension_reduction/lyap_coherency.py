from __future__ import annotations
import torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Assuming BaseReducer is imported correctly based on the project structure
try:
    from neural_clbf.dimension_reduction.base import BaseReducer
except ImportError:
    # Fallback for local execution
    # from base import BaseReducer
    # We assume the import works in the user's environment.
    pass 


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency with explicit Lyapunov energy bound."""

    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, λ: float = 0.7):
        # Initialize BaseReducer
        super().__init__(latent_dim=2 * n_groups)
        self.sys, self.n_groups = sys, n_groups
        
        # Ensure n_machines attribute exists
        if not hasattr(self.sys, 'n_machines'):
            if hasattr(self.sys, 'N_NODES'):
                self.sys.n_machines = self.sys.N_NODES
            else:
                raise AttributeError("System object must have 'n_machines' or 'N_NODES' attribute.")

        # Build the reducer components
        self._build(snaps, λ)

    def _build(self, X, λ):
        N = self.sys.n_machines
        M = torch.as_tensor(self.sys.M, dtype=X.dtype, device=X.device)

        # Debug prints
        # print(f"[DEBUG LyapCoherency] X.shape = {X.shape}, N = {N}")
        
        # Extract angles and velocities from state vector
        # Assumes state vector structure: [angles (N-1), frequencies (N)]
        delta_abs = self.sys.state_to_absolute_angles(X)
        omega = X[:, self.sys.N_NODES - 1:]
        
        # print(f"[DEBUG LyapCoherency] delta_abs.shape = {delta_abs.shape}, omega.shape = {omega.shape}")

        # Compute kinetic and potential energies
        kin = 0.5 * M * omega ** 2
        
        if hasattr(self.sys, 'potential_energy_per_machine'):
            pot = self.sys.potential_energy_per_machine(delta_abs)
        else:
            print("Warning: potential_energy_per_machine not found. Using kinetic energy only.")
            pot = torch.zeros_like(kin)

        E = kin + pot                                    # (T,N)
        
        # FIX 4: Robustness checks for Energy calculation
        if not torch.isfinite(E).all():
            print("Warning: Non-finite energy values detected in LCR. Clamping.")
            E = torch.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # print(f"[DEBUG LyapCoherency] E.shape = {E.shape}")

        # Compute correlation and gap matrices
        # FIX 4: Robust correlation coefficient calculation
        try:
            # Add small noise to prevent issues with constant signals (zero variance)
            E_noisy = E + 1e-9 * torch.randn_like(E)
            corr = torch.corrcoef(E_noisy.T).detach().cpu().numpy()
            # Handle NaNs if variance was still zero
            if np.isnan(corr).any():
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
        except Exception as e:
            print(f"Warning: Correlation computation failed: {e}. Using identity.")
            corr = np.identity(N)

        
        gap_tensor = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        
        # FIX 4: Robust gap normalization
        max_gap = gap_tensor.max()
        if max_gap < 1e-9:
            # If the maximum gap is near zero, all machines are coherent (gap is zero)
            gap = np.zeros((N,N))
        else:
            gap = (gap_tensor / max_gap).detach().cpu().numpy()
        
        D = λ * (1 - np.abs(corr)) + (1 - λ) * gap

        # Hierarchical clustering
        # Ensure D is finite and symmetric before linkage
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        D_condensed = D[np.triu_indices(N, 1)]
        
        if not np.isfinite(D_condensed).all():
             D_condensed = np.nan_to_num(D_condensed, nan=0.0, posinf=1.0, neginf=0.0)

        try:
            Z = linkage(D_condensed, "average")
            labels = fcluster(Z, t=self.n_groups, criterion="maxclust") - 1
        except Exception as e:
            print(f"Clustering failed: {e}. Assigning fallback groups.")
            labels = np.arange(N) % self.n_groups # Fallback grouping

        labels = torch.as_tensor(labels, dtype=torch.long, device=X.device)

        # Build projection matrix P
        state_dim = 2 * N - 1
        P = torch.zeros(state_dim, 2 * self.n_groups, dtype=X.dtype, device=X.device)
        
        # Map angles (first N-1 components) and velocities
        for g in range(self.n_groups):
            idx = torch.where(labels == g)[0]
            if len(idx) > 0:
                norm_factor = float(len(idx))
                # For angles (skip reference machine 0, indices 0 to N-2)
                for i in idx[idx > 0]:
                    P[i - 1, 2 * g] = 1 / norm_factor
                
                # For velocities (indices N-1 to 2N-2)
                for i in idx:
                    P[N - 1 + i, 2 * g + 1] = 1 / norm_factor

        # Use register_buffer for P and Pi if they don't exist, otherwise update .data
        if not hasattr(self, "P"):
            self.register_buffer("P", P.detach())
            self.register_buffer("Pi", torch.linalg.pinv(P).detach())
        else:
            self.P.data = P.detach()
            self.Pi.data = torch.linalg.pinv(P).detach()

        self.register_buffer("labels", labels)

        # print(f"[DEBUG LyapCoherency] P.shape = {P.shape}, Pi.shape = {self.Pi.shape}")

        # Compute maximum energy deviation
        # This relies on forward/inverse working correctly.
        # When called from ImprovedLCR, this uses the overridden methods.
        X_reconstructed = self.inverse(self.forward(X)) 
        
        try:
            if hasattr(self.sys, 'energy_function'):
                E_orig = self.sys.energy_function(X)
                E_recon = self.sys.energy_function(X_reconstructed)
                ΔV = (E_orig - E_recon).abs()
                # Filter out potential non-finite values
                ΔV_finite = ΔV[torch.isfinite(ΔV)]
                if ΔV_finite.numel() > 0:
                    max_dev = ΔV_finite.detach().max()
                else:
                    max_dev = torch.tensor(0.0, device=X.device)
            else:
                max_dev = torch.tensor(0.0, device=X.device)

        except Exception as e:
            print(f"Energy deviation calculation failed: {e}. Setting deltaV_max to 0.")
            max_dev = torch.tensor(0.0, device=X.device)

        if not hasattr(self, "deltaV_max"):
            self.register_buffer("deltaV_max", max_dev)
        else:
            self.deltaV_max.data = max_dev

        # print(f"[DEBUG LyapCoherency] deltaV_max = {self.deltaV_max.item()}")


    # -------------- BaseReducer API ------------------------------------
    def fit(self, x):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space: z = x @ P"""
        # Note: If ImprovedLCR overrides this (which it does), that version will be called during _build.
        return x @ self.P
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space: x = z @ Pi"""
        # Note: If ImprovedLCR overrides this (which it does), that version will be called during _build.
        return z @ self.Pi
    
    # Analytical Jacobian (as implemented in previous fixes)
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). Analytical: J = P.T"""
        B = X.shape[0] if X.dim() > 1 else 1
        # P.T has shape (d, n), expand to (B, d, n)
        J = self.P.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()

    def compute_gamma(self, V_min: float) -> float:
        """γ = sqrt(ΔV_max / V_min)"""
        if V_min < 1e-9:
            return float('inf')
        # Use sqrt as per the theoretical definition of gamma for LCR
        return torch.sqrt(self.deltaV_max / V_min).item()

    def to(self, device):
        super().to(device)
        # Ensure buffers are moved
        if hasattr(self, 'P'): self.P = self.P.to(device)
        if hasattr(self, 'Pi'): self.Pi = self.Pi.to(device)
        if hasattr(self, 'labels'): self.labels = self.labels.to(device)
        if hasattr(self, 'deltaV_max'): self.deltaV_max = self.deltaV_max.to(device)
        return self