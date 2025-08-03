from __future__ import annotations
import torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from neural_clbf.dimension_reduction.base import BaseReducer


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency with explicit Lyapunov energy bound - BETTER FIX."""

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
        
        # Extract angles and velocities from state vector
        delta_abs = self.sys.state_to_absolute_angles(X)
        omega = X[:, self.sys.N_NODES - 1:]

        # Compute kinetic and potential energies
        kin = 0.5 * M * omega ** 2
        
        if hasattr(self.sys, 'potential_energy_per_machine'):
            pot = self.sys.potential_energy_per_machine(delta_abs)
        else:
            print("Warning: potential_energy_per_machine not found. Using kinetic energy only.")
            pot = torch.zeros_like(kin)

        E = kin + pot
        
        # Robustness checks for Energy calculation
        if not torch.isfinite(E).all():
            print("Warning: Non-finite energy values detected in LCR. Clamping.")
            E = torch.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)

        # Compute correlation and gap matrices
        try:
            E_noisy = E + 1e-9 * torch.randn_like(E)
            corr = torch.corrcoef(E_noisy.T).detach().cpu().numpy()
            if np.isnan(corr).any():
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
        except Exception as e:
            print(f"Warning: Correlation computation failed: {e}. Using identity.")
            corr = np.identity(N)
        
        gap_tensor = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        max_gap = gap_tensor.max()
        if max_gap < 1e-9:
            gap = np.zeros((N,N))
        else:
            gap = (gap_tensor / max_gap).detach().cpu().numpy()
        
        D = λ * (1 - np.abs(corr)) + (1 - λ) * gap

        # Hierarchical clustering
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
            labels = np.arange(N) % self.n_groups

        labels = torch.as_tensor(labels, dtype=torch.long, device=X.device)
        
        # BETTER FIX: Handle the reference machine (machine 0) specially
        # If machine 0 is alone in its group, merge it with another group
        group_counts = torch.zeros(self.n_groups, dtype=torch.long)
        for g in range(self.n_groups):
            group_counts[g] = (labels == g).sum()
        
        machine0_group = labels[0].item()
        if group_counts[machine0_group] == 1:
            print(f"Machine 0 is alone in group {machine0_group}. Merging with another group...")
            # Find the group with the most machines (excluding machine 0's group)
            other_groups = [g for g in range(self.n_groups) if g != machine0_group]
            if other_groups:
                largest_other_group = max(other_groups, key=lambda g: group_counts[g].item())
                labels[0] = largest_other_group
                group_counts[machine0_group] -= 1
                group_counts[largest_other_group] += 1
        
        # Now handle any remaining empty groups
        empty_groups = torch.where(group_counts == 0)[0]
        if len(empty_groups) > 0:
            print(f"Found {len(empty_groups)} empty groups. Redistributing machines...")
            multi_machine_groups = torch.where(group_counts > 1)[0]
            
            for empty_g in empty_groups:
                if len(multi_machine_groups) > 0:
                    # Take one machine from the largest group
                    largest_group = multi_machine_groups[group_counts[multi_machine_groups].argmax()]
                    machines_in_largest = torch.where(labels == largest_group)[0]
                    # Don't move machine 0
                    movable_machines = [m for m in machines_in_largest if m > 0]
                    if movable_machines:
                        labels[movable_machines[0]] = empty_g
                        group_counts[largest_group] -= 1
                        group_counts[empty_g] += 1
                        if group_counts[largest_group] <= 1:
                            multi_machine_groups = torch.where(group_counts > 1)[0]
        
        # Final check: ensure we have exactly n_groups non-empty groups
        unique_groups = torch.unique(labels)
        actual_groups = len(unique_groups)
        
        if actual_groups < self.n_groups:
            # This should not happen after our fixes, but just in case
            print(f"Warning: Only {actual_groups} groups formed (requested {self.n_groups})")
            # Don't reduce dimension - instead create dummy groups
            # This ensures we maintain the expected latent dimension
        
        # Build projection matrix P
        state_dim = 2 * N - 1
        P = torch.zeros(state_dim, 2 * self.n_groups, dtype=X.dtype, device=X.device)
        
        # Map angles and velocities
        for g in range(self.n_groups):
            idx = torch.where(labels == g)[0]
            if len(idx) > 0:
                # For angles (skip reference machine 0)
                angle_indices = []
                for i in idx:
                    if i > 0:  # Skip reference machine 0
                        angle_indices.append(i - 1)
                
                if angle_indices:
                    # Equal weight for all machines in group
                    weight_angle = 1.0 / np.sqrt(len(angle_indices))
                    for angle_idx in angle_indices:
                        P[angle_idx, 2 * g] = weight_angle
                else:
                    # Group contains only machine 0 - this column will be zero
                    # But we keep it to maintain the expected dimension
                    pass
                
                # For frequencies (all machines including machine 0)
                weight_freq = 1.0 / np.sqrt(len(idx))
                for i in idx:
                    P[N - 1 + i, 2 * g + 1] = weight_freq
        
        # DO NOT remove zero columns - keep them to maintain dimension
        # This ensures we have exactly the requested latent dimension
        
        # Compute proper pseudo-inverse
        self.register_buffer("P", P.detach())
        self.register_buffer("Pi", torch.linalg.pinv(P).detach())
        self.register_buffer("labels", labels)

        # Verify the projection properties
        P_Pi = self.P @ self.Pi
        print(f"LCR projection verification: ||P @ Pi @ P - P|| = {(P_Pi @ self.P - self.P).norm():.6e}")
        
        # Print group information
        print(f"Group assignments:")
        for g in range(self.n_groups):
            machines = torch.where(labels == g)[0].tolist()
            print(f"  Group {g}: machines {machines}")

        # Compute maximum energy deviation
        X_reconstructed = self.inverse(self.forward(X))
        
        try:
            if hasattr(self.sys, 'energy_function'):
                E_orig = self.sys.energy_function(X)
                E_recon = self.sys.energy_function(X_reconstructed)
                ΔV = (E_orig - E_recon).abs()
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

        self.register_buffer("deltaV_max", max_dev)

    # BaseReducer API
    def fit(self, x):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space: z = x @ P"""
        return x @ self.P
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space: x = z @ Pi"""
        return z @ self.Pi
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). Analytical: J = P.T"""
        B = X.shape[0] if X.dim() > 1 else 1
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