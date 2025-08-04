from __future__ import annotations
import torch, numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from neural_clbf.dimension_reduction.base import BaseReducer


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency with explicit Lyapunov energy bound - FIXED VERSION."""

    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, λ: float = 0.7):
        # Initialize BaseReducer with proper dimension
        # Account for the fact that actual dimension might be less than 2*n_groups
        max_latent_dim = 2 * sys.n_machines - 1  # Maximum possible dimension
        requested_latent_dim = 2 * n_groups
        actual_latent_dim = min(requested_latent_dim, max_latent_dim)
        
        super().__init__(latent_dim=actual_latent_dim)
        self.sys = sys
        self.n_groups = n_groups
        self.requested_groups = n_groups
        
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
        
        print(f"\nBuilding LyapCoherencyReducer:")
        print(f"  Machines: {N}, Requested groups: {self.n_groups}")
        
        # Extract angles and velocities from state vector
        try:
            delta_abs = self.sys.state_to_absolute_angles(X)
        except Exception as e:
            print(f"  Warning: state_to_absolute_angles failed: {e}")
            # Fallback: assume first N-1 are relative angles
            delta_abs = torch.zeros(X.shape[0], N, device=X.device)
            if N > 1:
                delta_abs[:, 1:] = X[:, :N-1]
        
        omega = X[:, self.sys.N_NODES - 1:]

        # Compute kinetic and potential energies
        kin = 0.5 * M * omega ** 2
        
        if hasattr(self.sys, 'potential_energy_per_machine'):
            try:
                pot = self.sys.potential_energy_per_machine(delta_abs)
            except Exception as e:
                print(f"  Warning: potential_energy_per_machine failed: {e}. Using kinetic only.")
                pot = torch.zeros_like(kin)
        else:
            print("  Warning: potential_energy_per_machine not found. Using kinetic energy only.")
            pot = torch.zeros_like(kin)

        E = kin + pot
        
        # Robustness checks for Energy calculation
        if not torch.isfinite(E).all():
            print("  Warning: Non-finite energy values detected. Clamping.")
            E = torch.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add small noise for numerical stability
        E_noisy = E + 1e-9 * torch.randn_like(E)

        # Compute correlation and gap matrices
        try:
            corr = torch.corrcoef(E_noisy.T).detach().cpu().numpy()
            if np.isnan(corr).any() or np.isinf(corr).any():
                print("  Warning: Invalid correlation matrix. Using identity.")
                corr = np.eye(N)
                np.fill_diagonal(corr, 1.0)
        except Exception as e:
            print(f"  Warning: Correlation computation failed: {e}. Using identity.")
            corr = np.eye(N)
        
        # Compute gap matrix
        gap_tensor = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        max_gap = gap_tensor.max()
        if max_gap > 1e-9:
            gap = (gap_tensor / max_gap).detach().cpu().numpy()
        else:
            gap = np.zeros((N, N))
        
        # Combined distance matrix
        D = λ * (1 - np.abs(corr)) + (1 - λ) * gap

        # Ensure symmetry and zero diagonal
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        
        # Hierarchical clustering with robustness
        D_condensed = D[np.triu_indices(N, 1)]
        
        if not np.isfinite(D_condensed).all():
            D_condensed = np.nan_to_num(D_condensed, nan=0.0, posinf=1.0, neginf=0.0)

        try:
            # Only cluster if there's actual variation
            if D_condensed.max() > 1e-10:
                Z = linkage(D_condensed, "average")
                labels = fcluster(Z, t=self.n_groups, criterion="maxclust") - 1
            else:
                print("  Warning: All machines perfectly coherent. Using cyclic grouping.")
                labels = np.arange(N) % self.n_groups
        except Exception as e:
            print(f"  Clustering failed: {e}. Assigning fallback groups.")
            labels = np.arange(N) % self.n_groups

        labels = torch.as_tensor(labels, dtype=torch.long, device=X.device)
        
        # Fix reference machine grouping and empty groups
        labels = self._fix_grouping_issues(labels, N)
        
        # Build projection matrix
        self._build_projection_matrix(labels, N, X.device, X.dtype)
        
        # Compute maximum energy deviation for gamma
        self._compute_energy_deviation(X)
        
        # Print final group assignments
        print("  Final group assignments:")
        for g in range(self.actual_groups):
            machines = torch.where(self.labels == g)[0].tolist()
            print(f"    Group {g}: machines {machines}")

    def _fix_grouping_issues(self, labels, N):
        """Fix reference machine isolation and empty groups."""
        labels = labels.clone()
        
        # Count machines per group
        group_counts = torch.zeros(self.n_groups, dtype=torch.long)
        for g in range(self.n_groups):
            group_counts[g] = (labels == g).sum()
        
        # Fix 1: Handle reference machine (machine 0) if alone
        machine0_group = labels[0].item()
        if group_counts[machine0_group] == 1:
            print(f"  Fixing: Machine 0 alone in group {machine0_group}")
            # Find the group with the most machines
            other_groups = [g for g in range(self.n_groups) if g != machine0_group]
            if other_groups:
                largest_other_group = max(other_groups, key=lambda g: group_counts[g].item())
                labels[0] = largest_other_group
                group_counts[machine0_group] -= 1
                group_counts[largest_other_group] += 1
        
        # Fix 2: Redistribute empty groups
        empty_groups = torch.where(group_counts == 0)[0]
        if len(empty_groups) > 0:
            print(f"  Found {len(empty_groups)} empty groups. Redistributing...")
            
            for empty_g in empty_groups:
                # Find groups with multiple machines
                multi_machine_groups = torch.where(group_counts > 1)[0]
                
                if len(multi_machine_groups) > 0:
                    # Take one machine from the largest group
                    largest_group = multi_machine_groups[group_counts[multi_machine_groups].argmax()]
                    machines_in_largest = torch.where(labels == largest_group)[0]
                    
                    # Don't move machine 0
                    movable_machines = [m.item() for m in machines_in_largest if m > 0]
                    
                    if movable_machines:
                        labels[movable_machines[0]] = empty_g
                        group_counts[largest_group] -= 1
                        group_counts[empty_g] += 1
        
        # Recompute actual number of groups
        unique_groups = torch.unique(labels)
        self.actual_groups = len(unique_groups)
        
        # Relabel to ensure continuous indices
        new_labels = torch.zeros_like(labels)
        for new_idx, old_idx in enumerate(unique_groups):
            new_labels[labels == old_idx] = new_idx
        
        return new_labels

    def _build_projection_matrix(self, labels, N, device, dtype):
        """Build projection matrix ensuring full rank."""
        state_dim = 2 * N - 1  # N-1 angles + N frequencies
        
        # Actual number of groups after fixing
        n_actual_groups = labels.max().item() + 1
        
        # Compute actual latent dimension
        # Each group contributes at most 2 dimensions (angle + frequency)
        # But groups with only machine 0 contribute only 1 (frequency only)
        latent_dim = 0
        for g in range(n_actual_groups):
            machines_in_group = torch.where(labels == g)[0]
            if len(machines_in_group) > 0:
                # Check if group has non-reference machines
                has_angle_machines = any(m > 0 for m in machines_in_group)
                if has_angle_machines:
                    latent_dim += 2  # Both angle and frequency
                else:
                    latent_dim += 1  # Only frequency
        
        # Ensure we don't exceed state dimension
        latent_dim = min(latent_dim, state_dim)
        self.latent_dim = latent_dim
        self.actual_groups = n_actual_groups
        
        print(f"  Actual groups: {n_actual_groups}, Latent dimension: {latent_dim}")
        
        # Build projection matrix P
        P = torch.zeros(state_dim, latent_dim, dtype=dtype, device=device)
        
        col_idx = 0
        for g in range(n_actual_groups):
            if col_idx >= latent_dim:
                break
                
            idx = torch.where(labels == g)[0]
            if len(idx) == 0:
                continue
            
            # Angle component (skip if only contains machine 0)
            angle_machines = [i.item() for i in idx if i > 0]
            
            if angle_machines and col_idx < latent_dim:
                # Equal weight for all non-reference machines in group
                weight_angle = 1.0 / np.sqrt(len(angle_machines))
                for i in angle_machines:
                    P[i - 1, col_idx] = weight_angle
                col_idx += 1
            
            # Frequency component (all machines including machine 0)
            if col_idx < latent_dim:
                weight_freq = 1.0 / np.sqrt(len(idx))
                for i in idx:
                    P[N - 1 + i, col_idx] = weight_freq
                col_idx += 1
        
        # Verify and fix rank if needed
        rank = torch.linalg.matrix_rank(P).item()
        if rank < latent_dim:
            print(f"  Warning: Initial projection rank {rank} < {latent_dim}. Fixing...")
            
            # Use QR decomposition to get orthogonal basis
            Q, R = torch.linalg.qr(P)
            P = Q[:, :latent_dim]
            
            # Recheck rank
            rank = torch.linalg.matrix_rank(P).item()
            print(f"  Fixed projection rank: {rank}")
        
        # Compute proper pseudo-inverse with regularization
        reg = 1e-10
        try:
            # Method 1: Direct pseudoinverse
            Pi = torch.linalg.pinv(P, rcond=1e-8)
        except:
            # Method 2: Regularized inverse
            Pi = P.T @ torch.linalg.inv(P @ P.T + reg * torch.eye(state_dim, device=device))
        
        # Store matrices
        self.register_buffer("P", P.detach())
        self.register_buffer("Pi", Pi.detach())
        self.register_buffer("labels", labels.detach())
        
        # Verify projection properties
        P_Pi_P = self.P @ self.Pi @ self.P
        proj_error = (P_Pi_P - self.P).norm()
        print(f"  Projection verification: ||P @ Pi @ P - P|| = {proj_error:.6e}")

    def _compute_energy_deviation(self, X):
        """Compute maximum energy deviation for gamma calculation."""
        try:
            # Test reconstruction
            X_reconstructed = self.inverse(self.forward(X))
            
            if hasattr(self.sys, 'energy_function'):
                E_orig = self.sys.energy_function(X)
                E_recon = self.sys.energy_function(X_reconstructed)
                
                # Handle shape mismatches
                if E_orig.dim() != E_recon.dim():
                    E_orig = E_orig.reshape(-1)
                    E_recon = E_recon.reshape(-1)
                
                ΔV = (E_orig - E_recon).abs()
                ΔV_finite = ΔV[torch.isfinite(ΔV)]
                
                if ΔV_finite.numel() > 0:
                    max_dev = ΔV_finite.detach().max()
                else:
                    max_dev = torch.tensor(0.0, device=X.device)
            else:
                # Use state reconstruction error as proxy
                state_errors = (X - X_reconstructed).norm(dim=1)
                max_dev = state_errors.max()
        
        except Exception as e:
            print(f"  Energy deviation calculation failed: {e}. Using default.")
            max_dev = torch.tensor(0.1, device=X.device)
        
        self.register_buffer("deltaV_max", max_dev)
        print(f"  Max energy deviation: {max_dev.item():.6e}")

    # BaseReducer API
    def fit(self, x):
        """No fitting needed for coherency reducer."""
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
        """γ = sqrt(ΔV_max / V_min) for theoretical consistency"""
        if V_min < 1e-9:
            return float('inf')
        # Use sqrt as per the theoretical definition of gamma for LCR
        gamma_val = torch.sqrt(self.deltaV_max / V_min).item()
        
        # Cap at reasonable value
        return min(gamma_val, 100.0)

    def to(self, device):
        """Move all tensors to device."""
        super().to(device)
        # Ensure all buffers are moved
        if hasattr(self, 'P'): 
            self.P = self.P.to(device)
        if hasattr(self, 'Pi'): 
            self.Pi = self.Pi.to(device)
        if hasattr(self, 'labels'): 
            self.labels = self.labels.to(device)
        if hasattr(self, 'deltaV_max'): 
            self.deltaV_max = self.deltaV_max.to(device)
        return self