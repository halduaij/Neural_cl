from __future__ import annotations
import torch, numpy as np
import warnings
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from neural_clbf.dimension_reduction.base import BaseReducer

logger = logging.getLogger(__name__)


class LyapCoherencyReducer(BaseReducer):
    """Dynamic coherency reducer with robust clustering and orthonormal projection - FIXED VERSION."""

    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, λ: float = 0.7, 
                 strict_dim: bool = False):
        # Initialize BaseReducer with proper dimension
        # Account for the fact that actual dimension might be less than 2*n_groups
        max_latent_dim = 2 * sys.n_machines - 1  # Maximum possible dimension
        requested_latent_dim = 2 * n_groups
        actual_latent_dim = min(requested_latent_dim, max_latent_dim)
        
        super().__init__(latent_dim=actual_latent_dim)
        self.sys = sys
        self.n_groups = n_groups
        self.requested_groups = n_groups
        self.strict_dim = strict_dim
        
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
        
        logger.info(f"\nBuilding LyapCoherencyReducer:")
        logger.info(f"  Machines: {N}, Requested groups: {self.n_groups}")
        
        # Extract angles and velocities from state vector
        try:
            delta_abs = self.sys.state_to_absolute_angles(X)
        except Exception as e:
            logger.warning(f"  state_to_absolute_angles failed: {e}")
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
                logger.warning(f"  potential_energy_per_machine failed: {e}. Using kinetic only.")
                pot = torch.zeros_like(kin)
        else:
            logger.warning("  potential_energy_per_machine not found. Using kinetic energy only.")
            pot = torch.zeros_like(kin)

        E = kin + pot
        
        # Robustness checks for Energy calculation
        if not torch.isfinite(E).all():
            logger.warning("  Non-finite energy values detected. Clamping.")
            E = torch.nan_to_num(E, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add small noise for numerical stability
        E_noisy = E + 1e-9 * torch.randn_like(E)

        # Compute correlation and gap matrices
        try:
            corr = torch.corrcoef(E_noisy.T).detach().cpu().numpy()
            if np.isnan(corr).any() or np.isinf(corr).any():
                logger.warning("  Invalid correlation matrix. Using identity.")
                corr = np.eye(N)
                np.fill_diagonal(corr, 1.0)
        except Exception as e:
            logger.warning(f"  Correlation computation failed: {e}. Using identity.")
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
                logger.warning("  All machines perfectly coherent. Using cyclic grouping.")
                labels = np.arange(N) % self.n_groups
        except Exception as e:
            logger.error(f"  Clustering failed: {e}. Assigning fallback groups.")
            labels = np.arange(N) % self.n_groups

        labels = torch.as_tensor(labels, dtype=torch.long, device=X.device)
        
        # Fix reference machine grouping and empty groups
        labels = self._fix_grouping_issues(labels, N)
        
        # Build projection matrix
        self._build_projection_matrix(labels, N, X.device, X.dtype)
        
        # Check strict dimension requirement
        requested_dim = 2 * self.n_groups
        if self.strict_dim and self.latent_dim != requested_dim:
            raise ValueError(f"Could not achieve requested dimension {requested_dim}, got {self.latent_dim}")
        elif self.latent_dim != requested_dim:
            warnings.warn(f"Requested dimension {requested_dim} but achieved {self.latent_dim}")
        
        # Compute maximum energy deviation for gamma
        self._compute_energy_deviation(X)
        
        # Build reduced-order dynamics
        self._build_reduced_dynamics()
        
        # Print final group assignments
        logger.info("  Final group assignments:")
        for g in range(self.actual_groups):
            machines = torch.where(self.labels == g)[0].tolist()
            logger.info(f"    Group {g}: machines {machines}")

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
            logger.info(f"  Fixing: Machine 0 alone in group {machine0_group}")
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
            logger.info(f"  Found {len(empty_groups)} empty groups. Redistributing...")
            
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
        """Build orthonormal projection matrix P ensuring full rank - FIXED VERSION."""
        state_dim = 2 * N - 1  # N-1 angles + N frequencies
        n_actual_groups = labels.max().item() + 1
        
        # Determine the actual latent dimension based on group composition
        latent_dim = 0
        group_has_angle = torch.zeros(n_actual_groups, dtype=torch.bool)
        
        for g in range(n_actual_groups):
            machines_in_group = torch.where(labels == g)[0]
            if len(machines_in_group) > 0:
                # Angle state exists if any machine i > 0 is present (relative angles theta_1i)
                if any(m > 0 for m in machines_in_group):
                    latent_dim += 1
                    group_has_angle[g] = True
                # Frequency state always exists for non-empty group
                latent_dim += 1
        
        latent_dim = min(latent_dim, state_dim)
        self.latent_dim = latent_dim
        self.actual_groups = n_actual_groups
        
        logger.info(f"  Actual groups: {n_actual_groups}, Latent dimension: {latent_dim}")
        
        # Build projection matrix P (n x d)
        P = torch.zeros(state_dim, latent_dim, dtype=dtype, device=device)
        
        col_idx = 0
        for g in range(n_actual_groups):
            if col_idx >= latent_dim:
                break
                
            idx = torch.where(labels == g)[0]
            if len(idx) == 0:
                continue
            
            # Get inertias for this group
            M_group = self.sys.M[idx]
            M_total = M_group.sum()
            
            # Angle component - FIX: Use inertia weighting
            if group_has_angle[g]:
                angle_machines_idx = [i.item() for i in idx if i > 0]
                for i in angle_machines_idx:
                    # FIX: Weight by sqrt(M_i / M_group)
                    weight_angle = torch.sqrt(self.sys.M[i] / M_total).item()
                    P[i - 1, col_idx] = weight_angle  # Angle state index is i-1
                col_idx += 1
            
            # Frequency component - FIX: Use inertia weighting
            if col_idx < latent_dim:
                for i in idx:
                    # FIX: Weight by sqrt(M_i / M_group)
                    weight_freq = torch.sqrt(self.sys.M[i] / M_total).item()
                    P[N - 1 + i, col_idx] = weight_freq  # Frequency state index is N-1+i
                col_idx += 1
        
        # Ensure Orthogonality
        # Verify orthogonality (should hold by construction due to disjoint groups and components)
        ortho_error = (P.T @ P - torch.eye(latent_dim, device=device)).norm()
        
        if ortho_error > 1e-6:
            # Enforce via QR if construction failed or numerical issues arose
            logger.warning(f"  P not orthogonal (error: {ortho_error:.2e}). Enforcing via QR.")
            P, _ = torch.linalg.qr(P)

        # Pseudo-inverse is the transpose for orthonormal P
        Pi = P.T
        
        # Store matrices
        self.register_buffer("P", P.detach())
        self.register_buffer("Pi", Pi.detach())
        self.register_buffer("labels", labels.detach())
        
        # Verification
        proj_error = (self.P @ self.Pi @ self.P - self.P).norm()
        logger.info(f"  Projection verification: ||P @ Pi @ P - P|| = {proj_error:.6e}")

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
            logger.error(f"  Energy deviation calculation failed: {e}. Using default.")
            max_dev = torch.tensor(0.1, device=X.device)
        
        self.register_buffer("deltaV_max", max_dev)
        logger.info(f"  Max energy deviation: {max_dev.item():.6e}")

    def _build_reduced_dynamics(self):
        """Build aggregated swing equation dynamics for coherent groups - FIXED VERSION."""
        logger.info("  Building reduced-order dynamics...")
        
        device = self.P.device
        
        # Aggregate parameters for each group
        M_groups = []
        D_groups = []
        P_groups = []
        
        for g in range(self.actual_groups):
            machines_in_group = torch.where(self.labels == g)[0]
            if len(machines_in_group) > 0:
                # Sum inertias, damping for the group
                M_groups.append(self.sys.M[machines_in_group].sum())
                D_groups.append(self.sys.D[machines_in_group].sum())
                # FIX: Use negative mechanical power for correct sign convention
                P_groups.append(-self.sys.P_mechanical[machines_in_group].sum())
        
        self.M_eq = torch.stack(M_groups).to(device)
        self.D_eq = torch.stack(D_groups).to(device)
        self.P_eq = torch.stack(P_groups).to(device)
        
        # FIX: Build weighted coupling matrix K_red
        B_full = self.sys.B_matrix.to(device)
        
        # Build aggregated coupling matrix with proper weighting
        # CRITICAL FIX: Count actual angle groups properly
        angle_to_group = {}
        angle_group_idx = 0
        
        for g in range(self.actual_groups):
            machines = torch.where(self.labels == g)[0]
            if any(m > 0 for m in machines):  # Has angle states
                angle_to_group[g] = angle_group_idx
                angle_group_idx += 1
        
        n_angle_groups = len(angle_to_group)
        logger.info(f"    Found {n_angle_groups} groups with angle states")
        
        K_agg = torch.zeros(n_angle_groups, n_angle_groups, device=device)
        
        # FIX: Sum weighted couplings between groups
        for g1 in range(self.actual_groups):
            if g1 not in angle_to_group:
                continue
            machines1 = torch.where(self.labels == g1)[0]
            M_g1 = self.sys.M[machines1].sum()
            
            for g2 in range(self.actual_groups):
                if g2 not in angle_to_group:
                    continue
                machines2 = torch.where(self.labels == g2)[0]
                M_g2 = self.sys.M[machines2].sum()
                
                # FIX: Weight coupling by (M_i * M_j) / (M_G * M_H)
                coupling_sum = 0.0
                for m1 in machines1:
                    for m2 in machines2:
                        weight = (self.sys.M[m1] * self.sys.M[m2]) / (M_g1 * M_g2)
                        coupling_sum += B_full[m1, m2].item() * weight.item()
                
                K_agg[angle_to_group[g1], angle_to_group[g2]] = coupling_sum
        
        self.K_red = K_agg
        self.angle_to_group = angle_to_group  # Store for use in f_red
        
        # Define reduced ODE for aggregated swing equations
        def f_red(z, u=None):
            """Aggregated swing equation dynamics - FIXED."""
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            batch_size = z.shape[0]
            z_dot = torch.zeros_like(z)
            
            # FIX: Extract angles and frequencies correctly
            # State is ordered as [θ1, ω1, θ2, ω2, ...]
            theta = z[:, 0::2]  # angles
            omega = z[:, 1::2]  # frequencies
            
            n_groups_with_angles = self.K_red.shape[0]
            
            # FIX: Map angle indices correctly
            # Not all groups have angles, so we need to map carefully
            angle_indices = []
            freq_indices = []
            
            state_idx = 0
            for g in range(self.actual_groups):
                if state_idx >= z.shape[1]:
                    break
                    
                # Check if this group has angle states
                if g in self.angle_to_group:
                    angle_indices.append(state_idx)
                    state_idx += 1
                
                # All groups have frequency states
                if state_idx < z.shape[1]:
                    freq_indices.append(state_idx)
                    state_idx += 1
            
            # Angle dynamics: θ_dot = ω
            for i, angle_idx in enumerate(angle_indices):
                # Find corresponding frequency index
                freq_idx = angle_idx + 1  # frequency follows angle in state vector
                if freq_idx < z.shape[1]:
                    z_dot[:, angle_idx] = z[:, freq_idx]
            
            # Frequency dynamics: ω_dot = (P - D*ω - coupling) / M
            if n_groups_with_angles > 0 and len(angle_indices) > 0:
                # Extract angles for groups that have them
                theta_angle_groups = theta[:, :len(angle_indices)]
                
                # Compute coupling terms: K_ij * sin(θ_i - θ_j)
                θ_i = theta_angle_groups.unsqueeze(2)  # (batch, n_angle, 1)
                θ_j = theta_angle_groups.unsqueeze(1)  # (batch, 1, n_angle)
                sin_diff = torch.sin(θ_i - θ_j)  # (batch, n_angle, n_angle)
                
                K_expanded = self.K_red.unsqueeze(0)  # (1, n_angle, n_angle)
                coupling_terms = (K_expanded * sin_diff).sum(dim=2)  # (batch, n_angle)
            
            # Update frequency dynamics for all groups
            for g in range(self.actual_groups):
                # Find the frequency index for this group
                freq_idx = None
                state_idx = 0
                for g_iter in range(g + 1):
                    if g_iter in self.angle_to_group and g_iter < g:
                        state_idx += 1  # Skip angle
                    if g_iter <= g:
                        state_idx += 1  # Count frequency
                freq_idx = state_idx - 1
                
                if freq_idx >= z.shape[1]:
                    continue
                    
                # Get the frequency for this group
                omega_g = z[:, freq_idx]
                
                # Check if this group has angle states and coupling
                if g in self.angle_to_group and n_groups_with_angles > 0:
                    # Include coupling term
                    angle_idx = self.angle_to_group[g]
                    if angle_idx < coupling_terms.shape[1]:
                        coupling = coupling_terms[:, angle_idx]
                        omega_dot = (self.P_eq[g] - self.D_eq[g] * omega_g - coupling) / self.M_eq[g]
                    else:
                        omega_dot = (self.P_eq[g] - self.D_eq[g] * omega_g) / self.M_eq[g]
                else:
                    # No coupling (no angle state)
                    omega_dot = (self.P_eq[g] - self.D_eq[g] * omega_g) / self.M_eq[g]
                
                z_dot[:, freq_idx] = omega_dot
            
            return z_dot
        
        self.f_red = f_red
        
        logger.info(f"    Aggregated {self.sys.n_machines} machines into {self.actual_groups} groups")
        logger.info(f"    Reduced coupling matrix shape: {self.K_red.shape}")
        logger.info(f"    Angle-to-group mapping: {self.angle_to_group}")
    # BaseReducer API
    def fit(self, x):
        """No fitting needed for coherency reducer."""
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space: z = x @ P"""
        return x @ self.P
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space: x = z @ Pi (Pi = P.T)"""
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
        if hasattr(self, 'M_eq'):
            self.M_eq = self.M_eq.to(device)
        if hasattr(self, 'D_eq'):
            self.D_eq = self.D_eq.to(device)
        if hasattr(self, 'P_eq'):
            self.P_eq = self.P_eq.to(device)
        if hasattr(self, 'K_red'):
            self.K_red = self.K_red.to(device)
        return self