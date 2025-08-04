"""
Fixed Lyapunov-Coherency Reducer with Aggregated Dynamics
=========================================================

This implementation fixes all issues from the post-mortem:
1. Proper dimension handling with strict/flexible modes
2. Robust clustering with empty group handling
3. Aggregated swing equation dynamics for coherent groups
4. Energy-based gamma computation
"""

import torch
import numpy as np
import warnings
import logging
from typing import Optional, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from neural_clbf.dimension_reduction.base import BaseReducer

logger = logging.getLogger(__name__)


class LyapCoherencyReducer(BaseReducer):
    """
    Dynamic coherency reducer with aggregated dynamics.
    
    Key improvements:
    1. Flexible dimension handling (strict vs adaptive)
    2. Robust clustering with fallbacks
    3. Physical aggregated dynamics for coherent groups
    4. Proper energy deviation tracking
    """
    
    def __init__(self, sys, n_groups: int, snaps: torch.Tensor,
                 λ: float = 0.7, strict_dim: bool = False,
                 min_group_size: int = 1, reference_weighting: float = 2.0):
        """
        Args:
            sys: Power system object
            n_groups: Desired number of coherent groups
            snaps: State snapshots for coherency analysis
            λ: Weighting between correlation (1) and gap (0)
            strict_dim: If True, raise error if exact dimension not achieved
            min_group_size: Minimum machines per group
            reference_weighting: Extra weight for reference machine clustering
        """
        # Ensure system has required attributes
        if not hasattr(sys, 'n_machines'):
            if hasattr(sys, 'N_NODES'):
                sys.n_machines = sys.N_NODES
            else:
                raise AttributeError("System must have n_machines or N_NODES")
        
        # Compute achievable dimension
        self.n_machines = sys.n_machines
        self.requested_groups = n_groups
        self.min_group_size = min_group_size
        self.reference_weighting = reference_weighting
        
        # Maximum possible dimension (2*n_groups - 1 due to reference)
        max_dim = min(2 * n_groups, 2 * self.n_machines - 1)
        
        # Initialize with expected dimension
        super().__init__(latent_dim=max_dim)
        
        self.sys = sys
        self.λ = λ
        self.strict_dim = strict_dim
        
        # Build the reducer
        self._build(snaps)
    
    def _build(self, X):
        """Build coherency-based projection with robust clustering."""
        device = X.device
        dtype = X.dtype
        N = self.n_machines
        
        logger.info(f"\nBuilding Lyapunov-Coherency Reducer:")
        logger.info(f"  Machines: {N}, Requested groups: {self.requested_groups}")
        logger.info(f"  Correlation weight λ: {self.λ}")
        
        # Step 1: Extract physical quantities
        delta_abs, omega = self._extract_states(X)
        
        # Step 2: Compute machine energies
        E_machines = self._compute_machine_energies(delta_abs, omega)
        
        # Step 3: Build distance matrix
        D = self._compute_distance_matrix(E_machines)
        
        # Step 4: Hierarchical clustering
        labels = self._perform_clustering(D, N)
        
        # Step 5: Fix clustering issues
        labels = self._fix_clustering_issues(labels, N)
        
        # Step 6: Build projection matrix
        self._build_projection_matrix(labels, N, device, dtype)
        
        # Step 7: Compute energy deviation
        self._compute_energy_deviation(X)
        
        # Step 8: Build aggregated dynamics
        self._build_aggregated_dynamics(labels)
        
        # Final reporting
        self._report_grouping()
    
    def _extract_states(self, X):
        """Extract absolute angles and frequencies from state vector."""
        try:
            # Use system method if available
            delta_abs = self.sys.state_to_absolute_angles(X)
        except Exception as e:
            logger.warning(f"  state_to_absolute_angles failed: {e}")
            # Fallback: assume standard structure
            n_angles = self.n_machines - 1
            delta_abs = torch.zeros(X.shape[0], self.n_machines, device=X.device)
            if n_angles > 0:
                delta_abs[:, 1:] = X[:, :n_angles]
        
        # Extract frequencies (last N states)
        omega = X[:, -self.n_machines:]
        
        return delta_abs, omega
    
    def _compute_machine_energies(self, delta_abs, omega):
        """Compute kinetic and potential energy for each machine."""
        device = delta_abs.device
        dtype = delta_abs.dtype
        
        # Get system parameters
        M = torch.as_tensor(self.sys.M, device=device, dtype=dtype)
        
        # Kinetic energy: 0.5 * M * ω²
        E_kinetic = 0.5 * M.unsqueeze(0) * omega ** 2
        
        # Potential energy
        try:
            if hasattr(self.sys, 'potential_energy_per_machine'):
                E_potential = self.sys.potential_energy_per_machine(delta_abs)
            else:
                # Approximate potential energy
                E_potential = self._approximate_potential_energy(delta_abs)
        except Exception as e:
            logger.warning(f"  Potential energy computation failed: {e}")
            E_potential = torch.zeros_like(E_kinetic)
        
        # Total energy per machine
        E_total = E_kinetic + E_potential
        
        # Handle numerical issues
        if not torch.isfinite(E_total).all():
            logger.warning("  Non-finite energies detected, clamping...")
            E_total = torch.nan_to_num(E_total, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add small noise for numerical stability
        E_total = E_total + 1e-9 * torch.randn_like(E_total)
        
        return E_total
    
    def _approximate_potential_energy(self, delta_abs):
        """Approximate potential energy using network coupling."""
        device = delta_abs.device
        dtype = delta_abs.dtype
        
        # Get coupling matrix
        if hasattr(self.sys, 'K'):
            K = torch.as_tensor(self.sys.K, device=device, dtype=dtype)
        elif hasattr(self.sys, 'B_matrix'):
            K = torch.as_tensor(self.sys.B_matrix, device=device, dtype=dtype)
        else:
            logger.warning("  No coupling matrix found")
            return torch.zeros_like(delta_abs)
        
        # Potential energy: -sum_ij K_ij cos(δ_i - δ_j)
        N = delta_abs.shape[0]
        E_pot = torch.zeros(N, self.n_machines, device=device)
        
        for i in range(self.n_machines):
            for j in range(self.n_machines):
                if i != j and K[i, j] != 0:
                    E_pot[:, i] -= K[i, j] * torch.cos(delta_abs[:, i] - delta_abs[:, j])
        
        return E_pot
    
    def _compute_distance_matrix(self, E_machines):
        """Compute coherency distance matrix."""
        device = E_machines.device
        N = self.n_machines
        
        # Correlation matrix
        E_cpu = E_machines.detach().cpu().numpy()
        
        try:
            corr = np.corrcoef(E_cpu.T)
            # Handle NaN/Inf
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            np.fill_diagonal(corr, 1.0)
        except Exception as e:
            logger.warning(f"  Correlation failed: {e}")
            corr = np.eye(N)
        
        # Gap matrix (max energy deviation)
        gap_tensor = (E_machines[:, :, None] - E_machines[:, None, :]).abs().max(0).values
        max_gap = gap_tensor.max()
        
        if max_gap > 1e-9:
            gap = (gap_tensor / max_gap).cpu().numpy()
        else:
            gap = np.zeros((N, N))
        
        # Combined distance
        D = self.λ * (1 - np.abs(corr)) + (1 - self.λ) * gap
        
        # Reference machine weighting
        if self.reference_weighting > 1.0:
            D[0, :] *= self.reference_weighting
            D[:, 0] *= self.reference_weighting
        
        # Ensure symmetry
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        
        return D
    
    def _perform_clustering(self, D, N):
        """Perform hierarchical clustering with robustness."""
        # Convert to condensed form
        D_condensed = squareform(D, checks=False)
        
        # Check for valid distances
        if not np.isfinite(D_condensed).all() or D_condensed.max() < 1e-10:
            logger.warning("  Invalid distance matrix, using fallback grouping")
            # Fallback: cyclic grouping
            labels = np.arange(N) % self.requested_groups
        else:
            try:
                # Hierarchical clustering
                Z = linkage(D_condensed, method='average')
                
                # Extract clusters
                labels = fcluster(Z, t=self.requested_groups, criterion='maxclust') - 1
                
            except Exception as e:
                logger.error(f"  Clustering failed: {e}")
                # Fallback grouping
                labels = np.arange(N) % self.requested_groups
        
        return torch.as_tensor(labels, dtype=torch.long, device=D_condensed.device)
    
    def _fix_clustering_issues(self, labels, N):
        """Fix reference isolation and empty groups."""
        labels = labels.clone()
        device = labels.device
        
        # Count group sizes
        unique_groups, counts = torch.unique(labels, return_counts=True)
        n_actual_groups = len(unique_groups)
        
        logger.info(f"  Initial clustering: {n_actual_groups} groups")
        
        # Fix 1: Ensure reference machine (0) is not alone
        ref_group = labels[0].item()
        ref_group_size = (labels == ref_group).sum().item()
        
        if ref_group_size == 1:
            logger.info(f"    Fixing: Reference machine alone in group {ref_group}")
            # Find largest other group
            other_groups = [g for g in unique_groups if g != ref_group]
            if other_groups:
                group_sizes = [(labels == g).sum().item() for g in other_groups]
                largest_group = other_groups[np.argmax(group_sizes)]
                labels[0] = largest_group
        
        # Fix 2: Merge small groups
        if self.min_group_size > 1:
            for g in unique_groups:
                size = (labels == g).sum().item()
                if size < self.min_group_size and size > 0:
                    # Merge with nearest group
                    machines_in_g = torch.where(labels == g)[0]
                    
                    # Find nearest group based on original distance
                    # This requires access to distance matrix
                    # For now, merge with largest group
                    other_groups = [og for og in unique_groups if og != g]
                    if other_groups:
                        sizes = [(labels == og).sum().item() for og in other_groups]
                        target_group = other_groups[np.argmax(sizes)]
                        labels[machines_in_g] = target_group
        
        # Fix 3: Relabel to ensure continuous indices
        unique_final = torch.unique(labels)
        relabel_map = {old.item(): new for new, old in enumerate(unique_final)}
        
        new_labels = labels.clone()
        for old, new in relabel_map.items():
            new_labels[labels == old] = new
        
        self.actual_groups = len(unique_final)
        logger.info(f"  Final clustering: {self.actual_groups} groups")
        
        return new_labels
    
    def _build_projection_matrix(self, labels, N, device, dtype):
        """Build orthonormal projection matrix."""
        state_dim = 2 * N - 1  # N-1 angles + N frequencies
        
        # Determine actual latent dimension
        latent_dim = 0
        group_has_angle = torch.zeros(self.actual_groups, dtype=torch.bool)
        
        for g in range(self.actual_groups):
            machines = torch.where(labels == g)[0]
            if len(machines) > 0:
                # Angle state exists if any non-reference machine in group
                if any(m > 0 for m in machines):
                    latent_dim += 1
                    group_has_angle[g] = True
                # Frequency state always exists
                latent_dim += 1
        
        # Update latent dimension
        self.latent_dim = min(latent_dim, state_dim)
        
        logger.info(f"  Building projection: {state_dim}D → {self.latent_dim}D")
        
        # Check strict dimension requirement
        if self.strict_dim and self.latent_dim != 2 * self.requested_groups:
            raise ValueError(
                f"Could not achieve requested dimension {2*self.requested_groups}, "
                f"got {self.latent_dim}"
            )
        
        # Build projection matrix
        P = torch.zeros(state_dim, self.latent_dim, dtype=dtype, device=device)
        
        col_idx = 0
        self.group_to_cols = {}  # Map group to column indices
        
        for g in range(self.actual_groups):
            if col_idx >= self.latent_dim:
                break
            
            machines = torch.where(labels == g)[0]
            if len(machines) == 0:
                continue
            
            self.group_to_cols[g] = []
            
            # Angle component (if exists)
            if group_has_angle[g]:
                angle_machines = [m.item() for m in machines if m > 0]
                if angle_machines:
                    # Normalized aggregation
                    weight = 1.0 / np.sqrt(len(angle_machines))
                    for m in angle_machines:
                        P[m - 1, col_idx] = weight  # Angle index is m-1
                    self.group_to_cols[g].append(col_idx)
                    col_idx += 1
            
            # Frequency component
            if col_idx < self.latent_dim:
                # Normalized aggregation
                weight = 1.0 / np.sqrt(len(machines))
                for m in machines:
                    P[N - 1 + m, col_idx] = weight  # Frequency index
                self.group_to_cols[g].append(col_idx)
                col_idx += 1
        
        # Verify orthogonality
        ortho_check = P.T @ P
        ortho_error = (ortho_check - torch.eye(self.latent_dim, device=device)).norm()
        
        if ortho_error > 1e-6:
            logger.warning(f"  Orthogonality error: {ortho_error:.2e}")
            # Force orthogonalization
            Q, R = torch.linalg.qr(P)
            P = Q[:, :self.latent_dim]
        
        # Store matrices
        Pi = P.T  # Pseudo-inverse for orthonormal P
        
        self.register_buffer("P", P.detach())
        self.register_buffer("Pi", Pi.detach())
        self.register_buffer("labels", labels.detach())
        self.register_buffer("group_has_angle", group_has_angle.detach())
        
        # Verification
        logger.info(f"  Projection properties:")
        logger.info(f"    ||P^T @ P - I|| = {ortho_error:.2e}")
        logger.info(f"    Condition number: {torch.linalg.cond(P):.2e}")
    
    def _compute_energy_deviation(self, X):
        """Compute maximum energy deviation for gamma."""
        logger.info("  Computing energy deviation...")
        
        try:
            # Test reconstruction on samples
            n_test = min(1000, X.shape[0])
            idx = torch.randperm(X.shape[0])[:n_test]
            X_test = X[idx]
            
            # Project and reconstruct
            Z_test = self.forward(X_test)
            X_recon = self.inverse(Z_test)
            
            # Compute energy deviation
            if hasattr(self.sys, 'energy_function'):
                E_orig = self.sys.energy_function(X_test)
                E_recon = self.sys.energy_function(X_recon)
                
                # Handle shape mismatches
                if E_orig.shape != E_recon.shape:
                    E_orig = E_orig.flatten()
                    E_recon = E_recon.flatten()
                
                ΔE = (E_orig - E_recon).abs()
                ΔE_finite = ΔE[torch.isfinite(ΔE)]
                
                if ΔE_finite.numel() > 0:
                    # Use 95th percentile as robust estimate
                    self.deltaE_max = torch.quantile(ΔE_finite, 0.95)
                else:
                    self.deltaE_max = torch.tensor(0.1, device=X.device)
            else:
                # Use state reconstruction error
                state_errors = (X_test - X_recon).norm(dim=1)
                self.deltaE_max = torch.quantile(state_errors, 0.95)
            
            logger.info(f"    Max energy deviation: {self.deltaE_max:.2e}")
            
        except Exception as e:
            logger.error(f"  Energy deviation failed: {e}")
            self.deltaE_max = torch.tensor(1.0, device=X.device)
        
        self.register_buffer("deltaE_max", self.deltaE_max.detach())
    
    def _build_aggregated_dynamics(self, labels):
        """Build aggregated swing equation dynamics."""
        logger.info("  Building aggregated dynamics...")
        
        device = self.P.device
        dtype = self.P.dtype
        
        # Aggregate parameters by group
        M_agg = []  # Aggregated inertia
        D_agg = []  # Aggregated damping
        P_agg = []  # Aggregated mechanical power
        machines_per_group = []
        
        for g in range(self.actual_groups):
            machines = torch.where(labels == g)[0]
            if len(machines) == 0:
                continue
            
            # Sum parameters
            M_g = sum(self.sys.M[m] for m in machines)
            D_g = sum(self.sys.D[m] for m in machines)
            P_g = sum(self.sys.P_mechanical[m] for m in machines)
            
            M_agg.append(M_g)
            D_agg.append(D_g)
            P_agg.append(P_g)
            machines_per_group.append(machines)
        
        self.M_agg = torch.tensor(M_agg, device=device, dtype=dtype)
        self.D_agg = torch.tensor(D_agg, device=device, dtype=dtype)
        self.P_agg = torch.tensor(P_agg, device=device, dtype=dtype)
        
        # Build aggregated coupling matrix
        self._build_aggregated_coupling(machines_per_group, device, dtype)
        
        # Define aggregated dynamics
        self.f_red = self._create_aggregated_dynamics()
        
        logger.info(f"    Aggregated {self.n_machines} machines → {self.actual_groups} groups")
    
    def _build_aggregated_coupling(self, machines_per_group, device, dtype):
        """Build aggregated coupling matrix between coherent groups."""
        n_angle_groups = sum(1 for g in range(self.actual_groups) if self.group_has_angle[g])
        
        # Get original coupling matrix
        if hasattr(self.sys, 'K'):
            K_full = torch.as_tensor(self.sys.K, device=device, dtype=dtype)
        elif hasattr(self.sys, 'B_matrix'):
            K_full = torch.as_tensor(self.sys.B_matrix, device=device, dtype=dtype)
        else:
            logger.warning("  No coupling matrix found")
            K_full = torch.zeros(self.n_machines, self.n_machines, device=device)
        
        # Aggregate coupling
        K_agg = torch.zeros(n_angle_groups, n_angle_groups, device=device, dtype=dtype)
        
        angle_group_map = {}
        angle_idx = 0
        
        for g in range(self.actual_groups):
            if self.group_has_angle[g]:
                angle_group_map[g] = angle_idx
                angle_idx += 1
        
        # Sum inter-group couplings
        for g1 in range(self.actual_groups):
            if g1 not in angle_group_map:
                continue
            
            for g2 in range(self.actual_groups):
                if g2 not in angle_group_map:
                    continue
                
                # Sum all couplings between groups
                coupling = 0.0
                for m1 in machines_per_group[g1]:
                    for m2 in machines_per_group[g2]:
                        coupling += K_full[m1, m2].item()
                
                K_agg[angle_group_map[g1], angle_group_map[g2]] = coupling
        
        self.K_agg = K_agg
        self.angle_group_map = angle_group_map
    
    def _create_aggregated_dynamics(self):
        """Create aggregated swing equation dynamics function."""
        def f_red(z, u=None):
            """
            Aggregated swing equation dynamics.
            State ordering: [θ₁, ω₁, θ₂, ω₂, ...]
            where θᵢ, ωᵢ are aggregated angle/frequency for group i
            """
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            batch_size = z.shape[0]
            device = z.device
            z_dot = torch.zeros_like(z)
            
            # Parse state vector
            state_idx = 0
            angles = []
            frequencies = []
            group_to_state_idx = {}
            
            for g in range(self.actual_groups):
                if g not in self.group_to_cols:
                    continue
                
                cols = self.group_to_cols[g]
                group_to_state_idx[g] = {}
                
                # Angle state (if exists)
                if self.group_has_angle[g] and state_idx < z.shape[1]:
                    angles.append((g, state_idx))
                    group_to_state_idx[g]['angle'] = state_idx
                    state_idx += 1
                
                # Frequency state
                if state_idx < z.shape[1]:
                    frequencies.append((g, state_idx))
                    group_to_state_idx[g]['freq'] = state_idx
                    state_idx += 1
            
            # Angle dynamics: θ̇ = ω
            for g, idx in angles:
                if 'freq' in group_to_state_idx[g]:
                    freq_idx = group_to_state_idx[g]['freq']
                    z_dot[:, idx] = z[:, freq_idx]
            
            # Frequency dynamics: M·ω̇ = P - D·ω - K·sin(θ)
            if len(angles) > 0:
                # Extract angle values
                angle_indices = [idx for _, idx in angles]
                θ = z[:, angle_indices]  # (batch, n_angle_groups)
                
                # Compute coupling terms
                if self.K_agg.shape[0] > 0:
                    θ_i = θ.unsqueeze(2)  # (batch, n_angle, 1)
                    θ_j = θ.unsqueeze(1)  # (batch, 1, n_angle)
                    sin_diff = torch.sin(θ_i - θ_j)  # (batch, n_angle, n_angle)
                    
                    K_expanded = self.K_agg.unsqueeze(0)
                    coupling = (K_expanded * sin_diff).sum(dim=2)  # (batch, n_angle)
                else:
                    coupling = torch.zeros(batch_size, len(angles), device=device)
            
            # Update frequency derivatives
            for i, (g, idx) in enumerate(frequencies):
                # Base dynamics
                ω = z[:, idx]
                ω_dot = (self.P_agg[g] - self.D_agg[g] * ω) / self.M_agg[g]
                
                # Add coupling if this group has angles
                if g in self.angle_group_map:
                    angle_idx = self.angle_group_map[g]
                    if angle_idx < coupling.shape[1]:
                        ω_dot = ω_dot - coupling[:, angle_idx] / self.M_agg[g]
                
                z_dot[:, idx] = ω_dot
            
            return z_dot.squeeze(0) if batch_size == 1 else z_dot
        
        return f_red
    
    def _report_grouping(self):
        """Report final grouping configuration."""
        logger.info("\n  Final coherent groups:")
        
        for g in range(self.actual_groups):
            machines = torch.where(self.labels == g)[0].tolist()
            has_angle = self.group_has_angle[g].item()
            
            logger.info(f"    Group {g}: machines {machines}")
            logger.info(f"      M_agg = {self.M_agg[g]:.3f}, D_agg = {self.D_agg[g]:.3f}")
            logger.info(f"      P_agg = {self.P_agg[g]:.3f}, has angle: {has_angle}")
    
    # BaseReducer interface
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to coherent coordinates."""
        return x @ self.P
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from coherent coordinates."""
        return z @ self.Pi
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Batch Jacobian."""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.Pi.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
    
    def fit(self, X):
        """LCR doesn't require iterative fitting."""
        return self
    
    def compute_gamma(self, V_min: float) -> float:
        """Compute robustness margin."""
        if V_min < 1e-9:
            return float('inf')
        
        # Use square root as per theoretical definition
        gamma_val = torch.sqrt(self.deltaE_max / V_min).item()
        
        # Cap at reasonable value
        self.gamma = min(gamma_val, 100.0)
        return self.gamma