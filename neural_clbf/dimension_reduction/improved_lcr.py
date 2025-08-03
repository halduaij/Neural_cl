"""
Improved Lyapunov Coherency Reducer
===================================

Enhanced version with time-varying coherency, fuzzy membership,
and energy-preserving reconstruction for power system reduction.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from neural_clbf.dimension_reduction.base import BaseReducer


class ImprovedLyapCoherencyReducer(BaseReducer):
    """
    Enhanced Lyapunov Coherency with adaptive grouping and energy preservation.
    
    Key improvements:
    1. Time-varying coherency detection
    2. Fuzzy group membership
    3. Energy-preserving reconstruction
    4. Adaptive grouping during simulation
    """
    
    def __init__(
        self, 
        sys, 
        n_groups: int, 
        snaps: torch.Tensor, 
        λ: float = 0.7,
        adaptive: bool = True,
        fuzzy_sigma: float = 0.5,
        energy_weight: float = 0.3
    ):
        """
        Initialize enhanced coherency reducer.
        
        Args:
            sys: Power system object
            n_groups: Number of coherent groups
            snaps: Snapshot data for coherency identification
            λ: Weight between correlation and gap (0=gap only, 1=correlation only)
            adaptive: Enable adaptive grouping
            fuzzy_sigma: Sigma for fuzzy membership functions
            energy_weight: Weight for energy preservation in reconstruction
        """
        # Set latent dimension (2 per group: angle and frequency)
        super().__init__(latent_dim=2 * n_groups)
        
        self.sys = sys
        self.n_groups = n_groups
        self.λ = λ
        self.adaptive = adaptive
        self.fuzzy_sigma = fuzzy_sigma
        self.energy_weight = energy_weight
        
        # Initialize data structures
        self.time_varying_groups = {}
        self.group_centers = None
        self.membership = None
        self.energy_jacobian = None
        
        # Build reducer
        self._build_enhanced(snaps)
    
    def _build_enhanced(self, X):
        """Build enhanced coherency reducer with all improvements."""
        N = self.sys.n_machines
        self.N = N
        
        print(f"Building Enhanced Lyapunov Coherency Reducer:")
        print(f"  Machines: {N}, Groups: {self.n_groups}, Snapshots: {X.shape[0]}")
        
        # 1. Compute time-varying coherency if adaptive
        if self.adaptive:
            self._identify_time_varying_coherency(X)
        
        # 2. Compute base coherency using all data
        labels, coherency_metric = self._compute_enhanced_coherency(X)
        self.base_labels = labels
        
        # 3. Setup fuzzy membership
        self._compute_fuzzy_membership(X, labels)
        
        # 4. Build projection matrices
        self._build_projection_matrices(labels)
        
        # 5. Setup energy preservation
        self._setup_energy_preservation(X)
        
        # 6. Compute robustness margin
        self._compute_robustness_margin(X)
        
        print(f"  Completed. Max ΔV: {self.deltaV_max.item():.4f}")
    
    def _identify_time_varying_coherency(self, X):
        """Detect how coherency patterns change over time."""
        print("  Identifying time-varying coherency patterns...")
        
        window_size = min(100, X.shape[0] // 5)
        stride = window_size // 2
        
        for t in range(0, X.shape[0] - window_size, stride):
            X_window = X[t:t+window_size]
            labels, _ = self._compute_enhanced_coherency(X_window)
            self.time_varying_groups[t] = labels
        
        # Analyze group stability
        self._analyze_group_stability()
    
    def _compute_enhanced_coherency(self, X):
        """Compute coherency with improved metrics."""
        N = self.sys.n_machines
        device = X.device
        dtype = X.dtype
        
        # Extract states
        M = torch.as_tensor(self.sys.M, dtype=dtype, device=device)
        delta_abs = self.sys.state_to_absolute_angles(X)
        omega = X[:, self.sys.N_NODES - 1:]
        
        # Compute multiple coherency metrics
        
        # 1. Energy-based coherency (original)
        kin = 0.5 * M * omega ** 2
        pot = self.sys.potential_energy_per_machine(delta_abs)
        E = kin + pot
        
        # 2. Phase coherency (new)
        phase_coherency = self._compute_phase_coherency(delta_abs, omega)
        
        # 3. Frequency coherency (new)
        freq_coherency = self._compute_frequency_coherency(omega)
        
        # Combine metrics
        # Energy correlation
        corr_E = torch.corrcoef(E.T).detach().cpu().numpy()
        
        # Energy gap
        gap_E = (E[:, :, None] - E[:, None, :]).abs().max(0).values
        gap_E = (gap_E / (gap_E.max() + 1e-6)).detach().cpu().numpy()
        
        # Combined distance matrix
        D = self.λ * (1 - np.abs(corr_E)) + (1 - self.λ) * gap_E
        
        # Add phase and frequency coherency
        D = 0.5 * D + 0.3 * phase_coherency + 0.2 * freq_coherency
        
        # Hierarchical clustering
        D_condensed = D[np.triu_indices(N, 1)]
        Z = linkage(D_condensed, method='average')
        labels = fcluster(Z, t=self.n_groups, criterion='maxclust') - 1
        
        return torch.tensor(labels, dtype=torch.long, device=device), D
    
    def _compute_phase_coherency(self, delta, omega):
        """Compute coherency based on phase dynamics."""
        N = delta.shape[1]
        
        # Compute phase differences
        phase_diff = delta[:, :, None] - delta[:, None, :]  # (T, N, N)
        
        # Compute phase velocities
        phase_vel = omega[:, :, None] - omega[:, None, :]   # (T, N, N)
        
        # Coherency: machines with similar phase dynamics
        coherency = torch.exp(-phase_diff.std(0) - 0.5 * phase_vel.std(0))
        
        return 1 - coherency.cpu().numpy()
    
    def _compute_frequency_coherency(self, omega):
        """Compute coherency based on frequency dynamics."""
        # Frequency correlation over time
        corr = torch.corrcoef(omega.T)
        
        # Frequency deviation similarity
        omega_dev = omega - omega.mean(1, keepdim=True)
        dev_similarity = torch.exp(-(omega_dev[:, :, None] - omega_dev[:, None, :]).abs().mean(0))
        
        combined = 0.7 * corr + 0.3 * dev_similarity
        
        return 1 - combined.cpu().numpy()
    
    def _compute_fuzzy_membership(self, X, crisp_labels):
        """Compute fuzzy membership functions for each machine."""
        print("  Computing fuzzy membership functions...")
        
        N = self.sys.n_machines
        device = X.device
        
        # Compute group centers in feature space
        features = self._extract_coherency_features(X)
        
        self.group_centers = torch.zeros(self.n_groups, features.shape[1], device=device)
        for g in range(self.n_groups):
            mask = crisp_labels == g
            if mask.sum() > 0:
                self.group_centers[g] = features[mask].mean(0)
        
        # Compute fuzzy membership
        self.membership = torch.zeros(N, self.n_groups, device=device)
        
        for i in range(N):
            # Distance to each group center
            distances = torch.norm(features[i] - self.group_centers, dim=1)
            
            # Gaussian membership function
            self.membership[i] = torch.exp(-distances**2 / (2 * self.fuzzy_sigma**2))
            
            # Normalize
            self.membership[i] /= self.membership[i].sum()
    
    def _extract_coherency_features(self, X):
        """Extract features for coherency analysis."""
        N = self.sys.n_machines
        
        # Get energy components
        M = torch.as_tensor(self.sys.M, device=X.device)
        delta_abs = self.sys.state_to_absolute_angles(X)
        omega = X[:, self.sys.N_NODES - 1:]
        
        kin = 0.5 * M * omega ** 2
        pot = self.sys.potential_energy_per_machine(delta_abs)
        
        # Features for each machine
        features = torch.zeros(N, 6, device=X.device)
        
        features[:, 0] = kin.mean(0)      # Mean kinetic energy
        features[:, 1] = kin.std(0)       # Std kinetic energy
        features[:, 2] = pot.mean(0)      # Mean potential energy
        features[:, 3] = pot.std(0)       # Std potential energy
        features[:, 4] = omega.mean(0)    # Mean frequency
        features[:, 5] = omega.std(0)     # Std frequency
        
        # Normalize features
        features = (features - features.mean(0)) / (features.std(0) + 1e-6)
        
        return features
    
    def _build_projection_matrices(self, labels):
        """Build projection matrices with fuzzy membership."""
        state_dim = 2 * self.N - 1  # N-1 angles + N frequencies
        device = labels.device
        
        if self.membership is not None:
            # Fuzzy projection
            P = torch.zeros(state_dim, 2 * self.n_groups, device=device)
            
            for g in range(self.n_groups):
                # Angle components (skip reference machine 0)
                for i in range(1, self.N):
                    if i > 0:  # Skip reference
                        P[i-1, 2*g] = self.membership[i, g]
                
                # Frequency components
                for i in range(self.N):
                    P[self.N-1+i, 2*g+1] = self.membership[i, g]
            
            # Normalize columns
            col_norms = P.norm(dim=0, keepdim=True)
            P = P / (col_norms + 1e-6)
            
        else:
            # Crisp projection (fallback)
            P = torch.zeros(state_dim, 2 * self.n_groups, device=device)
            
            for g in range(self.n_groups):
                idx = torch.where(labels == g)[0]
                if len(idx) > 0:
                    # Angles
                    for i in idx[idx > 0]:
                        P[i-1, 2*g] = 1 / len(idx)
                    
                    # Frequencies
                    for i in idx:
                        P[self.N-1+i, 2*g+1] = 1 / len(idx)
        
        self.register_buffer("P", P.detach())
        self.register_buffer("Pi", torch.linalg.pinv(P).detach())
    
    def _setup_energy_preservation(self, X):
        """Setup energy-preserving reconstruction."""
        print("  Setting up energy-preserving reconstruction...")
        
        # Compute energy sensitivity
        self.energy_jacobian = self._compute_energy_jacobian(X)
        
        # Store original inverse function
        self._inverse_original = self.inverse
        
        # Replace with energy-preserving version
        def energy_preserving_inverse(z):
            # Basic reconstruction
            x_base = self._inverse_original(z)
            
            if self.energy_weight > 0 and hasattr(self, 'last_forward_energy'):
                # Compute energy of reconstruction
                E_recon = self.sys.energy_function(x_base.unsqueeze(0) if x_base.dim() == 1 else x_base)
                
                # Energy error
                E_error = self.last_forward_energy - E_recon
                
                # Newton step to correct energy
                if E_error.abs() > 1e-3:
                    correction = self._energy_correction_step(x_base, E_error)
                    x_corrected = x_base + self.energy_weight * correction
                    
                    # Ensure correction doesn't violate state limits
                    upper, lower = self.sys.state_limits
                    x_corrected = torch.clamp(x_corrected, lower, upper)
                    
                    return x_corrected
            
            return x_base
        
        # Monkey patch the method
        self.inverse = energy_preserving_inverse
    
    def _compute_energy_jacobian(self, X):
        """Compute Jacobian of energy function."""
        # Sample points
        n_samples = min(100, X.shape[0])
        idx = torch.randperm(X.shape[0])[:n_samples]
        X_sample = X[idx].clone().requires_grad_(True)
        
        # Compute energy and gradients
        E = self.sys.energy_function(X_sample)
        
        # Get gradients
        grads = []
        for i in range(n_samples):
            g = torch.autograd.grad(E[i], X_sample, retain_graph=True)[0][i]
            grads.append(g)
        
        return torch.stack(grads).mean(0)
    
    def _energy_correction_step(self, x, E_error):
        """Compute correction to match target energy."""
        # Approximate Newton step
        # ΔE ≈ ∇E^T Δx
        # Δx = ΔE / ||∇E||^2 * ∇E
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.clone().requires_grad_(True)
        E = self.sys.energy_function(x)
        grad_E = torch.autograd.grad(E.sum(), x)[0]
        
        # Newton step
        grad_norm_sq = (grad_E ** 2).sum() + 1e-6
        correction = (E_error / grad_norm_sq) * grad_E
        
        return correction.squeeze(0) if correction.shape[0] == 1 else correction
    
    def _analyze_group_stability(self):
        """Analyze how stable the groups are over time."""
        if not self.time_varying_groups:
            return
        
        print("  Analyzing group stability...")
        
        # Convert to matrix form
        times = sorted(self.time_varying_groups.keys())
        group_matrix = torch.stack([self.time_varying_groups[t] for t in times])
        
        # Compute group persistence
        persistence = torch.zeros(self.n_groups)
        
        for g in range(self.n_groups):
            # Find most common machines in this group
            group_masks = (group_matrix == g)
            machine_freq = group_masks.float().mean(0)
            
            # Persistence = how often the core machines stay together
            core_machines = machine_freq > 0.7
            if core_machines.sum() > 0:
                persistence[g] = machine_freq[core_machines].mean()
        
        self.group_persistence = persistence
        print(f"    Group persistence: {persistence}")
    
    def _compute_robustness_margin(self, X):
        """Compute the Lyapunov robustness margin gamma."""
        # Compute max energy deviation
        X_recon = self._inverse_original(self.forward(X))
        
        # Energy difference
        E_orig = self.sys.energy_function(X)
        E_recon = self.sys.energy_function(X_recon)
        
        deltaV = (E_orig - E_recon).abs()
        self.register_buffer("deltaV_max", deltaV.max().unsqueeze(0))
    
    def adaptive_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward projection with adaptive grouping."""
        if not self.adaptive:
            return self.forward(x)
        
        # Check if regrouping is needed
        features = self._extract_coherency_features(x.unsqueeze(0) if x.dim() == 1 else x)
        
        # Compute distances to group centers
        min_dist = float('inf')
        for i in range(features.shape[0]):
            distances = torch.norm(features[i] - self.group_centers, dim=1)
            min_dist = min(min_dist, distances.min().item())
        
        # If features have drifted significantly, update grouping
        if min_dist > 2 * self.fuzzy_sigma:
            print("  Adaptive regrouping triggered...")
            self._update_grouping(x)
        
        return self.forward(x)
    
    def _update_grouping(self, x):
        """Update grouping based on current state."""
        # Simplified version - full implementation would be more sophisticated
        X_current = x.unsqueeze(0) if x.dim() == 1 else x
        
        # Recompute membership based on current state
        features = self._extract_coherency_features(X_current)
        
        for i in range(self.N):
            distances = torch.norm(features[i] - self.group_centers, dim=1)
            self.membership[i] = torch.exp(-distances**2 / (2 * self.fuzzy_sigma**2))
            self.membership[i] /= self.membership[i].sum()
        
        # Rebuild projection matrices
        self._build_projection_matrices(self.base_labels)
    
    # Base methods
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space."""
        # Store energy for reconstruction
        if self.energy_weight > 0:
            self.last_forward_energy = self.sys.energy_function(
                x.unsqueeze(0) if x.dim() == 1 else x
            )
        
        return x @ self.P
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """This will be replaced by energy_preserving_inverse in __init__"""
        return z @ self.Pi
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Analytical Jacobian."""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.P.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
    
    def compute_gamma(self, V_min: float) -> float:
        """Compute Lyapunov robustness margin."""
        return float(self.deltaV_max.item() / V_min) if V_min > 0 else float('inf')
    
    def fit(self, X):
        """Refit with new data if needed."""
        if X is not None and X.shape[0] > 100:
            print("Refitting Lyapunov Coherency with new data...")
            self._build_enhanced(X)
        return self