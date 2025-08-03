"""
Consistent Validation for Standard and Improved Reducers
========================================================

Tests both standard and improved reducers using the EXACT same evaluation
methodology to ensure fair comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.linalg import solve_continuous_are

from neural_clbf.systems import SwingEquationSystem
from neural_clbf.dimension_reduction.base import BaseReducer
from neural_clbf.dimension_reduction.opinf import OpInfReducer
from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer


# ============ IMPROVED REDUCER IMPLEMENTATIONS ============

class ImprovedLyapCoherencyReducer(LyapCoherencyReducer):
    """
    Enhanced Lyapunov Coherency with fuzzy membership and energy preservation.
    Inherits from standard LCR to ensure compatibility.
    """
    
    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, 
                 λ: float = 0.7, fuzzy_membership: bool = True, 
                 energy_weight: float = 0.3):
        # Initialize parent class first
        super().__init__(sys, n_groups, snaps, λ)
        
        self.fuzzy_membership = fuzzy_membership
        self.energy_weight = energy_weight
        
        # Apply enhancements
        if fuzzy_membership:
            self._compute_fuzzy_membership(snaps)
        
        # Store original inverse for energy preservation
        self._inverse_original = super().inverse
    
    def _compute_fuzzy_membership(self, X):
        """Add fuzzy membership to existing grouping."""
        N = self.sys.n_machines
        device = X.device
        
        # Use existing labels from parent
        crisp_labels = self.labels
        
        # Compute features for each machine
        M = torch.as_tensor(self.sys.M, device=device)
        delta_abs = self.sys.state_to_absolute_angles(X)
        omega = X[:, self.sys.N_NODES - 1:]
        
        kin = 0.5 * M * omega ** 2
        pot = self.sys.potential_energy_per_machine(delta_abs)
        E = kin + pot
        
        # Simple features: mean and std of energy
        features = torch.zeros(N, 2, device=device)
        features[:, 0] = E.mean(0)
        features[:, 1] = E.std(0)
        
        # Normalize
        features = (features - features.mean(0)) / (features.std(0) + 1e-6)
        
        # Compute group centers
        group_centers = torch.zeros(self.n_groups, 2, device=device)
        for g in range(self.n_groups):
            mask = crisp_labels == g
            if mask.sum() > 0:
                group_centers[g] = features[mask].mean(0)
        
        # Fuzzy membership with fixed sigma
        fuzzy_sigma = 0.5
        membership = torch.zeros(N, self.n_groups, device=device)
        
        for i in range(N):
            distances = torch.norm(features[i] - group_centers, dim=1)
            membership[i] = torch.exp(-distances**2 / (2 * fuzzy_sigma**2))
            membership[i] /= membership[i].sum()
        
        self.membership = membership
        
        # Rebuild projection with fuzzy membership
        self._rebuild_projection_fuzzy()
    
    def _rebuild_projection_fuzzy(self):
        """Rebuild projection matrices with fuzzy membership."""
        state_dim = 2 * self.sys.n_machines - 1
        device = self.P.device
        
        P_fuzzy = torch.zeros(state_dim, 2 * self.n_groups, device=device)
        
        for g in range(self.n_groups):
            # Angle components (weighted by membership)
            for i in range(1, self.sys.n_machines):
                P_fuzzy[i-1, 2*g] = self.membership[i, g]
            
            # Frequency components
            for i in range(self.sys.n_machines):
                P_fuzzy[self.sys.n_machines-1+i, 2*g+1] = self.membership[i, g]
        
        # Normalize columns
        col_norms = P_fuzzy.norm(dim=0, keepdim=True)
        P_fuzzy = P_fuzzy / (col_norms + 1e-6)
        
        # Update buffers
        self.P = P_fuzzy.detach()
        self.Pi = torch.linalg.pinv(P_fuzzy).detach()
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Energy-preserving reconstruction."""
        x_base = self._inverse_original(z)
        
        # Simple energy correction (only if weight > 0)
        if self.energy_weight > 0 and hasattr(self, '_last_forward_energy'):
            # Handle batch vs single state
            if z.dim() == 1:
                z_batch = z.unsqueeze(0)
                x_batch = x_base.unsqueeze(0)
            else:
                z_batch = z
                x_batch = x_base
            
            # Compute energy
            E_recon = self.sys.energy_function(x_batch)
            
            # Only apply correction if we have stored energy
            if self._last_forward_energy is not None:
                # Make sure dimensions match
                if E_recon.shape == self._last_forward_energy.shape:
                    E_error = (self._last_forward_energy - E_recon).mean()
                    
                    if abs(E_error.item()) > 1e-3:
                        # Simple scaling correction
                        scale = 1 + self.energy_weight * 0.1 * E_error.sign()
                        scale = max(0.95, min(1.05, scale.item()))
                        x_base = x_base * scale
        
        return x_base
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Store energy for reconstruction."""
        # Store energy for later use
        if self.energy_weight > 0:
            if x.dim() == 1:
                self._last_forward_energy = self.sys.energy_function(x.unsqueeze(0))
            else:
                self._last_forward_energy = self.sys.energy_function(x)
        
        return super().forward(x)


class ImprovedOpInfReducer(OpInfReducer):
    """
    Enhanced OpInf with physics-informed projection and stability enforcement.
    Inherits from standard OpInf to ensure compatibility.
    """
    
    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1,
                 physics_informed: bool = True, stability_margin: float = 0.5):
        super().__init__(latent_dim, n_full, n_controls)
        
        self.physics_informed = physics_informed
        self.stability_margin = stability_margin
    
    def fit(self, X, Xdot, V_fn, V_min):
        """Enhanced fitting with physics-informed projection."""
        device = X.device
        
        if self.physics_informed:
            print("  Using physics-informed projection...")
            
            # 1. Compute mean
            self.μ = X.mean(0)
            X_centered = X - self.μ
            
            # 2. Combine state and derivative information
            # Standard POD on states
            U1, S1, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
            
            # POD on derivatives (for dynamics)
            Xdot_centered = Xdot - Xdot.mean(0)
            U2, S2, _ = torch.linalg.svd(Xdot_centered.T, full_matrices=False)
            
            # Energy-based modes (if possible)
            try:
                # Sample gradients of energy function
                n_samples = min(100, X.shape[0])
                idx = torch.randperm(X.shape[0])[:n_samples]
                X_sample = X[idx].clone().requires_grad_(True)
                
                V_sample = V_fn(X_sample)
                if V_sample.requires_grad and V_sample.numel() == n_samples:
                    grad_V = torch.autograd.grad(V_sample.sum(), X_sample)[0]
                    grad_V_centered = grad_V - grad_V.mean(0)
                    U3, S3, _ = torch.linalg.svd(grad_V_centered.T, full_matrices=False)
                else:
                    U3 = None
            except:
                U3 = None
            
            # Combine modes (with proper dimension handling)
            n_state = min(self.latent_dim // 2, U1.shape[1])
            n_dyn = min(self.latent_dim // 3, U2.shape[1])
            n_energy = 0
            
            if U3 is not None:
                n_energy = min(self.latent_dim - n_state - n_dyn, U3.shape[1])
            
            # Adjust if total exceeds latent_dim
            total = n_state + n_dyn + n_energy
            if total > self.latent_dim:
                n_state = self.latent_dim - n_dyn - n_energy
            
            # Build combined basis
            bases = [U1[:, :n_state], U2[:, :n_dyn]]
            if n_energy > 0 and U3 is not None:
                bases.append(U3[:, :n_energy])
            
            combined = torch.cat(bases, dim=1)
            
            # Orthogonalize
            Q, _ = torch.linalg.qr(combined)
            self.proj = Q[:, :self.latent_dim]
            
        else:
            # Standard OpInf fitting
            super().fit(X, Xdot, V_fn, V_min)
            return self
        
        # Continue with dynamics fitting
        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        
        # Fit dynamics
        from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
        self.dyn = GPOpInfDynamics(self.latent_dim, self.n_controls).to(device)
        self.dyn.reg = torch.tensor(0.1, device=device)
        
        U = torch.zeros(Z.shape[0], self.n_controls, device=device)
        self.dyn.fit(Z, U, dZdt)
        
        # Enforce stability
        self._enforce_stability()
        
        # Compute gamma
        self.compute_gamma(X, V_fn, V_min)
        
        return self
    
    def _enforce_stability(self):
        """Ensure dynamics are stable."""
        if self.dyn is None:
            return
        
        A = self.dyn.A.data
        eigvals = torch.linalg.eigvals(A)
        max_real = eigvals.real.max().item()
        
        if max_real > -self.stability_margin:
            print(f"  Stabilizing dynamics (max eigenvalue: {max_real:.3f})...")
            shift = max_real + self.stability_margin
            self.dyn.A.data = A - shift * torch.eye(self.latent_dim, device=A.device)


class ImprovedSymplecticProjectionReducer(SymplecticProjectionReducer):
    """
    Enhanced SPR with data-informed basis selection.
    Simple improvements that maintain stability.
    """
    
    def __init__(self, A: torch.Tensor, J: torch.Tensor, R: torch.Tensor, 
                 latent_dim: int, use_data: bool = True):
        # Initialize parent
        super().__init__(A, J, R, latent_dim)
        
        self.use_data = use_data
        self._original_T = self.T.clone()
    
    def enhance_with_data(self, X: torch.Tensor):
        """Enhance projection using trajectory data."""
        if not self.use_data:
            return
        
        print("  Enhancing SPR with trajectory data...")
        
        # Compute POD modes from data
        X_centered = X - X.mean(0)
        U, S, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
        
        # Take dominant POD modes
        n_pod = min(self.latent_dim // 2, U.shape[1])
        pod_modes = U[:, :n_pod]
        
        # Combine with symplectic modes
        combined = torch.cat([self._original_T, pod_modes], dim=1)
        
        # Orthogonalize
        Q, _ = torch.linalg.qr(combined)
        
        # Select best modes (those that preserve symplectic structure)
        scores = []
        J = self.J if hasattr(self, 'J') else torch.zeros(self.T.shape[0], self.T.shape[0])
        
        for i in range(Q.shape[1]):
            T_test = Q[:, :i+1]
            # Simple score based on projection error
            X_proj = X @ T_test @ T_test.T
            error = (X - X_proj).norm() / X.norm()
            scores.append(1.0 / (error + 1e-6))
        
        # Select top modes
        idx = torch.argsort(torch.tensor(scores), descending=True)[:self.latent_dim]
        idx_sorted = torch.sort(idx).values
        
        self.T = Q[:, idx_sorted]
        self.Ti = torch.linalg.pinv(self.T)


# ============ SYSTEM SETUP ============

def create_test_power_system():
    """Create test system exactly as in validation_file4.py"""
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                      1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                      0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                     dtype=torch.float32) * 3.0
    
    P = torch.tensor([-0.19983394, -0.25653884, -0.25191885, -0.10242008, -0.34510365,
                       0.23206371, 0.4404325, 0.5896664, 0.26257738, -0.36892462], 
                     dtype=torch.float32)
    
    K = torch.tensor([[18.217514, 1.1387165, 1.360604, 1.2791332, 0.572532,
                       1.2913872, 1.051677, 3.1750703, 1.7979614, 5.1949754],
                      [1.1387165, 13.809675, 2.2804017, 0.7847816, 0.3512633,
                       0.7922997, 0.6452313, 0.6032209, 0.5339053, 3.1851373],
                      [1.360604, 2.2804017, 14.325089, 1.0379374, 0.46457428,
                       1.0478808, 0.85337085, 0.7284146, 0.66612864, 3.2071328],
                      [1.2791332, 0.7847816, 1.0379374, 14.242994, 2.7465186,
                       2.0251245, 1.6492164, 0.7610195, 0.90347135, 1.6209806],
                      [0.572532, 0.3512633, 0.46457428, 2.7465186, 10.82826,
                       0.90643305, 0.7381789, 0.34062758, 0.4043881, 0.72554076],
                      [1.2913872, 0.7922997, 1.0478808, 2.0251245, 0.90643305,
                       16.046877, 4.1242394, 0.76831, 0.9121265, 1.6365094],
                      [1.051677, 0.6452313, 0.85337085, 1.6492164, 0.7381789,
                       4.1242394, 12.7116995, 0.62569463, 0.74281555, 1.3327368],
                      [3.1750703, 0.6032209, 0.7284146, 0.7610195, 0.34062758,
                       0.76831, 0.62569463, 11.512734, 1.4515272, 2.5534966],
                      [1.7979614, 0.5339053, 0.66612864, 0.90347135, 0.4043881,
                       0.9121265, 0.74281555, 1.4515272, 10.306445, 1.6531545],
                      [5.1949754, 3.1851373, 3.2071328, 1.6209806, 0.72554076,
                       1.6365094, 1.3327368, 2.5534966, 1.6531545, 25.04985]], 
                     dtype=torch.float32)
    
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                               -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                              dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    sys = SwingEquationSystem(params, dt=0.01)
    sys.delta_star = delta_star
    
    # Fixed dynamics
    def _f_fixed(x, params):
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, 19, 1))
        
        theta = x[:, :9]
        omega = x[:, 9:]
        
        delta_eq = delta_star
        theta_eq = delta_eq[0] - delta_eq[1:]
        
        delta = torch.zeros(batch_size, 10)
        delta[:, 0] = delta_eq[0]
        delta[:, 1:] = delta_eq[1:] + (theta_eq - theta)
        
        for i in range(1, 10):
            f[:, i-1, 0] = omega[:, 0] - omega[:, i]
        
        for i in range(10):
            omega_dot = P[i] / M[i] - (D[i] / M[i]) * omega[:, i]
            for j in range(10):
                if i != j:
                    omega_dot -= (K[i, j] / M[i]) * torch.sin(delta[:, i] - delta[:, j])
            f[:, 9+i, 0] = omega_dot
        
        return f
    
    sys._f = _f_fixed
    return sys


# ============ CONSISTENT EVALUATION ============

def evaluate_reducer_consistent(sys, reducer, test_data_size=1000):
    """
    Evaluate ANY reducer using EXACTLY the same methodology.
    This ensures fair comparison between standard and improved versions.
    """
    # Get equilibrium
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    # Generate test data
    X_test = []
    for _ in range(test_data_size):
        x = x_eq + 0.01 * torch.randn_like(x_eq)
        X_test.append(x)
    X_test = torch.stack(X_test)
    
    print(f"\nEvaluating {type(reducer).__name__} (d={reducer.latent_dim}):")
    
    # 1. State Reconstruction
    Z = reducer.forward(X_test)
    X_reconstructed = reducer.inverse(Z)
    
    angle_error_rad = (X_reconstructed[:, :9] - X_test[:, :9]).abs()
    angle_error_deg = angle_error_rad * 180 / np.pi
    angle_rmse = torch.sqrt((angle_error_deg ** 2).mean()).item()
    
    freq_error_pu = (X_reconstructed[:, 9:] - X_test[:, 9:]).abs()
    freq_error_hz = freq_error_pu * 60 / (2 * np.pi)
    freq_rmse = torch.sqrt((freq_error_hz ** 2).mean()).item()
    
    print(f"1. State Reconstruction:")
    print(f"   Angle RMSE: {angle_rmse:.2f} degrees")
    print(f"   Frequency RMSE: {freq_rmse:.3f} Hz")
    
    # 2. Energy Preservation (KEY: use mean relative error)
    E_true = sys.energy_function(X_test)
    E_reconstructed = sys.energy_function(X_reconstructed)
    energy_errors = torch.abs(E_reconstructed - E_true) / (torch.abs(E_true) + 1e-10)
    mean_energy_error = energy_errors.mean().item()
    
    print(f"2. Energy Preservation:")
    print(f"   Mean relative energy error: {mean_energy_error:.1%}")
    
    # 3. Dynamic Trajectory Test
    x0 = x_eq + 0.02 * torch.randn_like(x_eq)
    dt = 0.001
    T = 3000
    
    # Full model
    x_full = x0.clone()
    traj_full = [x_full]
    for _ in range(T):
        f = sys._f(x_full.unsqueeze(0), sys.nominal_params).squeeze()
        x_full = x_full + dt * f
        traj_full.append(x_full.clone())
    traj_full = torch.stack(traj_full)
    
    # Reduced model
    z = reducer.forward(x0.unsqueeze(0))
    x_reduced = x0.clone()
    traj_reduced = [x_reduced]
    
    for _ in range(T):
        x_reduced = reducer.inverse(z).squeeze()
        f_full = sys._f(x_reduced.unsqueeze(0), sys.nominal_params)
        J = reducer.jacobian(x_reduced.unsqueeze(0))
        z_dot = torch.bmm(J, f_full).squeeze(0)
        z = z + dt * z_dot.T
        traj_reduced.append(x_reduced.clone())
    
    traj_reduced = torch.stack(traj_reduced)
    
    traj_errors = torch.norm(traj_full - traj_reduced, dim=1)
    mean_traj_error = traj_errors.mean().item()
    
    print(f"3. Dynamic Trajectory Test:")
    print(f"   Mean trajectory error: {mean_traj_error:.3f}")
    
    # 4. Frequency Response Test
    omega_pert = 0.1 * torch.randn(10)
    x_pert = x_eq.clone()
    x_pert[9:] += omega_pert
    
    z_eq = reducer.forward(x_eq.unsqueeze(0))
    z_pert = reducer.forward(x_pert.unsqueeze(0))
    
    x_eq_recon = reducer.inverse(z_eq).squeeze()
    x_pert_recon = reducer.inverse(z_pert).squeeze()
    
    omega_pert_recon = x_pert_recon[9:] - x_eq_recon[9:]
    pert_preservation = torch.norm(omega_pert_recon) / torch.norm(omega_pert)
    
    print(f"4. Frequency Response Test:")
    print(f"   Perturbation preservation: {pert_preservation:.1%}")
    
    # Overall score
    score = (
        10 * mean_energy_error +
        1 * angle_rmse +
        10 * freq_rmse +
        100 * mean_traj_error +
        10 * (1 - pert_preservation)
    )
    
    print(f"   Overall Score: {score:.3f} (lower is better)")
    
    return {
        'angle_rmse': angle_rmse,
        'freq_rmse': freq_rmse,
        'energy_error': mean_energy_error,
        'traj_error': mean_traj_error,
        'pert_preservation': pert_preservation,
        'score': score
    }


def main():
    """Main function for consistent comparison"""
    print("="*80)
    print("CONSISTENT VALIDATION - STANDARD vs IMPROVED REDUCERS")
    print("="*80)
    
    # Create system
    sys = create_test_power_system()
    print(f"System: {sys.n_machines}-generator network")
    print(f"States: {sys.n_dims} total")
    
    # Generate training data
    print("\nGenerating training data...")
    theta_eq = sys.delta_star[0] - sys.delta_star[1:]
    omega_eq = torch.zeros(10)
    x_eq = torch.cat([theta_eq, omega_eq])
    
    n_train = 1000
    X_train = []
    Xdot_train = []
    
    for i in range(n_train):
        x = x_eq + 0.01 * torch.randn_like(x_eq)
        X_train.append(x)
        
        x_tensor = x.unsqueeze(0)
        f = sys._f(x_tensor, sys.nominal_params).squeeze()
        Xdot_train.append(f)
    
    X_train = torch.stack(X_train)
    Xdot_train = torch.stack(Xdot_train)
    print(f"Training data: {n_train} snapshots")
    
    # Setup
    V_fn = sys.energy_function
    V_min = V_fn(X_train).min().item()
    
    reducers = {}
    
    # ===== STANDARD REDUCERS =====
    print("\n" + "="*40)
    print("Creating STANDARD reducers...")
    print("="*40)
    
    # Standard SPR
    try:
        A, J, R = sys.linearise(return_JR=True)
        spr = SymplecticProjectionReducer(A, J, R, latent_dim=6)
        spr.full_dim = sys.n_dims
        reducers['SPR-Standard'] = spr
        print("✓ Standard SPR created")
    except Exception as e:
        print(f"✗ Standard SPR failed: {e}")
    
    # Standard LCR
    try:
        lcr = LyapCoherencyReducer(sys, n_groups=3, snaps=X_train, λ=0.7)
        lcr.full_dim = sys.n_dims
        lcr.gamma = lcr.compute_gamma(V_min)
        reducers['LCR-Standard'] = lcr
        print("✓ Standard LCR created")
    except Exception as e:
        print(f"✗ Standard LCR failed: {e}")
    
    # Standard OpInf
    try:
        opinf = OpInfReducer(latent_dim=6, n_full=sys.n_dims, n_controls=1)
        opinf.sys = sys
        opinf.fit(X_train, Xdot_train, V_fn, V_min)
        reducers['OpInf-Standard'] = opinf
        print("✓ Standard OpInf created")
    except Exception as e:
        print(f"✗ Standard OpInf failed: {e}")
    
    # ===== IMPROVED REDUCERS =====
    print("\n" + "="*40)
    print("Creating IMPROVED reducers...")
    print("="*40)
    
    # Improved SPR
    try:
        spr_imp = ImprovedSymplecticProjectionReducer(A, J, R, latent_dim=6, use_data=True)
        spr_imp.enhance_with_data(X_train)
        spr_imp.full_dim = sys.n_dims
        reducers['SPR-Improved'] = spr_imp
        print("✓ Improved SPR created")
    except Exception as e:
        print(f"✗ Improved SPR failed: {e}")
    
    # Improved LCR
    try:
        lcr_imp = ImprovedLyapCoherencyReducer(
            sys, n_groups=3, snaps=X_train, λ=0.7,
            fuzzy_membership=True, energy_weight=0.3
        )
        lcr_imp.full_dim = sys.n_dims
        lcr_imp.gamma = lcr_imp.compute_gamma(V_min)
        reducers['LCR-Improved'] = lcr_imp
        print("✓ Improved LCR created")
    except Exception as e:
        print(f"✗ Improved LCR failed: {e}")
    
    # Improved OpInf
    try:
        opinf_imp = ImprovedOpInfReducer(
            latent_dim=6, n_full=sys.n_dims, n_controls=1,
            physics_informed=True, stability_margin=0.5
        )
        opinf_imp.sys = sys
        opinf_imp.fit(X_train, Xdot_train, V_fn, V_min)
        reducers['OpInf-Improved'] = opinf_imp
        print("✓ Improved OpInf created")
    except Exception as e:
        print(f"✗ Improved OpInf failed: {e}")
    
    # ===== EVALUATE ALL REDUCERS WITH SAME METHODOLOGY =====
    print("\n" + "="*80)
    print("EVALUATION - Using Consistent Methodology")
    print("="*80)
    
    results = {}
    for name, reducer in reducers.items():
        results[name] = evaluate_reducer_consistent(sys, reducer, test_data_size=1000)
    
    # ===== COMPARISON TABLE =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by type for better comparison
    standard_results = {k: v for k, v in results.items() if 'Standard' in k}
    improved_results = {k: v for k, v in results.items() if 'Improved' in k}
    
    print(f"\n{'Reducer':<18} {'Angle':<8} {'Freq':<8} {'Energy':<10} {'Traj':<8} {'Pert':<8} {'Score':<8}")
    print(f"{'':<18} {'(deg)':<8} {'(Hz)':<8} {'(%)':<10} {'Error':<8} {'(%)':<8} {'':<8}")
    print("-" * 90)
    
    # Print standard first
    for name, res in sorted(standard_results.items()):
        print(f"{name:<18} {res['angle_rmse']:<8.2f} {res['freq_rmse']:<8.3f} "
              f"{res['energy_error']*100:<10.1f} {res['traj_error']:<8.3f} "
              f"{res['pert_preservation']*100:<8.0f} {res['score']:<8.2f}")
    
    print("-" * 90)
    
    # Print improved
    for name, res in sorted(improved_results.items()):
        print(f"{name:<18} {res['angle_rmse']:<8.2f} {res['freq_rmse']:<8.3f} "
              f"{res['energy_error']*100:<10.1f} {res['traj_error']:<8.3f} "
              f"{res['pert_preservation']*100:<8.0f} {res['score']:<8.2f}")
    
    # ===== IMPROVEMENT ANALYSIS =====
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    for base in ['SPR', 'LCR', 'OpInf']:
        std_key = f"{base}-Standard"
        imp_key = f"{base}-Improved"
        
        if std_key in results and imp_key in results:
            std = results[std_key]
            imp = results[imp_key]
            
            print(f"\n{base} Improvements:")
            print(f"  Energy Error: {std['energy_error']*100:.1f}% → {imp['energy_error']*100:.1f}% "
                  f"({'↓' if imp['energy_error'] < std['energy_error'] else '↑'} "
                  f"{abs(imp['energy_error'] - std['energy_error'])*100:.1f}%)")
            print(f"  Trajectory Error: {std['traj_error']:.3f} → {imp['traj_error']:.3f} "
                  f"({'↓' if imp['traj_error'] < std['traj_error'] else '↑'} "
                  f"{abs(imp['traj_error'] - std['traj_error']):.3f})")
            print(f"  Overall Score: {std['score']:.2f} → {imp['score']:.2f} "
                  f"({'↓' if imp['score'] < std['score'] else '↑'} "
                  f"{abs(imp['score'] - std['score']):.2f})")
    
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = main()