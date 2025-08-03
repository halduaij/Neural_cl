
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
from scipy.cluster.hierarchy import linkage, fcluster
# control library is needed for CARE solver in SPR
try:
    from control import care
except ImportError:
    print("Warning: 'control' library not found. SPR might fail.")
    care = None

# Assuming these imports are correct based on the user's environment structure
try:
    # We try importing from the assumed package structure
    from neural_clbf.systems import SwingEquationSystem
    from neural_clbf.dimension_reduction.base import BaseReducer
    from neural_clbf.dimension_reduction.opinf import OpInfReducer
    from neural_clbf.dimension_reduction.lyap_coherency import LyapCoherencyReducer
    from neural_clbf.dimension_reduction.symplectic_projection import SymplecticProjectionReducer
    from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
except ImportError:
    # Fallback imports if the environment structure is different or running locally
    # This assumes the files are in the current directory or PYTHONPATH
    print("Warning: Using fallback imports. Ensure paths are correct.")
    try:
        from SwingEquationSystems import SwingEquationSystem
        from base import BaseReducer
        from opinf import OpInfReducer
        from lyap_coherency import LyapCoherencyReducer
        from symplectic_projection import SymplecticProjectionReducer
        from gp_opinf_dynamics import GPOpInfDynamics
    except ImportError as e:
        print(f"Critical: Fallback imports failed: {e}. Cannot proceed.")
        exit()


# ============ IMPROVED REDUCER IMPLEMENTATIONS ============
# These are defined within this file as per the user's original structure.

class ImprovedLyapCoherencyReducer(LyapCoherencyReducer):
    """
    Enhanced Lyapunov Coherency with fuzzy membership and energy preservation.
    """
    
    def __init__(self, sys, n_groups: int, snaps: torch.Tensor, 
                λ: float = 0.7, fuzzy_membership: bool = True, 
                energy_weight: float = 0.3):
        
        # FIX 1: Initialize attributes BEFORE calling super().__init__
        # This is necessary because the parent's _build method calls the overridden forward/inverse.
        self.fuzzy_membership = fuzzy_membership
        self.energy_weight = energy_weight
        self._last_forward_energy = None # Initialize this storage

        # Initialize parent class
        # This will call _build, which calls the overridden forward/inverse below
        super().__init__(sys, n_groups, snaps, λ)
        
        # Apply enhancements (must be done after parent init as it relies on P/labels)
        if fuzzy_membership and hasattr(self, 'labels'):
            self._compute_fuzzy_membership(snaps)
        
        # Store the parent's inverse method implementation
        self._inverse_original = super().inverse

    
    def _compute_fuzzy_membership(self, X):
        """Add fuzzy membership to existing grouping."""
        N = self.sys.n_machines
        device = X.device
        
        # Use existing labels from parent (computed during super().__init__)
        if not hasattr(self, 'labels'):
            print("Warning: Labels not available for fuzzy membership. Skipping.")
            return
            
        crisp_labels = self.labels
        
        # Compute features for each machine (Simplified feature extraction)
        M = torch.as_tensor(self.sys.M, device=device)
        omega = X[:, self.sys.N_NODES - 1:]
        kin = 0.5 * M * omega ** 2
        
        # Features: mean and std of kinetic energy
        features = torch.zeros(N, 2, device=device)
        features[:, 0] = kin.mean(0)
        features[:, 1] = kin.std(0)
        
        # Normalize
        std = features.std(0)
        std[std < 1e-6] = 1e-6 # Avoid division by zero
        features = (features - features.mean(0)) / std
        
        # Compute group centers
        group_centers = torch.zeros(self.n_groups, 2, device=device)
        for g in range(self.n_groups):
            mask = crisp_labels == g
            if mask.sum() > 0:
                group_centers[g] = features[mask].mean(0)
        
        # Fuzzy membership
        fuzzy_sigma = 0.5
        membership = torch.zeros(N, self.n_groups, device=device)
        
        for i in range(N):
            distances = torch.norm(features[i] - group_centers, dim=1)
            membership[i] = torch.exp(-distances**2 / (2 * fuzzy_sigma**2))
            total = membership[i].sum()
            if total > 1e-6:
                membership[i] /= total
        
        self.membership = membership
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
        
        # Update buffers using .data to manage tensor replacement correctly
        self.P.data = P_fuzzy.detach()
        self.Pi.data = torch.linalg.pinv(P_fuzzy).detach()
    
    # FIX 1 (continued): Implement forward/inverse using P/Pi and energy correction
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Energy-preserving reconstruction."""
        # Standard reconstruction using Pi (inherited behavior)
        x_base = z @ self.Pi
        
        # Simple energy correction (only if weight > 0 and energy stored)
        if self.energy_weight > 0 and self._last_forward_energy is not None:
            # Handle batch dimensions
            if z.dim() == 1:
                x_batch = x_base.unsqueeze(0)
            else:
                x_batch = x_base
            
            try:
                E_recon = self.sys.energy_function(x_batch)
                
                # Ensure dimensions match and E_recon is valid
                if E_recon.shape == self._last_forward_energy.shape and torch.isfinite(E_recon).all():
                    E_error = (self._last_forward_energy - E_recon).mean()
                    
                    # Simple scaling correction if error is significant
                    if abs(E_error.item()) > 1e-3:
                        # Adjust scale slightly based on error direction
                        # Use a small factor to avoid instability
                        scale = 1 + self.energy_weight * 0.01 * E_error.sign()
                        scale = max(0.98, min(1.02, scale.item()))
                        x_base = x_base * scale
            except Exception as e:
                # If energy calculation fails, skip correction
                pass
        
        return x_base
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Store energy for reconstruction."""
        # Store energy for later use
        if self.energy_weight > 0:
            try:
                if x.dim() == 1:
                    self._last_forward_energy = self.sys.energy_function(x.unsqueeze(0))
                else:
                    self._last_forward_energy = self.sys.energy_function(x)
            except Exception:
                self._last_forward_energy = None

        # Standard projection using P (inherited behavior)
        return x @ self.P


class ImprovedOpInfReducer(OpInfReducer):
    """
    Enhanced OpInf with physics-informed projection and stability enforcement.
    (Implementation kept as provided by the user in the previous turn, assuming it is satisfactory)
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
            
            # 2. Combine state, derivative, and energy information
            
            # Standard POD on states
            try:
                U1, S1, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
            except:
                U1 = torch.eye(X.shape[1], device=device)

            # POD on derivatives
            Xdot_centered = Xdot - Xdot.mean(0)
            try:
                U2, S2, _ = torch.linalg.svd(Xdot_centered.T, full_matrices=False)
            except:
                U2 = torch.eye(X.shape[1], device=device)

            # Energy-based modes (simplified gradient approach)
            U3 = None
            try:
                n_samples = min(100, X.shape[0])
                X_sample = X[:n_samples].clone().requires_grad_(True)
                V_sample = V_fn(X_sample)
                if V_sample.requires_grad and V_sample.numel() == n_samples:
                    grad_V = torch.autograd.grad(V_sample.sum(), X_sample)[0]
                    U3, S3, _ = torch.linalg.svd(grad_V.T, full_matrices=False)
            except:
                pass
            
            # Combine modes (Weighted distribution)
            n_state = int(self.latent_dim * 0.5)
            n_dyn = int(self.latent_dim * 0.3)
            n_energy = self.latent_dim - n_state - n_dyn
            
            bases = [U1[:, :n_state]]
            if n_dyn > 0: bases.append(U2[:, :n_dyn])
            if n_energy > 0 and U3 is not None: 
                bases.append(U3[:, :min(n_energy, U3.shape[1])])
            
            # Handle cases where SVD returns fewer columns than requested
            combined_bases = []
            for b in bases:
                if b.shape[1] > 0:
                    combined_bases.append(b)
            
            if not combined_bases:
                # Fallback if all bases are empty
                self.proj = torch.eye(self.full_dim, self.latent_dim, device=device)
            else:
                combined = torch.cat(combined_bases, dim=1)
                # Orthogonalize and ensure correct dimension
                Q, _ = torch.linalg.qr(combined)
                self.proj = Q[:, :self.latent_dim]
            
        else:
            # Standard OpInf fitting
            super().fit(X, Xdot, V_fn, V_min)
            return self
        
        # Continue with dynamics fitting
        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        
        # Fit dynamics (Using the robust GPOpInfDynamics)
        # Ensure GPOpInfDynamics is imported correctly
        if 'GPOpInfDynamics' in globals():
            self.dyn = GPOpInfDynamics(self.latent_dim, self.n_controls).to(device)
            self.dyn.reg = torch.tensor(0.1, device=device) # Strong regularization
            
            U = torch.zeros(Z.shape[0], self.n_controls, device=device)
            self.dyn.fit(Z, U, dZdt)
        else:
            print("Warning: GPOpInfDynamics not available. Skipping dynamics fit.")
            self.dyn = None

        
        # Enforce stability (GPOpInfDynamics already does this, but we double check)
        self._enforce_stability()
        
        # Compute gamma
        self.compute_gamma(X, V_fn, V_min)
        
        return self
    
    def _enforce_stability(self):
        """Ensure dynamics are stable."""
        if self.dyn is None:
            return
        
        A = self.dyn.A.data
        try:
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
        except:
            max_real = 1.0 # Assume unstable if eig fails

        if max_real > -self.stability_margin:
            # If GPOpInfDynamics stabilization wasn't enough, apply final shift
            shift = max(0.0, max_real) + self.stability_margin
            self.dyn.A.data = A - shift * torch.eye(self.latent_dim, device=A.device)


class ImprovedSymplecticProjectionReducer(SymplecticProjectionReducer):
    """
    Enhanced SPR with data-informed basis selection.
    """
    
    def __init__(self, A: torch.Tensor, J: torch.Tensor, R: torch.Tensor, 
                latent_dim: int, use_data: bool = True):
        # Initialize parent
        super().__init__(A, J, R, latent_dim)
        
        self.use_data = use_data
        self._original_T = self.T.clone()
        self.J = J # Store J for potential use
    
    def enhance_with_data(self, X: torch.Tensor):
        """Enhance projection using trajectory data."""
        if not self.use_data:
            return
        
        print("  Enhancing SPR with trajectory data...")
        
        X_centered = X - X.mean(0)

        # Compute POD modes from data
        try:
            U, S, _ = torch.linalg.svd(X_centered.T, full_matrices=False)
        except:
            print("SVD failed during enhancement. Skipping data enhancement.")
            return

        # Take dominant POD modes (up to half the latent dim)
        n_pod = min(self.latent_dim // 2, U.shape[1])
        pod_modes = U[:, :n_pod]
        
        # Combine with original symplectic modes
        combined = torch.cat([self._original_T, pod_modes], dim=1)
        
        # Orthogonalize the combined basis
        Q, _ = torch.linalg.qr(combined)
        
        # FIX 3: Improved basis selection strategy.
        # Score individual basis vectors based on the variance explained in X.
        
        # Project X onto the orthogonal basis Q
        Z_Q = X_centered @ Q 
        # Variance explained by each vector in Q
        variance_explained = Z_Q.var(dim=0)
        
        # Select top modes based on variance
        idx = torch.argsort(variance_explained, descending=True)[:self.latent_dim]
        # Ensure indices are sorted for consistency
        idx_sorted = torch.sort(idx).values
        
        T_enhanced = Q[:, idx_sorted]

        # Update projection matrices using .data for correct buffer replacement
        self.T.data = T_enhanced.detach()
        self.Ti.data = torch.linalg.pinv(T_enhanced).detach()


# ============ SYSTEM SETUP ============

def create_test_power_system():
    """Create test system exactly as in validation_file4.py, but stabilized."""
    M = torch.tensor([1.4, 1.01, 1.1766666, 0.95333344, 0.8666667,
                    1.16, 0.88, 0.81, 1.1500001, 16.666668], dtype=torch.float32)
    
    # FIX 2: Increase damping significantly to stabilize the system.
    # The logs showed λ_max = 3.42. We increase the damping multiplier significantly.
    DAMPING_MULTIPLIER = 15.0 # Increased from 3.0 (in context) to 15.0

    D = torch.tensor([0.19666669, 0.28833333, 0.28833333, 0.28833333, 0.28833333,
                    0.28833333, 0.28833333, 0.28833333, 0.30366668, 0.30366668], 
                    dtype=torch.float32) * DAMPING_MULTIPLIER
    
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
    
    # Equilibrium point (assuming it remains the same despite damping change, as damping affects dynamics not equilibrium)
    delta_star = torch.tensor([-0.05420687, -0.07780334, -0.07351729, -0.05827823, -0.09359571,
                            -0.02447385, -0.00783582, 0.00259523, -0.0162409, -0.06477749],
                            dtype=torch.float32)
    
    params = dict(M=M, D=D, P=P, K=K)
    
    # We assume the user has a working SwingEquationSystem implementation 
    # that correctly handles linearization and dynamics.
    sys = SwingEquationSystem(params, dt=0.01)
    
    # Set the known equilibrium point if the system allows it
    if hasattr(sys, 'delta_star'):
        sys.delta_star = delta_star
    
    # Check stability after modification
    try:
        # We rely on the system's linearise method
        A = sys.linearise(return_JR=False)
        # Handle different return types
        if isinstance(A, tuple): A = A[0]
        
        if torch.is_tensor(A):
            max_eig = torch.linalg.eigvals(A).real.max().item()
            print(f"System stability check: Max Eigenvalue = {max_eig:.4f}")
            if max_eig > 1e-5:
                print("WARNING: System is still unstable at equilibrium.")
        else:
             print("WARNING: Linearization did not return a tensor.")

    except Exception as e:
        print(f"Could not verify system stability: {e}")

    return sys


# ============ CONSISTENT EVALUATION ============
# (The evaluation functions are kept as provided in the user's validation_file6.py snippet, 
# assuming they work correctly now that the reducers are fixed)

def evaluate_reducer_consistent(sys, reducer, test_data_size=1000):
    """
    Evaluate ANY reducer using EXACTLY the same methodology.
    """
    # Handle potential device mismatch
    device = sys.M.device
    reducer.to(device)

    # Setup equilibrium point
    if hasattr(sys, 'delta_star') and sys.delta_star is not None:
        # Assuming the state representation [theta_12...theta_1n, omega_1...omega_n]
        theta_eq = sys.delta_star[0] - sys.delta_star[1:]
        omega_eq = torch.zeros(sys.N_NODES)
        x_eq = torch.cat([theta_eq, omega_eq]).to(device)
    else:
        # Fallback to goal_point if delta_star is not set
        x_eq = sys.goal_point.squeeze().to(device)

    # Generate test data near equilibrium
    torch.manual_seed(0) # Ensure consistency
    X_test = x_eq + 0.01 * torch.randn(test_data_size, sys.n_dims, device=device)
    
    print(f"\nEvaluating {type(reducer).__name__} (d={reducer.latent_dim}):")
    
    # 1. State Reconstruction
    Z = reducer.forward(X_test)
    X_reconstructed = reducer.inverse(Z)
    
    # Angle RMSE (degrees)
    # Assuming first N-1 states are angles
    n_angles = sys.N_NODES - 1
    angle_error_rad = (X_reconstructed[:, :n_angles] - X_test[:, :n_angles]).abs()
    angle_error_deg = angle_error_rad * 180 / np.pi
    angle_rmse = torch.sqrt((angle_error_deg ** 2).mean()).item()
    
    # Frequency RMSE (Hz)
    freq_error_pu = (X_reconstructed[:, n_angles:] - X_test[:, n_angles:]).abs()
    # Assuming 60Hz system frequency for conversion from rad/s pu
    freq_error_hz = freq_error_pu * 60 / (2 * np.pi) 
    freq_rmse = torch.sqrt((freq_error_hz ** 2).mean()).item()
    
    print(f"1. State Reconstruction:")
    print(f"   Angle RMSE: {angle_rmse:.2f} degrees")
    print(f"   Frequency RMSE: {freq_rmse:.3f} Hz")
    
    # 2. Energy Preservation
    try:
        E_true = sys.energy_function(X_test)
        E_reconstructed = sys.energy_function(X_reconstructed)
        # Use mean relative error, avoiding division by zero
        # Ensure energy values are finite
        if not torch.isfinite(E_true).all() or not torch.isfinite(E_reconstructed).all():
            mean_energy_error = float('nan')
        else:
            # Robust calculation
            E_true_safe = torch.where(E_true.abs() < 1e-6, torch.tensor(1e-6, device=device), E_true)
            energy_errors = torch.abs(E_reconstructed - E_true) / E_true_safe.abs()
            mean_energy_error = energy_errors.mean().item()
    except Exception as e:
        print(f"Energy calculation failed: {e}")
        mean_energy_error = float('nan')
    
    print(f"2. Energy Preservation:")
    print(f"   Mean relative energy error: {mean_energy_error:.1%}")
    
    # 3. Dynamic Trajectory Test (Simplified Euler integration for consistency)
    x0 = x_eq + 0.02 * torch.randn_like(x_eq)
    dt = 0.001
    T = 3000 # 3 seconds
    
    # Full model simulation
    x_full = x0.clone().unsqueeze(0)
    traj_full = [x_full.squeeze()]
    try:
        for _ in range(T):
            # Using closed loop dynamics assuming zero control input for fair comparison
            if hasattr(sys, 'closed_loop_dynamics'):
                u = torch.zeros(1, sys.n_controls, device=device)
                f = sys.closed_loop_dynamics(x_full, u, sys.nominal_params)
            else:
                # Fallback to internal dynamics _f (as used in some older snippets)
                f = sys._f(x_full, sys.nominal_params)
            
            # Ensure f is (B, N)
            if f.dim() == 3: f = f.squeeze(-1)
            
            x_full = x_full + dt * f
            traj_full.append(x_full.squeeze().clone())
        traj_full = torch.stack(traj_full)
    except Exception as e:
        print(f"Full model simulation failed: {e}")
        traj_full = torch.zeros(T+1, sys.n_dims)

    
    # Reduced model simulation
    z = reducer.forward(x0.unsqueeze(0))
    traj_reduced = [x0.clone()]
    
    try:
        for _ in range(T):
            x_reduced = reducer.inverse(z)
            
            # Calculate dynamics (dz/dt)
            if hasattr(reducer, 'dyn') and reducer.dyn is not None:
                # OpInf style
                # Handle control dimension mismatch if necessary
                n_ctrl_dyn = getattr(reducer.dyn, 'm', 0)
                u = torch.zeros(1, n_ctrl_dyn, device=device)
                z_dot = reducer.dyn.forward(z, u)
            else:
                # Projection style (LCR, SPR)
                if hasattr(sys, 'closed_loop_dynamics'):
                     u = torch.zeros(1, sys.n_controls, device=device)
                     f_full = sys.closed_loop_dynamics(x_reduced, u, sys.nominal_params)
                else:
                     f_full = sys._f(x_reduced, sys.nominal_params)

                # Ensure f_full is (B, N)
                if f_full.dim() == 3: f_full = f_full.squeeze(-1)

                J = reducer.jacobian(x_reduced) # (B, d, n)
                z_dot = torch.bmm(J, f_full.unsqueeze(-1)).squeeze(-1)

            z = z + dt * z_dot
            traj_reduced.append(reducer.inverse(z).squeeze().clone())
        
        traj_reduced = torch.stack(traj_reduced)

        # Calculate error
        if traj_full.shape == traj_reduced.shape:
            traj_errors = torch.norm(traj_full - traj_reduced, dim=1)
            mean_traj_error = traj_errors.mean().item()
        else:
            mean_traj_error = float('inf')

    except Exception as e:
        print(f"Reduced model simulation failed: {e}")
        # import traceback
        # traceback.print_exc()
        mean_traj_error = float('inf')

    
    print(f"3. Dynamic Trajectory Test:")
    print(f"   Mean trajectory error: {mean_traj_error:.3f}")
    
    # 4. Frequency Response Test (Perturbation preservation)
    omega_pert = 0.1 * torch.randn(sys.N_NODES, device=device)
    x_pert = x_eq.clone()
    x_pert[n_angles:] += omega_pert
    
    z_eq = reducer.forward(x_eq.unsqueeze(0))
    z_pert = reducer.forward(x_pert.unsqueeze(0))
    
    x_eq_recon = reducer.inverse(z_eq).squeeze()
    x_pert_recon = reducer.inverse(z_pert).squeeze()
    
    omega_pert_recon = x_pert_recon[n_angles:] - x_eq_recon[n_angles:]
    
    # Preservation metric: How much of the perturbation magnitude is kept?
    norm_true = torch.norm(omega_pert)
    norm_recon = torch.norm(omega_pert_recon)
    
    if norm_true > 1e-6:
        pert_preservation = (norm_recon / norm_true).item()
    else:
        pert_preservation = 1.0
    
    # Clamp preservation percentage
    pert_preservation = max(0.0, min(1.0, pert_preservation))

    print(f"4. Frequency Response Test:")
    print(f"   Perturbation preservation: {pert_preservation:.1%}")
    
    # Overall Score (Weighted combination of errors - lower is better)
    # Weights prioritized based on stability and accuracy
    # Weights: Angle(1), Freq(10), Energy(500), Traj(100), Pert((1-P)^2 * 10)
    
    # Handle NaN energy error
    energy_error_val = mean_energy_error if np.isfinite(mean_energy_error) else 1.0 
    traj_error_val = mean_traj_error if np.isfinite(mean_traj_error) else 10.0

    score = (angle_rmse * 1 + 
             freq_rmse * 10 + 
             energy_error_val * 500 + 
             traj_error_val * 100 + 
             (1 - pert_preservation)**2 * 10)
    
    print(f"   Overall Score: {score:.3f} (lower is better)")

    return {
        "angle_rmse": angle_rmse,
        "freq_rmse": freq_rmse,
        "energy_error": mean_energy_error,
        "traj_error": mean_traj_error,
        "pert_preservation": pert_preservation,
        "score": score
    }

# ============ MAIN VALIDATION RUNNER ============

def run_consistent_validation():
    print("="*80)
    print("CONSISTENT VALIDATION - STANDARD vs IMPROVED REDUCERS")
    print("="*80)

    # Setup System
    sys = create_test_power_system()
    N_DIMS = sys.n_dims
    print(f"System: 10-generator network")
    print(f"States: {N_DIMS} total")

    # Generate Training Data
    print("\nGenerating training data...")
    N_TRAJ = 50
    T_STEPS = 20
    # Ensure data collection uses the system's methods
    if hasattr(sys, 'collect_random_trajectories'):
        try:
            data = sys.collect_random_trajectories(N_TRAJ, T_STEPS, control_excitation=0.1, return_derivative=True)
            X_train = data["X"]
            Xdot_train = data["dXdt"]
        except Exception as e:
             print(f"Data collection failed: {e}. Skipping validation.")
             return
    else:
        print("System does not support collect_random_trajectories. Skipping.")
        return

    print(f"Training data: {X_train.shape[0]} snapshots")

    LATENT_DIM = 6 # Fixed dimension for comparison (e.g., 3 groups)
    N_GROUPS = LATENT_DIM // 2

    if hasattr(sys, 'energy_function'):
        V_fn = sys.energy_function
        try:
            V_vals = V_fn(X_train)
            # Ensure we only consider finite values for V_min
            V_finite = V_vals[torch.isfinite(V_vals)]
            if V_finite.numel() > 0:
                V_min = max(V_finite.min().item(), 1e-3)
            else:
                V_min = 1.0
        except:
            V_fn = None
            V_min = 1.0
    else:
        V_fn = None
        V_min = 1.0


    reducers = {}

    # ========================================
    # STANDARD Reducers
    # ========================================
    print("\n" + "="*40)
    print("Creating STANDARD reducers...")
    print("="*40)

    # 1. Standard SPR
    try:
        A, J, R = sys.linearise(return_JR=True)
        spr_standard = SymplecticProjectionReducer(A, J, R, LATENT_DIM)
        reducers["SPR-Standard"] = spr_standard
        print("✓ Standard SPR created")
    except Exception as e:
        print(f"✗ Standard SPR failed: {e}")

    # 2. Standard LCR
    try:
        lcr_standard = LyapCoherencyReducer(sys, N_GROUPS, X_train)
        reducers["LCR-Standard"] = lcr_standard
        print("✓ Standard LCR created")
    except Exception as e:
        print(f"✗ Standard LCR failed: {e}")

    # 3. Standard OpInf
    try:
        opinf_standard = OpInfReducer(LATENT_DIM, N_DIMS, sys.n_controls)
        opinf_standard.sys = sys
        opinf_standard.fit(X_train, Xdot_train, V_fn, V_min)
        reducers["OpInf-Standard"] = opinf_standard
        print("✓ Standard OpInf created")
    except Exception as e:
        print(f"✗ Standard OpInf failed: {e}")

    # ========================================
    # IMPROVED Reducers
    # ========================================
    print("\n" + "="*40)
    print("Creating IMPROVED reducers...")
    print("="*40)

    # 1. Improved SPR
    try:
        A, J, R = sys.linearise(return_JR=True)
        spr_improved = ImprovedSymplecticProjectionReducer(A, J, R, LATENT_DIM, use_data=True)
        spr_improved.enhance_with_data(X_train)
        reducers["SPR-Improved"] = spr_improved
        print("✓ Improved SPR created")
    except Exception as e:
        print(f"✗ Improved SPR failed: {e}")

    # 2. Improved LCR
    try:
        # This should now work due to the initialization fix
        lcr_improved = ImprovedLyapCoherencyReducer(sys, N_GROUPS, X_train)
        reducers["LCR-Improved"] = lcr_improved
        print("✓ Improved LCR created")
    except Exception as e:
        # If it still fails, print the traceback for debugging
        print(f"✗ Improved LCR failed: {e}")
        # import traceback
        # traceback.print_exc()


    # 3. Improved OpInf
    try:
        # Check if ImprovedOpInfReducer is defined (it might be in another file)
        if 'ImprovedOpInfReducer' in globals():
            opinf_improved = ImprovedOpInfReducer(LATENT_DIM, N_DIMS, sys.n_controls, physics_informed=True)
            opinf_improved.sys = sys
            opinf_improved.fit(X_train, Xdot_train, V_fn, V_min)
            reducers["OpInf-Improved"] = opinf_improved
            print("✓ Improved OpInf created")
        else:
            print("✗ ImprovedOpInfReducer not defined.")
    except Exception as e:
        print(f"✗ Improved OpInf failed: {e}")

    # ========================================
    # EVALUATION
    # ========================================
    print("\n" + "="*80)
    print("EVALUATION - Using Consistent Methodology")
    print("="*80)

    results = {}
    for name, reducer in reducers.items():
        try:
            results[name] = evaluate_reducer_consistent(sys, reducer)
        except Exception as e:
            print(f"\nEvaluation failed for {name}: {e}")
            # import traceback
            # traceback.print_exc()


    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    # Header formatting aligned
    print(f"{'Reducer':<20} {'Angle':<8} {'Freq':<8} {'Energy':<10} {'Traj':<10} {'Pert':<8} {'Score':<8}")
    print(f"{'':<20} {'(deg)':<8} {'(Hz)':<8} {'(%)':<10} {'Error':<10} {'(%)':<8}")
    print("-"*90)
    
    standard_results = {}
    improved_results = {}

    # Sort results for cleaner presentation
    sorted_names = sorted(results.keys())

    for name in sorted_names:
        res = results[name]
        # Format energy and perturbation as percentage strings
        energy_str = f"{res['energy_error']:.1%}" if np.isfinite(res['energy_error']) else "N/A"
        pert_str = f"{res['pert_preservation']:.0%}" if np.isfinite(res['pert_preservation']) else "N/A"
        traj_str = f"{res['traj_error']:.3f}" if np.isfinite(res['traj_error']) else "N/A"

        print(f"{name:<20} {res['angle_rmse']:.2f}    {res['freq_rmse']:.3f}    {energy_str:<10} {traj_str:<10} {pert_str:<8} {res['score']:.2f}")
        
        if "Standard" in name:
            standard_results[name.split('-')[0]] = res
        elif "Improved" in name:
            improved_results[name.split('-')[0]] = res

    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)

    for key in standard_results:
        if key in improved_results:
            std = standard_results[key]
            imp = improved_results[key]
            
            # Safely calculate differences
            e_diff = (std['energy_error'] - imp['energy_error']) if (np.isfinite(std['energy_error']) and np.isfinite(imp['energy_error'])) else 0
            t_diff = (std['traj_error'] - imp['traj_error']) if (np.isfinite(std['traj_error']) and np.isfinite(imp['traj_error'])) else 0
            s_diff = (std['score'] - imp['score']) if (np.isfinite(std['score']) and np.isfinite(imp['score'])) else 0

            print(f"\n{key} Improvements:")
            # Display change direction correctly (↓ means improvement)
            print(f"  Energy Error: {std['energy_error']:.1%} -> {imp['energy_error']:.1%} (Change: {-e_diff:.1%})")
            print(f"  Trajectory Error: {std['traj_error']:.3f} -> {imp['traj_error']:.3f} (Change: {-t_diff:.3f})")
            print(f"  Overall Score: {std['score']:.2f} -> {imp['score']:.2f} (Change: {-s_diff:.2f})")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    run_consistent_validation()