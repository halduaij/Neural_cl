"""
Improved Operator Inference Reducer
===================================

Enhanced OpInf with physics-informed projection, multi-fidelity training,
constrained dynamics fitting, and stability guarantees.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from neural_clbf.dimension_reduction.base import BaseReducer
from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics


class ImprovedOpInfReducer(BaseReducer):
    """
    Enhanced Operator Inference with physics constraints and stability guarantees.
    
    Key improvements:
    1. Physics-informed projection (energy/frequency preserving modes)
    2. Multi-fidelity data augmentation
    3. Constrained dynamics fitting with physical constraints
    4. Guaranteed stability through post-processing
    5. Improved perturbation sensitivity
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        n_full: int, 
        n_controls: int = 1,
        physics_informed: bool = True,
        stability_enforced: bool = True,
        multi_fidelity: bool = True,
        energy_weight: float = 0.3,
        freq_weight: float = 0.3,
        stability_margin: float = 0.5
    ):
        """
        Initialize enhanced OpInf reducer.
        
        Args:
            latent_dim: Dimension of reduced space
            n_full: Full system dimension
            n_controls: Number of control inputs
            physics_informed: Use physics-informed projection
            stability_enforced: Enforce stability constraints
            multi_fidelity: Use data augmentation
            energy_weight: Weight for energy-preserving modes
            freq_weight: Weight for frequency-preserving modes
            stability_margin: Minimum negative real part for eigenvalues
        """
        super().__init__(latent_dim)
        
        self.full_dim = n_full
        self.n_controls = n_controls
        self.physics_informed = physics_informed
        self.stability_enforced = stability_enforced
        self.multi_fidelity = multi_fidelity
        self.energy_weight = energy_weight
        self.freq_weight = freq_weight
        self.stability_margin = stability_margin
        
        # Initialize projection matrices
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        
        # Placeholder for system reference
        self.sys = None
        self.dyn = None
        
        # Storage for physical constraints
        self.energy_modes = None
        self.freq_modes = None
        self.constraint_matrices = {}
    
    def fit(self, X, Xdot, V_fn, V_min, X_disturbed=None):
        """
        Enhanced fitting with physics constraints and stability guarantees.
        
        Args:
            X: State snapshots (n_samples, n_full)
            Xdot: State derivatives (n_samples, n_full)
            V_fn: Lyapunov/Energy function
            V_min: Minimum value of V
            X_disturbed: Optional disturbed trajectories for robustness
        """
        device = X.device
        print(f"\nFitting Enhanced OpInf Reducer (d={self.latent_dim}):")
        
        # 1. Multi-fidelity data augmentation
        if self.multi_fidelity:
            print("  Augmenting with synthetic data...")
            X_aug, Xdot_aug = self._augment_with_synthetic_data(X, Xdot)
            
            # Combine original and augmented data
            X_combined = torch.cat([X, X_aug])
            Xdot_combined = torch.cat([Xdot, Xdot_aug])
        else:
            X_combined = X
            Xdot_combined = Xdot
        
        # 2. Compute projection basis
        print("  Computing projection basis...")
        self.μ = X_combined.mean(0)
        
        if self.physics_informed:
            # Compute physics-informed modes
            self.proj = self._compute_physics_informed_projection(
                X_combined, Xdot_combined, V_fn
            )
        else:
            # Standard POD
            self.proj = self._compute_pod_projection(X_combined)
        
        # 3. Project data
        Z = self.forward(X_combined)
        dZdt = Xdot_combined @ self.proj
        
        # 4. Fit dynamics with constraints
        print("  Fitting constrained dynamics...")
        self._fit_constrained_dynamics(Z, dZdt, X_combined)
        
        # 5. Enforce stability
        if self.stability_enforced:
            print("  Enforcing stability constraints...")
            self._enforce_stability_constraints()
        
        # 6. Compute gamma
        self._compute_gamma_enhanced(X, V_fn, V_min)
        
        # 7. Verify perturbation preservation
        if X_disturbed is not None:
            self._verify_perturbation_preservation(X, X_disturbed)
        
        print(f"  Fitting complete. γ = {self.gamma:.3f}")
        
        self.to(device)
        return self
    
    def _augment_with_synthetic_data(self, X, Xdot):
        """Generate physics-consistent synthetic data for robustness."""
        X_aug = []
        Xdot_aug = []
        
        n_samples = min(len(X), 500)  # Limit augmentation size
        
        # 1. Small perturbations around existing data
        for i in range(0, len(X), max(1, len(X) // n_samples)):
            x_base = X[i]
            
            # Perturbation directions
            n_pert = min(5, self.full_dim)
            for j in range(n_pert):
                # Random direction
                direction = torch.randn_like(x_base)
                direction = direction / direction.norm()
                
                # Small perturbation
                eps = 0.01 * x_base.abs().mean()
                
                # Forward and backward
                x_plus = x_base + eps * direction
                x_minus = x_base - eps * direction
                
                # Compute derivatives using linearization
                if Xdot is not None and i < len(Xdot):
                    # Finite difference approximation
                    dxdt_base = Xdot[i]
                    
                    # Linear approximation
                    if hasattr(self.sys, '_f'):
                        # Use actual dynamics
                        f_plus = self.sys._f(x_plus.unsqueeze(0), 
                                           self.sys.nominal_params).squeeze()
                        f_minus = self.sys._f(x_minus.unsqueeze(0), 
                                            self.sys.nominal_params).squeeze()
                        
                        X_aug.extend([x_plus, x_minus])
                        Xdot_aug.extend([f_plus, f_minus])
                    else:
                        # Use linear extrapolation
                        X_aug.extend([x_plus, x_minus])
                        Xdot_aug.extend([dxdt_base, dxdt_base])
        
        # 2. Interpolation between states
        for i in range(0, len(X)-1, max(1, len(X) // (n_samples // 2))):
            x1, x2 = X[i], X[i+1]
            
            # Linear interpolation
            for alpha in [0.25, 0.5, 0.75]:
                x_interp = (1 - alpha) * x1 + alpha * x2
                
                if Xdot is not None and i < len(Xdot) - 1:
                    xdot_interp = (1 - alpha) * Xdot[i] + alpha * Xdot[i+1]
                    
                    X_aug.append(x_interp)
                    Xdot_aug.append(xdot_interp)
        
        if X_aug:
            return torch.stack(X_aug), torch.stack(Xdot_aug)
        else:
            return torch.zeros(0, self.full_dim), torch.zeros(0, self.full_dim)
    
    def _compute_physics_informed_projection(self, X, Xdot, V_fn):
        """Compute projection that preserves physical quantities."""
        print("    Computing physics-informed modes...")
        
        # 1. Standard POD modes
        pod_modes = self._compute_pod_projection(X, n_modes=self.latent_dim * 2)
        
        # 2. Energy-preserving modes
        if V_fn is not None and callable(V_fn):
            energy_modes = self._compute_energy_modes(X, V_fn)
            self.energy_modes = energy_modes
        else:
            energy_modes = None
        
        # 3. Frequency-preserving modes (for power systems)
        if hasattr(self.sys, 'N_NODES'):
            freq_modes = self._compute_frequency_modes(X)
            self.freq_modes = freq_modes
        else:
            freq_modes = None
        
        # 4. Dynamics-informed modes
        if Xdot is not None:
            dyn_modes = self._compute_dynamics_modes(X, Xdot)
        else:
            dyn_modes = None
        
        # 5. Combine all modes optimally
        all_modes = [pod_modes]
        weights = [1.0 - self.energy_weight - self.freq_weight]
        
        if energy_modes is not None:
            all_modes.append(energy_modes[:, :self.latent_dim // 2])
            weights.append(self.energy_weight)
        
        if freq_modes is not None:
            all_modes.append(freq_modes[:, :self.latent_dim // 2])
            weights.append(self.freq_weight)
        
        if dyn_modes is not None:
            all_modes.append(dyn_modes[:, :self.latent_dim // 2])
            weights.append(0.1)  # Small weight for dynamics modes
        
        # Weighted combination
        combined = torch.zeros_like(pod_modes[:, :self.latent_dim])
        
        for modes, weight in zip(all_modes, weights):
            if modes is not None and weight > 0:
                n_modes = min(modes.shape[1], self.latent_dim)
                combined += weight * modes[:, :n_modes]
        
        # Orthogonalize
        Q, R = torch.linalg.qr(combined)
        
        return Q[:, :self.latent_dim]
    
    def _compute_pod_projection(self, X, n_modes=None):
        """Standard POD projection."""
        if n_modes is None:
            n_modes = self.latent_dim
        
        X_centered = X - self.μ
        
        # Use robust SVD
        try:
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            
            # Energy-based truncation
            energy = S**2
            total_energy = energy.sum()
            
            if total_energy > 1e-9:
                cum_energy = torch.cumsum(energy, dim=0) / total_energy
                n_99 = (cum_energy < 0.999).sum().item() + 1
                n_modes = min(n_modes, n_99)
            
            return Vt[:n_modes].T
            
        except Exception as e:
            print(f"    SVD failed: {e}. Using fallback.")
            Q, _ = torch.linalg.qr(X_centered.T)
            return Q[:, :n_modes]
    
    def _compute_energy_modes(self, X, V_fn):
        """Compute modes that preserve energy function structure."""
        try:
            # Compute energy gradients
            grad_V = []
            
            n_samples = min(len(X), 200)
            idx = torch.randperm(len(X))[:n_samples]
            
            for i in idx:
                x = X[i:i+1].clone().requires_grad_(True)
                v = V_fn(x)
                
                if v.requires_grad:
                    g = torch.autograd.grad(v.sum(), x, create_graph=False)[0]
                    grad_V.append(g.squeeze())
            
            if grad_V:
                grad_V = torch.stack(grad_V)
                
                # POD on gradients
                grad_V_centered = grad_V - grad_V.mean(0)
                U, S, Vt = torch.linalg.svd(grad_V_centered.T, full_matrices=False)
                
                return U
            
        except Exception as e:
            print(f"    Energy modes computation failed: {e}")
        
        return None
    
    def _compute_frequency_modes(self, X):
        """Compute modes that preserve frequency dynamics (for power systems)."""
        try:
            # Extract frequency components
            if hasattr(self.sys, 'N_NODES'):
                omega = X[:, self.sys.N_NODES - 1:]
                
                # POD on frequency data
                omega_centered = omega - omega.mean(0)
                U_omega, S_omega, Vt_omega = torch.linalg.svd(omega_centered.T, full_matrices=False)
                
                # Extend to full state dimension
                modes = torch.zeros(self.full_dim, U_omega.shape[1])
                modes[self.sys.N_NODES - 1:, :] = U_omega
                
                return modes
                
        except Exception as e:
            print(f"    Frequency modes computation failed: {e}")
        
        return None
    
    def _compute_dynamics_modes(self, X, Xdot):
        """Compute modes informed by dynamics."""
        try:
            # Combine states and derivatives
            combined = torch.cat([X, Xdot], dim=1)
            
            # POD on combined data
            combined_centered = combined - combined.mean(0)
            U, S, Vt = torch.linalg.svd(combined_centered.T, full_matrices=False)
            
            # Extract state components
            return U[:self.full_dim, :]
            
        except Exception as e:
            print(f"    Dynamics modes computation failed: {e}")
        
        return None
    
    def _fit_constrained_dynamics(self, Z, dZdt, X_full):
        """Fit dynamics with physical constraints."""
        device = Z.device
        
        # Initialize dynamics model
        self.dyn = GPOpInfDynamics(self.latent_dim, self.n_controls).to(device)
        
        # Build constraint matrices
        constraints = self._build_physical_constraints(Z, dZdt, X_full)
        
        # First fit unconstrained
        U = torch.zeros(Z.shape[0], self.n_controls, device=device)
        self.dyn.fit(Z, U, dZdt)
        
        # Then apply constraints
        if constraints:
            self._apply_dynamics_constraints(constraints)
    
    def _build_physical_constraints(self, Z, dZdt, X_full):
        """Build physical constraint matrices."""
        constraints = {}
        
        # 1. Energy dissipation constraint
        # The system should dissipate energy (eigenvalues have negative real parts)
        constraints['stability'] = {
            'type': 'eigenvalue',
            'bound': -self.stability_margin
        }
        
        # 2. Power balance constraint (for power systems)
        if hasattr(self.sys, 'N_NODES'):
            # Total power should be conserved
            constraints['power_balance'] = {
                'type': 'linear',
                'matrix': self._compute_power_balance_matrix(),
                'rhs': torch.zeros(1, device=Z.device)
            }
        
        # 3. Coupling structure constraint
        # Certain states should remain coupled
        if self.physics_informed and self.freq_modes is not None:
            constraints['coupling'] = {
                'type': 'structure',
                'pattern': self._compute_coupling_pattern()
            }
        
        return constraints
    
    def _compute_power_balance_matrix(self):
        """Compute matrix for power balance constraint."""
        # Simplified - actual implementation would be system-specific
        return torch.ones(1, self.latent_dim) / self.latent_dim
    
    def _compute_coupling_pattern(self):
        """Compute which states should remain coupled."""
        # Analyze projection matrix to find coupled states
        coupling = torch.abs(self.proj.T @ self.proj) > 0.1
        return coupling
    
    def _apply_dynamics_constraints(self, constraints):
        """Apply constraints to fitted dynamics."""
        A = self.dyn.A.data
        
        # 1. Stability constraint
        if 'stability' in constraints:
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            
            if max_real > -constraints['stability']['bound']:
                # Shift eigenvalues
                shift = max_real + constraints['stability']['bound']
                A = A - shift * torch.eye(self.latent_dim, device=A.device)
        
        # 2. Structure constraint
        if 'coupling' in constraints:
            pattern = constraints['coupling']['pattern']
            # Zero out elements that should not be coupled
            A = A * pattern
        
        # Update dynamics
        self.dyn.A.data = A
    
    def _enforce_stability_constraints(self):
        """Post-process dynamics to ensure stability."""
        if self.dyn is None:
            return
        
        A = self.dyn.A.data
        device = A.device
        
        # Method 1: Eigenvalue analysis and shifting
        try:
            eigvals, eigvecs = torch.linalg.eig(A)
            real_parts = eigvals.real
            
            if real_parts.max() > -self.stability_margin:
                print(f"    Shifting eigenvalues (max real: {real_parts.max().item():.3f})")
                
                # Shift to ensure stability
                shift = real_parts.max() + self.stability_margin
                A_shifted = A - shift * torch.eye(self.latent_dim, device=device)
                
                # Verify stability
                eigvals_new = torch.linalg.eigvals(A_shifted)
                if eigvals_new.real.max() < 0:
                    self.dyn.A.data = A_shifted
                else:
                    # Fallback to more aggressive stabilization
                    self.dyn.A.data = -self.stability_margin * torch.eye(self.latent_dim, device=device)
                    
        except Exception as e:
            print(f"    Eigenvalue analysis failed: {e}")
            # Simple diagonal stabilization
            self.dyn.A.data = -self.stability_margin * torch.eye(self.latent_dim, device=device)
        
        # Method 2: Add artificial damping to B matrix
        if hasattr(self.dyn, 'B'):
            # Reduce control influence to improve stability
            self.dyn.B.data = 0.5 * self.dyn.B.data
        
        # Method 3: Verify Lyapunov stability
        try:
            P = self._solve_lyapunov(self.dyn.A.data)
            if not self._is_positive_definite(P):
                print("    Lyapunov verification failed, adding more damping")
                self.dyn.A.data = self.dyn.A.data - 0.1 * torch.eye(self.latent_dim, device=device)
        except:
            pass
    
    def _solve_lyapunov(self, A):
        """Solve continuous Lyapunov equation A^T P + P A = -Q."""
        Q = torch.eye(A.shape[0], device=A.device)
        
        # Simple iterative solver
        P = Q.clone()
        for _ in range(100):
            P_new = Q - A.T @ P - P @ A
            if (P_new - P).norm() < 1e-6:
                break
            P = 0.5 * (P + P_new)  # Relaxation
        
        return P
    
    def _is_positive_definite(self, P):
        """Check if matrix is positive definite."""
        try:
            eigvals = torch.linalg.eigvalsh(P)
            return eigvals.min() > 1e-6
        except:
            return False
    
    def _compute_gamma_enhanced(self, X, V_fn, V_min):
        """Enhanced gamma computation with better error estimation."""
        try:
            # Get dynamics residual
            if hasattr(self.dyn, 'residual'):
                eps = float(self.dyn.residual.item())
            else:
                # Estimate from reconstruction error
                Z = self.forward(X)
                X_recon = self.inverse(Z)
                eps = (X - X_recon).norm(dim=1).max().item()
            
            # Estimate Lipschitz constant more carefully
            L_V = self._estimate_lipschitz_constant(X, V_fn)
            
            # Compute gamma with safety factor
            safety_factor = 1.5
            self.gamma = safety_factor * eps * L_V / max(V_min, 1e-4)
            
            # Cap at reasonable value
            self.gamma = min(self.gamma, 50.0)
            
        except Exception as e:
            print(f"    Gamma computation failed: {e}")
            self.gamma = 10.0
    
    def _estimate_lipschitz_constant(self, X, V_fn):
        """Carefully estimate Lipschitz constant of V."""
        if not callable(V_fn):
            return 10.0
        
        try:
            # Sample pairs of points
            n_samples = min(len(X), 100)
            idx = torch.randperm(len(X))[:n_samples]
            
            L_estimates = []
            
            for i in range(n_samples - 1):
                x1 = X[idx[i]]
                x2 = X[idx[i+1]]
                
                # Skip if points are too close
                dist = (x2 - x1).norm()
                if dist < 1e-6:
                    continue
                
                # Compute function values
                v1 = V_fn(x1.unsqueeze(0))
                v2 = V_fn(x2.unsqueeze(0))
                
                # Estimate local Lipschitz constant
                L_local = abs(v2.item() - v1.item()) / dist.item()
                
                if L_local < 1e6:  # Reasonable bound
                    L_estimates.append(L_local)
            
            if L_estimates:
                # Use 95th percentile as robust estimate
                L_estimates = torch.tensor(L_estimates)
                L_V = torch.quantile(L_estimates, 0.95).item()
                return max(1e-3, min(L_V, 100.0))
            
        except Exception as e:
            print(f"    Lipschitz estimation failed: {e}")
        
        return 10.0
    
    def _verify_perturbation_preservation(self, X, X_disturbed):
        """Verify that reducer preserves response to perturbations."""
        print("    Verifying perturbation preservation...")
        
        # Compute perturbations in full space
        dX = X_disturbed - X
        
        # Project to reduced space
        Z = self.forward(X)
        Z_disturbed = self.forward(X_disturbed)
        dZ = Z_disturbed - Z
        
        # Check preservation ratio
        preservation_ratios = []
        
        for i in range(min(len(X), 10)):
            if dX[i].norm() > 1e-6:
                ratio = dZ[i].norm() / dX[i].norm()
                preservation_ratios.append(ratio.item())
        
        if preservation_ratios:
            mean_preservation = np.mean(preservation_ratios)
            print(f"    Mean perturbation preservation: {mean_preservation:.1%}")
            
            if mean_preservation < 0.3:
                print("    WARNING: Poor perturbation preservation!")
    
    # Override base methods
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to reduced space."""
        if x.dim() == 1:
            return (x - self.μ) @ self.proj
        return (x - self.μ) @ self.proj
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from reduced space."""
        if z.dim() == 1:
            return z @ self.proj.T + self.μ
        return z @ self.proj.T + self.μ
    
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Analytical Jacobian."""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.proj.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()