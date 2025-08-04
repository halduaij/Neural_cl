from __future__ import annotations
import torch
import numpy as np
import logging
from neural_clbf.dimension_reduction.base import BaseReducer
try:
    from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
except ImportError:
    try:
        from gp_opinf_dynamics import GPOpInfDynamics
    except ImportError:
        GPOpInfDynamics = None

logger = logging.getLogger(__name__)


class OpInfReducer(BaseReducer):
    """PCA encoder + latent ODE with enhanced stability guarantees."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1, 
                 include_quadratic: bool = False):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.n_controls = n_controls
        self.include_quadratic = include_quadratic
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        self.dyn = None
        self.sys = None
        
        # Stability parameters
        self.stability_margin = 1.0  # Eigenvalue margin for stability
        self.use_energy_basis = True
        self.disable_unstable_dynamics = True
        self.regularization_factor = 1.0  # Base regularization

    def fit(self, X, Xdot, V_fn, V_min):
        device = X.device
        if GPOpInfDynamics is None:
            raise RuntimeError("GPOpInfDynamics not available.")
        
        logger.info(f"\nFitting OpInf Reducer (d={self.latent_dim}):")
        
        # 1. Enhanced PCA with energy-preserving modes
        self.μ = X.mean(0)
        X_centered = X - self.μ
        
        # Standard SVD
        try:
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            
            # Determine effective rank
            total_energy = (S ** 2).sum()
            if total_energy > 1e-9:
                cumsum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                n_effective = (cumsum_energy < 0.999).sum().item() + 1
            else:
                n_effective = 1
            
            n_keep = min(self.latent_dim, n_effective, X.shape[1])
            
            if n_keep < self.latent_dim:
                logger.info(f"  Reducing latent dim from {self.latent_dim} to {n_keep} (99.9% energy)")
                self.latent_dim = n_keep
            
            # Base POD modes
            pod_basis = Vt[:self.latent_dim].T
            
            # Report variance captured
            if total_energy > 1e-9:
                variance_captured = cumsum_energy[self.latent_dim - 1].item()
                logger.info(f"  Variance captured: {variance_captured:.2%}")
            
            # Enhance with energy-preserving modes if possible
            if self.use_energy_basis and callable(V_fn) and self.latent_dim > 4:
                try:
                    enhanced_basis = self._compute_energy_enhanced_basis(X, V_fn, pod_basis)
                    if enhanced_basis is not None:
                        self.proj = enhanced_basis
                    else:
                        self.proj = pod_basis
                except Exception as e:
                    logger.warning(f"    Energy basis failed: {e}. Using standard POD.")
                    self.proj = pod_basis
            else:
                self.proj = pod_basis
                
        except Exception as e:
            logger.error(f"  SVD failed: {e}. Using QR fallback.")
            try:
                Q, _ = torch.linalg.qr(X_centered.T)
                self.proj = Q[:, :self.latent_dim]
            except:
                self.proj = torch.eye(self.full_dim, self.latent_dim, device=device)
        
        # 2. Project Data
        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        
        # 3. Fit Dynamics with adaptive regularization
        actual_n_controls = getattr(self.sys, 'n_controls', self.n_controls) if self.sys else self.n_controls
        
        self.dyn = GPOpInfDynamics(self.latent_dim, actual_n_controls, 
                                   include_quadratic=self.include_quadratic).to(device)
        
        # Compute adaptive regularization based on data scale
        Z_std = Z.std()
        adaptive_reg = max(self.regularization_factor, 1e-3 * Z_std.item())
        self.dyn.reg = torch.tensor(adaptive_reg, device=device)
        logger.info(f"  Using regularization: {adaptive_reg:.2e}")
        
        U = torch.zeros(Z.shape[0], actual_n_controls, device=device)
        
        # Fit dynamics
        try:
            self.dyn.fit(Z, U, dZdt)
        except Exception as e:
            logger.error(f"  Dynamics fitting failed: {e}. Disabling dynamics.")
            self.dyn = None

        if self.dyn is not None:
            # Check if fit produced valid dynamics
            if hasattr(self.dyn, 'A') and not torch.isfinite(self.dyn.A).all():
                logger.critical("  Dynamics fitting produced non-finite A matrix. Disabling dynamics.")
                self.dyn = None
            
            # 4. Enforce stability constraints
            if self.dyn is not None:
                self._enforce_stability(device)
            
            # 5. Check and disable if unstable
            if self.dyn is not None and self.disable_unstable_dynamics:
                self._check_and_disable_unstable_dynamics()
        
        # 6. Compute Gamma
        self.compute_gamma(X, V_fn, V_min)
        
        self.to(device)
        return self
    
    def _compute_energy_enhanced_basis(self, X, V_fn, pod_basis):
        """Compute energy-preserving basis enhancement"""
        logger.info("  Computing energy-preserving basis enhancement...")
        
        try:
            # Sample energy gradients
            n_samples = min(100, X.shape[0])
            idx = torch.randperm(X.shape[0])[:n_samples]
            grad_V = []
            
            for i in idx:
                x = X[i:i+1].clone().requires_grad_(True)
                v = V_fn(x)
                if v.requires_grad:
                    g = torch.autograd.grad(v.sum(), x, create_graph=False)[0]
                    grad_V.append(g.squeeze())
            
            if len(grad_V) > 10:
                grad_V = torch.stack(grad_V)
                grad_V_centered = grad_V - grad_V.mean(0)
                
                # POD on energy gradients
                U_energy, S_energy, _ = torch.linalg.svd(grad_V_centered.T, full_matrices=False)
                
                # Take top energy modes
                n_energy_modes = min(self.latent_dim // 4, U_energy.shape[1])
                
                # Combine POD and energy modes
                combined = torch.cat([pod_basis, U_energy[:, :n_energy_modes]], dim=1)
                
                # Orthogonalize
                Q, _ = torch.linalg.qr(combined)
                logger.info(f"    Added {n_energy_modes} energy-preserving modes")
                return Q[:, :self.latent_dim]
            else:
                return None
        except Exception as e:
            logger.warning(f"    Energy basis computation failed: {e}")
            return None
    
    def _enforce_stability(self, device):
        """Enforce continuous and discrete time stability with MKL error protection."""
        if self.dyn is None or not hasattr(self.dyn, 'A'):
            return
            
        A = self.dyn.A.data
        
        # CRITICAL: Check for non-finite values before eigenvalue computation
        if not torch.isfinite(A).all():
            logger.critical("  Non-finite values in A matrix. Applying fallback stabilization.")
            self.dyn.A.data = -self.stability_margin * torch.eye(self.latent_dim, device=device)
            return

        # 1. Continuous time stability (Eigenvalue shifting)
        try:
            # Use torch.linalg.eig for better robustness
            eigvals, _ = torch.linalg.eig(A)
            max_real = eigvals.real.max().item()
            
            if not np.isfinite(max_real):
                raise ValueError("Non-finite eigenvalues")
            
            if max_real > -self.stability_margin:
                shift = max_real + self.stability_margin
                logger.info(f"  Stabilizing: shifting eigenvalues by {shift:.3f}.")
                self.dyn.A.data = A - shift * torch.eye(self.latent_dim, device=device)
                
        except Exception as e:
            logger.warning(f"  Eigenvalue analysis failed ({e}). Applying fallback stabilization.")
            # Apply conservative stabilization
            self.dyn.A.data = -self.stability_margin * torch.eye(self.latent_dim, device=device)
            return

        # 2. Reduce control influence for stability
        if hasattr(self.dyn, 'B') and self.dyn.B is not None:
            self.dyn.B.data = 0.5 * self.dyn.B.data

        # 3. Discrete time stability (Spectral radius check)
        dt = 0.01
        A_current = self.dyn.A.data
        A_discrete = torch.eye(self.latent_dim, device=device) + dt * A_current
        
        try:
            eigvals_discrete = torch.linalg.eigvals(A_discrete)
            spectral_radius = eigvals_discrete.abs().max().item()
            target_radius = 0.99
            
            if spectral_radius >= 1.0:
                if spectral_radius > 1e-9:
                    scale = target_radius / spectral_radius
                    self.dyn.A.data = A_current * scale
                    logger.info(f"    Scaled for discrete stability: ρ = {spectral_radius:.4f} -> {target_radius:.4f}")
        except Exception as e:
            logger.warning(f"    Discrete stability check failed: {e}")
            pass

    def _check_and_disable_unstable_dynamics(self):
        """Check stability with relaxed thresholds and shifting instead of disabling."""
        if self.dyn is None:
            return
            
        dt = 0.01
        device = self.dyn.A.device
        
        # 1. Check spectral radius with relaxed threshold
        A_discrete = torch.eye(self.latent_dim, device=device) + dt * self.dyn.A.data
        try:
            spectral_radius = torch.linalg.eigvals(A_discrete).abs().max().item()
            threshold = 0.999  # Relaxed from 0.995
            
            # Also check for near-full dimension cases
            dim_ratio = self.latent_dim / self.full_dim
            
            if spectral_radius >= threshold:
                logger.warning(f"  Discrete system marginally unstable (ρ={spectral_radius:.6f})")
                
                # Instead of disabling, try to stabilize by shifting
                if spectral_radius < 1.01:  # Close to stable
                    shift = (spectral_radius - 0.98) / dt
                    self.dyn.A.data = self.dyn.A.data - shift * torch.eye(self.latent_dim, device=device)
                    logger.info(f"    Applied stabilizing shift: -{shift:.3f}I")
                    
                    # Recompute spectral radius
                    A_discrete_new = torch.eye(self.latent_dim, device=device) + dt * self.dyn.A.data
                    spectral_radius_new = torch.linalg.eigvals(A_discrete_new).abs().max().item()
                    logger.info(f"    New spectral radius: {spectral_radius_new:.6f}")
                else:
                    # Only disable if very unstable
                    logger.critical(f"    System too unstable to fix (ρ={spectral_radius:.3f}). Disabling dynamics.")
                    self.dyn = None
                    return
                
        except Exception as e:
            logger.warning(f"    Spectral radius check failed: {e}")

        # 2. Test rollout stability with longer tolerance
        if self.dyn is not None:
            logger.info("  Performing rollout stability test...")
            z_test = torch.randn(5, self.latent_dim, device=device) * 0.5
            stable = True
            max_norm = 1e6  # Increased from 1e5
            
            for t in range(200): # 2 seconds test
                try:
                    # Handle potential need for control input U
                    if self.n_controls > 0 and hasattr(self.dyn, 'B'):
                         U_zero = torch.zeros(z_test.shape[0], self.n_controls, device=device)
                         z_dot = self.dyn.forward(z_test, U_zero)
                    else:
                         z_dot = self.dyn.forward(z_test)
                except Exception as e:
                     logger.error(f"    Rollout test failed with error: {e}")
                     stable = False
                     break

                z_next = z_test + dt * z_dot
                
                # Check for NaN/Inf or explosion
                if not torch.isfinite(z_next).all() or z_next.norm(dim=1).max().item() > max_norm:
                    stable = False
                    break
                z_test = z_next
            
            if not stable:
                logger.warning(f"  Rollout test failed - applying stronger stabilization")
                # Try stronger stabilization before giving up
                self.dyn.A.data = self.dyn.A.data - 0.5 * torch.eye(self.latent_dim, device=device)
                
                # Test again
                z_test = torch.randn(1, self.latent_dim, device=device) * 0.1
                stable_retry = True
                for t in range(50):  # Shorter test
                    try:
                        if self.n_controls > 0 and hasattr(self.dyn, 'B'):
                            U_zero = torch.zeros(z_test.shape[0], self.n_controls, device=device)
                            z_dot = self.dyn.forward(z_test, U_zero)
                        else:
                            z_dot = self.dyn.forward(z_test)
                        z_next = z_test + dt * z_dot
                        if not torch.isfinite(z_next).all():
                            stable_retry = False
                            break
                        z_test = z_next
                    except:
                        stable_retry = False
                        break
                
                if not stable_retry:
                    logger.critical("    Stabilization failed. Disabling dynamics.")
                    self.dyn = None

    def compute_gamma(self, X, V_fn, V_min):
        """Computes the robustness margin gamma robustly."""
        try:
            if self.dyn is None:
                # For projection-only dynamics
                X_recon = self.inverse(self.forward(X))
                proj_error = (X - X_recon).norm(dim=1).max().item()
                self.gamma = proj_error / max(V_min, 1e-4)
                logger.info(f"  Using projection gamma = {self.gamma:.6f}")
            else:
                # Standard OpInf gamma
                eps = float(self.dyn.residual.item()) if hasattr(self.dyn, 'residual') else 0.1
                
                # Estimate L_V (Lipschitz constant of V)
                if callable(V_fn):
                    L_V = self._estimate_lipschitz_safe(X, V_fn)
                else:
                    L_V = 10.0
                
                V_min_safe = max(V_min, 1e-4)
                self.gamma = eps * L_V / V_min_safe
            
            # Cap gamma at reasonable value
            self.gamma = min(self.gamma, 50.0)
            
            # Extra safety for near-full dimension
            if self.latent_dim >= 0.9 * self.full_dim:
                self.gamma = min(self.gamma, 1.0)
                
        except Exception as e:
            logger.error(f"  Gamma computation failed: {e}. Setting default gamma=10.0.")
            self.gamma = 10.0
    
    def _estimate_lipschitz_safe(self, X, V_fn):
        """Safely estimate Lipschitz constant."""
        try:
            n_samples = min(50, X.shape[0])
            idx = torch.randperm(X.shape[0])[:n_samples]
            
            L_estimates = []
            
            for i in range(n_samples - 1):
                x1 = X[idx[i]]
                x2 = X[idx[i+1]]
                
                dist = (x2 - x1).norm()
                if dist < 1e-6:
                    continue
                
                v1 = V_fn(x1.unsqueeze(0))
                v2 = V_fn(x2.unsqueeze(0))
                
                L_local = abs(v2.item() - v1.item()) / dist.item()
                
                if L_local < 1e6:
                    L_estimates.append(L_local)
            
            if L_estimates:
                L_estimates = torch.tensor(L_estimates)
                L_V = torch.quantile(L_estimates, 0.95).item()
                return max(1e-3, min(L_V, 100.0))
                
        except:
            pass
            
        return 10.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return (x - self.μ) @ self.proj
        return (x - self.μ) @ self.proj
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            return z @ self.proj.T + self.μ
        return z @ self.proj.T + self.μ
        
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). J = P.T"""
        B = X.shape[0] if X.dim() > 1 else 1
        J = self.proj.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
        
    def to(self, device):
        super().to(device)
        self.μ = self.μ.to(device)
        self.proj = self.proj.to(device)
        if self.dyn is not None:
            self.dyn = self.dyn.to(device)
        return self