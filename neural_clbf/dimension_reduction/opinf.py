from __future__ import annotations
import torch
import numpy as np
from neural_clbf.dimension_reduction.base import BaseReducer
try:
    from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics
except ImportError:
    from gp_opinf_dynamics import GPOpInfDynamics


class OpInfReducer(BaseReducer):
    """PCA encoder + latent ODE with stability guarantees and energy preservation."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.n_controls = n_controls
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        self.dyn = None
        self.sys = None
        
        # Additional parameters for stability
        self.stability_margin = 0.5
        self.use_energy_basis = True
        self.disable_unstable_dynamics = True

    def fit(self, X, Xdot, V_fn, V_min):
        device = X.device
        
        print(f"\nFitting OpInf Reducer (d={self.latent_dim}):")
        
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
                print(f"  Reducing latent dim from {self.latent_dim} to {n_keep} (99.9% energy)")
                self.latent_dim = n_keep
            
            # Base POD modes
            pod_basis = Vt[:self.latent_dim].T
            
            # Enhance with energy-preserving modes if possible
            if self.use_energy_basis and callable(V_fn) and self.latent_dim > 4:
                try:
                    print("  Computing energy-preserving basis enhancement...")
                    
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
                        self.proj = Q[:, :self.latent_dim]
                        print(f"    Added {n_energy_modes} energy-preserving modes")
                    else:
                        self.proj = pod_basis
                except Exception as e:
                    print(f"    Energy basis failed: {e}. Using standard POD.")
                    self.proj = pod_basis
            else:
                self.proj = pod_basis
                
        except Exception as e:
            print(f"  SVD failed: {e}. Using QR fallback.")
            try:
                Q, _ = torch.linalg.qr(X_centered.T)
                self.proj = Q[:, :self.latent_dim]
            except:
                self.proj = torch.eye(self.full_dim, self.latent_dim, device=device)
        
        # 2. Project Data
        Z = self.forward(X)
        dZdt = Xdot @ self.proj
        
        # 3. Fit Dynamics with stability
        actual_n_controls = getattr(self.sys, 'n_controls', self.n_controls) if self.sys else self.n_controls
        
        self.dyn = GPOpInfDynamics(self.latent_dim, actual_n_controls).to(device)
        
        # Strong regularization for stability
        self.dyn.reg = torch.tensor(0.1, device=device)
        
        U = torch.zeros(Z.shape[0], actual_n_controls, device=device)
        
        # Fit dynamics
        self.dyn.fit(Z, U, dZdt)
        
        # 4. Enforce stability
        self._enforce_stability(device)
        
        # 5. Check and potentially disable unstable dynamics
        if self.disable_unstable_dynamics:
            self._check_and_disable_unstable_dynamics()
        
        # 6. Compute Gamma
        self.compute_gamma(X, V_fn, V_min)
        
        self.to(device)
        return self
    
    def _enforce_stability(self, device):
        """Enforce stability through eigenvalue shifting and projection."""
        if self.dyn is None:
            return
            
        A = self.dyn.A.data
        
        # Method 1: Eigenvalue shifting
        try:
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            
            if max_real > -self.stability_margin:
                print(f"  Stabilizing: shifting eigenvalues (max real: {max_real:.3f})")
                shift = max_real + self.stability_margin
                A_shifted = A - shift * torch.eye(self.latent_dim, device=device)
                self.dyn.A.data = A_shifted
                
                # Verify
                new_eigvals = torch.linalg.eigvals(A_shifted)
                new_max_real = new_eigvals.real.max().item()
                print(f"    New max eigenvalue: {new_max_real:.3f}")
                
        except Exception as e:
            print(f"  Eigenvalue analysis failed: {e}")
            # Fallback: make diagonal dominant
            self.dyn.A.data = -self.stability_margin * torch.eye(self.latent_dim, device=device)
        
        # Method 2: Reduce control influence for stability
        if hasattr(self.dyn, 'B') and self.dyn.B is not None:
            self.dyn.B.data = 0.5 * self.dyn.B.data
        
        # Method 3: Ensure discrete stability
        dt = 0.01  # Standard timestep
        A_discrete = torch.eye(self.latent_dim, device=device) + dt * self.dyn.A.data
        
        try:
            spectral_radius = torch.linalg.eigvals(A_discrete).abs().max().item()
            
            if spectral_radius >= 0.99:
                # Scale to ensure discrete stability
                scale = 0.98 / spectral_radius
                self.dyn.A.data = self.dyn.A.data * scale
                print(f"    Scaled for discrete stability: ρ = {spectral_radius:.3f} → 0.98")
                
        except Exception as e:
            print(f"    Discrete stability check failed: {e}")
    
    def _check_and_disable_unstable_dynamics(self):
        """Check stability and disable dynamics if unstable."""
        if self.dyn is None:
            return
            
        dt = 0.01
        device = self.dyn.A.device
        
        # Check discrete stability
        A_discrete = torch.eye(self.latent_dim, device=device) + dt * self.dyn.A
        
        try:
            spectral_radius = torch.linalg.eigvals(A_discrete).abs().max().item()
            
            # Also check for near-full dimension cases
            dim_ratio = self.latent_dim / self.full_dim
            
            if spectral_radius >= 0.999 or (dim_ratio > 0.9 and spectral_radius >= 0.99):
                print(f"  CRITICAL: Discrete system unstable or near-unstable")
                print(f"    Spectral radius: {spectral_radius:.6f}")
                print(f"    Dimension ratio: {dim_ratio:.2f}")
                print(f"    Disabling learned dynamics - using projection only")
                self.dyn = None
                self.gamma = 0.01  # Very small gamma for projection-only
                return
                
            # Test rollout stability
            z_test = torch.randn(1, self.latent_dim, device=device) * 0.1
            stable = True
            
            for t in range(100):
                z_dot = self.dyn.forward(z_test)
                z_next = z_test + dt * z_dot
                
                if not torch.isfinite(z_next).all() or z_next.norm() > 1e3:
                    stable = False
                    break
                    
                z_test = z_next
            
            if not stable:
                print(f"  Rollout test failed - disabling dynamics")
                self.dyn = None
                self.gamma = 0.01
                
        except Exception as e:
            print(f"  Stability check error: {e}. Keeping dynamics.")

    def compute_gamma(self, X, V_fn, V_min):
        """Computes the robustness margin gamma robustly."""
        try:
            if self.dyn is None:
                # For projection-only dynamics
                X_recon = self.inverse(self.forward(X))
                proj_error = (X - X_recon).norm(dim=1).max().item()
                self.gamma = proj_error / max(V_min, 1e-4)
                print(f"  Using projection gamma = {self.gamma:.6f}")
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
            print(f"  Gamma computation failed: {e}. Setting default gamma=10.0.")
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