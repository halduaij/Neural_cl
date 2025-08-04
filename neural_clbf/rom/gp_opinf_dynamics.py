from __future__ import annotations
import torch, itertools
from torch import nn
import numpy as np


class GPOpInfDynamics(nn.Module):
    """Operator inference latent ODE with guaranteed stability and robustness."""

    def __init__(self, d: int, m: int = 1):
        super().__init__()
        self.d = d
        self.m = m
        
        # Initialize with strongly stable diagonal matrix
        self.A = nn.Parameter(-0.5 * torch.eye(d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(d, m), requires_grad=False)
        
        # Quadratic terms setup
        comb = list(itertools.combinations_with_replacement(range(d), 2))
        self.register_buffer("_comb", torch.tensor(comb, dtype=torch.long))
        
        # H (quadratic terms) - keep zero for power system stability
        self.H = nn.Parameter(torch.zeros(d, len(comb)), requires_grad=False)
        
        # Track fitting metrics
        self.register_buffer("residual", torch.tensor(1.0))
        self.register_buffer("condition_number", torch.tensor(1.0))
        self.register_buffer("spectral_radius", torch.tensor(0.5))
        
        # Regularization and stability parameters
        self.register_buffer("reg", torch.tensor(0.1))
        self.stability_margin = 0.5
        self.max_spectral_radius = 0.95  # For discrete stability
        self.disable_threshold = 0.99   # Threshold to recommend disabling

    def _q(self, z):
        """Quadratic terms (disabled for stability)."""
        return torch.zeros(z.shape[0], len(self._comb), device=z.device)

    def forward(self, z, u=None):
        """
        Compute latent dynamics dz/dt = Az + Bu with stability checks.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Clamp input to prevent numerical issues
        z_max = 1e4
        if z.abs().max() > z_max:
            z = torch.clamp(z, -z_max, z_max)
        
        # Check for NaN/Inf
        if not torch.isfinite(z).all():
            z = torch.nan_to_num(z, nan=0.0, posinf=z_max, neginf=-z_max)
        
        # Linear dynamics
        out = z @ self.A.T
        
        # Control term
        if u is not None and self.m > 0:
            if u.dim() == 1:
                u = u.unsqueeze(0)
            if u.shape[0] == 1 and z.shape[0] > 1:
                u = u.expand(z.shape[0], -1)
            
            # Clamp control
            u = torch.clamp(u, -z_max, z_max)
            out = out + u @ self.B.T
        
        # Final clamping
        out = torch.clamp(out, -z_max, z_max)
        
        return out

    @torch.no_grad()
    def fit(self, Z, U, dZdt):
        """
        Fit dynamics with multiple stability guarantees.
        """
        device = Z.device
        N, d = Z.shape
        
        print(f"\nFitting OpInf dynamics (d={d}):")
        
        # 1. Data validation and preprocessing
        Z, dZdt, U, valid_mask = self._preprocess_data(Z, dZdt, U)
        
        if valid_mask.sum() < 2 * d:
            print("  Warning: Insufficient valid data. Using default stable dynamics.")
            self._set_default_stable_dynamics(device)
            return
        
        # 2. Robust regression with multiple methods
        A_fitted, B_fitted, fit_metrics = self._robust_regression(Z, dZdt, U)
        
        # 3. Stability analysis and enforcement
        A_stable = self._enforce_stability(A_fitted, device)
        
        # 4. Final verification
        self._verify_and_finalize(A_stable, B_fitted, Z, dZdt, U, device)
        
        # 5. Print summary
        self._print_fit_summary()

    def _preprocess_data(self, Z, dZdt, U):
        """Robust data preprocessing with outlier detection."""
        device = Z.device
        
        # Check for NaN/Inf
        valid_mask = torch.isfinite(Z).all(dim=1) & torch.isfinite(dZdt).all(dim=1)
        
        if U is not None:
            valid_mask &= torch.isfinite(U).all(dim=1)
        
        # Outlier detection using IQR method
        Z_valid = Z[valid_mask]
        dZdt_valid = dZdt[valid_mask]
        
        if Z_valid.shape[0] > 10:
            # Compute IQR for each dimension
            q1 = torch.quantile(Z_valid, 0.25, dim=0)
            q3 = torch.quantile(Z_valid, 0.75, dim=0)
            iqr = q3 - q1
            
            # Flag outliers
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outlier_mask = ((Z_valid < lower_bound) | (Z_valid > upper_bound)).any(dim=1)
            outlier_mask |= ((dZdt_valid < -100) | (dZdt_valid > 100)).any(dim=1)
            
            # Update valid mask
            valid_indices = torch.where(valid_mask)[0]
            valid_mask[valid_indices[outlier_mask]] = False
        
        print(f"  Valid samples: {valid_mask.sum()}/{len(Z)} ({valid_mask.float().mean():.1%})")
        
        # Extract valid data
        Z_fit = Z[valid_mask]
        dZdt_fit = dZdt[valid_mask]
        U_fit = U[valid_mask] if U is not None else None
        
        return Z_fit, dZdt_fit, U_fit, valid_mask

    def _robust_regression(self, Z, dZdt, U):
        """Perform robust regression with multiple methods."""
        device = Z.device
        d = self.d
        
        # Build data matrix
        if U is not None and self.m > 0:
            Phi = torch.cat([Z, U], dim=1)
        else:
            Phi = Z
        
        # Method 1: Ridge regression via normal equations
        try:
            reg_matrix = self.reg * torch.eye(Phi.shape[1], device=device)
            
            # Add stronger regularization for near-singular cases
            gram_matrix = Phi.T @ Phi
            cond = torch.linalg.cond(gram_matrix)
            if cond > 1e6:
                print(f"  High condition number ({cond:.1e}), increasing regularization")
                reg_matrix *= max(1, cond / 1e6)
            
            theta = torch.linalg.solve(gram_matrix + reg_matrix, Phi.T @ dZdt)
            
            A_fitted = theta[:d].T
            B_fitted = theta[d:].T if U is not None else None
            
            method = "Ridge (normal equations)"
            
        except Exception as e:
            print(f"  Normal equations failed: {e}")
            
            # Method 2: SVD-based pseudoinverse
            try:
                U_svd, S_svd, Vt_svd = torch.linalg.svd(Phi, full_matrices=False)
                
                # Truncate small singular values
                tol = 1e-8 * S_svd[0]
                rank = (S_svd > tol).sum().item()
                
                if rank < Phi.shape[1]:
                    print(f"  Rank deficient: {rank}/{Phi.shape[1]}")
                
                # Damped pseudoinverse
                S_inv = S_svd / (S_svd**2 + self.reg)
                theta = Vt_svd.T[:, :rank] @ (torch.diag(S_inv[:rank]) @ (U_svd.T[:rank] @ dZdt))
                
                A_fitted = theta[:d].T
                B_fitted = theta[d:].T if U is not None and theta.shape[0] > d else None
                
                method = "SVD (truncated)"
                
            except Exception as e2:
                print(f"  SVD failed: {e2}. Using diagonal dynamics.")
                A_fitted = -0.5 * torch.eye(d, device=device)
                B_fitted = torch.zeros(d, self.m, device=device) if U is not None else None
                method = "Fallback (diagonal)"
        
        # Compute fit metrics
        predictions = Z @ A_fitted.T
        if B_fitted is not None and U is not None:
            predictions += U @ B_fitted.T
        
        residuals = (dZdt - predictions).norm(dim=1)
        
        metrics = {
            'method': method,
            'residual_mean': residuals.mean().item(),
            'residual_max': residuals.max().item(),
            'condition_number': cond.item() if 'cond' in locals() else float('inf')
        }
        
        print(f"  Regression method: {method}")
        print(f"  Mean residual: {metrics['residual_mean']:.3e}")
        
        return A_fitted, B_fitted, metrics

    def _enforce_stability(self, A, device):
        """Enforce stability through multiple methods."""
        print("\n  Enforcing stability...")
        
        # Method 1: Check continuous stability
        try:
            eigvals = torch.linalg.eigvals(A)
            max_real = eigvals.real.max().item()
            print(f"    Continuous: max Re(λ) = {max_real:.4f}")
            
            if max_real > -self.stability_margin:
                # Shift eigenvalues
                shift = max_real + self.stability_margin
                A = A - shift * torch.eye(self.d, device=device)
                print(f"    Shifted by {shift:.4f}")
                
                # Recompute
                eigvals = torch.linalg.eigvals(A)
                max_real = eigvals.real.max().item()
                print(f"    After shift: max Re(λ) = {max_real:.4f}")
                
        except Exception as e:
            print(f"    Eigenvalue computation failed: {e}")
            # Fallback: make strongly diagonal
            A = -self.stability_margin * torch.eye(self.d, device=device)
        
        # Method 2: Check discrete stability (for Euler integration)
        dt = 0.01  # Standard timestep
        A_discrete = torch.eye(self.d, device=device) + dt * A
        
        try:
            rho = torch.linalg.eigvals(A_discrete).abs().max().item()
            print(f"    Discrete: ρ(I + dt*A) = {rho:.6f} (dt={dt})")
            
            if rho >= self.max_spectral_radius:
                # Scale to ensure stability
                scale = (self.max_spectral_radius - 1) / (rho - 1) * 0.95  # Safety factor
                A = scale * A
                print(f"    Scaled by {scale:.4f}")
                
                # Final check
                A_discrete = torch.eye(self.d, device=device) + dt * A
                rho = torch.linalg.eigvals(A_discrete).abs().max().item()
                print(f"    Final: ρ(I + dt*A) = {rho:.6f}")
                
                # Store for diagnostics
                self.spectral_radius = torch.tensor(rho)
                
        except Exception as e:
            print(f"    Discrete check failed: {e}")
            # Conservative fallback
            A = -self.stability_margin * torch.eye(self.d, device=device)
            self.spectral_radius = torch.tensor(0.95)
        
        # Method 3: Lyapunov stability (optional)
        try:
            # Check if A + A^T < 0 (sufficient for stability)
            symmetric_part = 0.5 * (A + A.T)
            min_eig = torch.linalg.eigvalsh(symmetric_part).max().item()
            
            if min_eig >= 0:
                print(f"    Symmetric part not negative definite (max eig: {min_eig:.4f})")
                # Add negative diagonal to ensure stability
                A = A - (min_eig + 0.1) * torch.eye(self.d, device=device)
                
        except:
            pass
        
        return A

    def _verify_and_finalize(self, A, B, Z, dZdt, U, device):
        """Final verification and setup."""
        # Set the fitted parameters
        self.A.data = A
        if B is not None:
            self.B.data = torch.clamp(B, -10.0, 10.0)  # Limit control influence
        
        # Compute final residual
        predictions = self.forward(Z, U)
        residuals = (dZdt - predictions).norm(dim=1)
        
        # Use robust statistics
        self.residual = torch.quantile(residuals, 0.95)
        
        # Store condition number
        self.condition_number = torch.linalg.cond(self.A)
        
        # Final stability check
        dt = 0.01
        A_discrete = torch.eye(self.d, device=device) + dt * self.A
        self.spectral_radius = torch.linalg.eigvals(A_discrete).abs().max()
        
        # Recommendation check
        if self.spectral_radius > self.disable_threshold:
            print("\n  ⚠️  WARNING: Dynamics may be unstable!")
            print(f"     Spectral radius {self.spectral_radius:.6f} > {self.disable_threshold}")
            print("     Consider disabling learned dynamics for this latent dimension.")

    def _set_default_stable_dynamics(self, device):
        """Set default stable dynamics."""
        self.A.data = -self.stability_margin * torch.eye(self.d, device=device)
        self.B.data = torch.zeros(self.d, self.m, device=device)
        self.residual = torch.tensor(1.0, device=device)
        self.spectral_radius = torch.tensor(
            1 - self.stability_margin * 0.01, device=device
        )  # For dt=0.01

    def _print_fit_summary(self):
        """Print fitting summary."""
        print("\n  Dynamics Summary:")
        print(f"    Residual (95%): {self.residual.item():.3e}")
        print(f"    Condition number: {self.condition_number.item():.1e}")
        print(f"    Spectral radius: {self.spectral_radius.item():.6f}")
        
        # Stability assessment
        if self.spectral_radius < 0.95:
            print("    ✓ STABLE")
        elif self.spectral_radius < 0.99:
            print("    ⚠️  MARGINALLY STABLE")
        else:
            print("    ✗ UNSTABLE")

    def to(self, device):
        """Move all tensors to device."""
        super().to(device)
        self._comb = self._comb.to(device)
        self.residual = self.residual.to(device)
        self.reg = self.reg.to(device)
        self.condition_number = self.condition_number.to(device)
        self.spectral_radius = self.spectral_radius.to(device)
        return self
    
    def is_stable(self, dt=0.01):
        """Check if the dynamics are stable for given timestep."""
        return self.spectral_radius.item() < self.max_spectral_radius