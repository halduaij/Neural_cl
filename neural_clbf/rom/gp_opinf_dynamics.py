from __future__ import annotations
import torch, itertools
from torch import nn
import numpy as np


class GPOpInfDynamics(nn.Module):
    """Operator inference latent ODE with improved numerical stability and aggressive stabilization."""

    def __init__(self, d: int, m: int = 1):
        super().__init__()
        self.d = d
        self.m = m
        # Initialize A with a stable diagonal matrix
        self.A = nn.Parameter(-0.1 * torch.eye(d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(d, m), requires_grad=False)
        comb = list(itertools.combinations_with_replacement(range(d), 2))
        self.register_buffer("_comb", torch.tensor(comb, dtype=torch.long))
        # H (quadratic terms) will be kept zero for stability in power systems.
        self.H = nn.Parameter(torch.zeros(d, len(comb)), requires_grad=False)
        self.register_buffer("residual", torch.tensor(1.0))
        # Regularization strength (set high by OpInfReducer)
        self.register_buffer("reg", torch.tensor(1e-3))

    def _q(self, z):
        # Quadratic terms are disabled (H=0)
        return torch.zeros(z.shape[0], len(self._comb), device=z.device)

    def forward(self, z, u=None):
        """
        Compute latent dynamics dz/dt = Az + Bu.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Handle NaNs/Infs and clamp input
        if not torch.isfinite(z).all():
             z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        z = torch.clamp(z, -1e4, 1e4)

        # Linear term
        out = z @ self.A.T
        
        # Control term
        if u is not None and self.m > 0:
            if u.dim() == 1:
                u = u.unsqueeze(0)
            if u.shape[0] == 1 and z.shape[0] > 1:
                u = u.expand(z.shape[0], -1)
            
            u = torch.clamp(u, -1e4, 1e4)
            out = out + u @ self.B.T
        
        # Clamp output
        out = torch.clamp(out, -1e4, 1e4)
        return out

    @torch.no_grad()
    def fit(self, Z, U, dZdt):
        """
        Fit LINEAR dynamics (A, B) using robust regression and guarantee stability.
        """
        device = Z.device
        N, d = Z.shape
        
        # 1. Robust Data Normalization (using Median and IQR)
        Z_median = Z.median(dim=0).values
        Z_iqr = (Z.quantile(0.75, dim=0) - Z.quantile(0.25, dim=0)).clamp(min=1e-6)
        dZdt_median = dZdt.median(dim=0).values
        dZdt_iqr = (dZdt.quantile(0.75, dim=0) - dZdt.quantile(0.25, dim=0)).clamp(min=1e-6)

        Z_norm = (Z - Z_median) / Z_iqr
        dZdt_norm = (dZdt - dZdt_median) / dZdt_iqr

        # Outlier rejection
        valid_mask = (Z_norm.abs() < 5).all(dim=1) & (dZdt_norm.abs() < 5).all(dim=1)
        
        if valid_mask.sum() < 2*d:
            print("Warning: Insufficient valid data. Using default stable dynamics.")
            self.A.data = -1.0 * torch.eye(d, device=device)
            return
        
        Z_fit = Z_norm[valid_mask]
        dZdt_fit = dZdt_norm[valid_mask]
        U_fit = U[valid_mask] if U is not None and self.m > 0 else None

        # 2. Build Data Matrix Φ = [Z, U]
        if U_fit is not None:
            U_median = U_fit.median(dim=0).values
            U_iqr = (U_fit.quantile(0.75, dim=0) - U_fit.quantile(0.25, dim=0)).clamp(min=1e-6)
            U_norm = (U_fit - U_median) / U_iqr
            Φ = torch.cat([Z_fit, U_norm], dim=1)
        else:
            Φ = Z_fit
        
        # 3. Robust Regression (Ridge via SVD)
        try:
            reg_lambda = self.reg.item()
            # Use SVD for robust damped least squares
            U_svd, S_svd, Vt_svd = torch.linalg.svd(Φ, full_matrices=False)
            
            # Damped SVD inverse (Tikhonov regularization)
            S_damped = S_svd / (S_svd**2 + reg_lambda)
            
            # Solve: θ = V @ diag(S_damped) @ U.T @ Y
            θ = Vt_svd.T @ (torch.diag(S_damped) @ (U_svd.T @ dZdt_fit))
            
        except Exception as e:
            print(f"Robust regression failed: {e}. Using default stable dynamics.")
            self.A.data = -1.0 * torch.eye(d, device=device)
            return

        # 4. Extract and Rescale parameters
        # A_rescaled = diag(dZdt_iqr) @ A_norm @ diag(1/Z_iqr)
        A_fitted = θ[:d].T * (dZdt_iqr.unsqueeze(1) / Z_iqr.unsqueeze(0))
        
        if U_fit is not None:
             B_fitted = θ[d:].T * (dZdt_iqr.unsqueeze(1) / U_iqr.unsqueeze(0))
             self.B.data = torch.clamp(B_fitted, -10.0, 10.0)

        # 5. Aggressive Stabilization (Fixes "ERROR: Dynamics still unstable")
        A_stable = A_fitted.clone()
        STABILITY_MARGIN = 0.5  # Ensure eigenvalues are <= -0.5
        
        try:
            eigvals = torch.linalg.eigvals(A_fitted).real
            max_eig = eigvals.max().item()
        except:
            max_eig = float('inf')

        if max_eig > -STABILITY_MARGIN or max_eig == float('inf'):
            if max_eig > 0:
                print(f"Warning: Found positive eigenvalues, max = {max_eig:.4f}")

            # Method: Shift the spectrum (A = A - shift * I)
            if max_eig != float('inf'):
                shift = max_eig + STABILITY_MARGIN
                A_stable = A_fitted - shift * torch.eye(d, device=device)
            else:
                print("Eigendecomposition failed, using fallback stabilization.")
                A_stable = -STABILITY_MARGIN * torch.eye(d, device=device)
            
        self.A.data = A_stable

        # 6. Final check
        final_max_eig = torch.linalg.eigvals(self.A.data).real.max().item()
        print(f"Final max eigenvalue after stabilization: {final_max_eig:.4f}")
        
        if final_max_eig > 0:
             print("CRITICAL: Stabilization failed. Forcing diagonal stable matrix.")
             self.A.data = -STABILITY_MARGIN * torch.eye(d, device=device)

        # 7. Compute residual (on original scale)
        Z_val = Z[valid_mask]
        dZdt_val = dZdt[valid_mask]
        U_val = U[valid_mask] if U is not None and self.m > 0 else None
        
        predictions = self.forward(Z_val, U_val)
        residuals = (dZdt_val - predictions).norm(dim=1)
        self.residual = torch.quantile(residuals, 0.95)

    def to(self, device):
        super().to(device)
        self._comb = self._comb.to(device)
        self.residual = self.residual.to(device)
        self.reg = self.reg.to(device)
        return self