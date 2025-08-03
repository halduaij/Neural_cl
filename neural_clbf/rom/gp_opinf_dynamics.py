from __future__ import annotations
import torch, itertools
from torch import nn


class GPOpInfDynamics(nn.Module):
    """Quadratic operator inference latent ODE with improved numerical stability."""

    def __init__(self, d: int, m: int = 1):
        super().__init__()
        self.d = d
        self.m = m
        self.A = nn.Parameter(torch.zeros(d, d), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(d, m), requires_grad=False)
        comb = list(itertools.combinations_with_replacement(range(d), 2))
        self.register_buffer("_comb", torch.tensor(comb, dtype=torch.long))
        self.H = nn.Parameter(torch.zeros(d, len(comb)), requires_grad=False)
        self.register_buffer("residual", torch.tensor(0.0))
        # Add regularization parameter
        self.register_buffer("reg", torch.tensor(1e-6))

    def _q(self, z):
        """Compute quadratic terms with bounds"""
        z1 = z[..., self._comb[:, 0]]
        z2 = z[..., self._comb[:, 1]]
        q = z1 * z2
        # Clip to avoid extreme values
        return torch.clamp(q, -1e6, 1e6)

    def forward(self, z, u=None):
        """
        Compute latent dynamics dz/dt = Az + H*q(z) + Bu
        
        Args:
            z: (B, d) latent state
            u: (B, m) control input (optional)
        
        Returns:
            dz/dt: (B, d) latent dynamics
        """
        # Ensure z has batch dimension
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Check for NaN/inf in input
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)
            
        # Linear term with damping for stability
        out = z @ self.A.T
        
        # Add small damping to prevent explosion
        out = out - self.reg * z
        
        # Quadratic term (only if H is not too large)
        if self.H.norm() < 1e3:
            q = self._q(z)
            out = out + q @ self.H.T
        
        # Control term
        if u is not None:
            # Ensure u has correct shape
            if u.dim() == 1:
                u = u.unsqueeze(0)  # (m,) -> (1, m)
            if u.shape[0] == 1 and z.shape[0] > 1:
                u = u.expand(z.shape[0], -1)  # (1, m) -> (B, m)
            
            # Check control magnitude
            if u.norm() < 1e3:
                out = out + u @ self.B.T
        
        # Clip output to prevent explosion
        out = torch.clamp(out, -1e3, 1e3)
            
        return out

    @torch.no_grad()
    def fit(self, Z, U, dZdt):
        """
        Fit dynamics using regularized least squares.
        
        Args:
            Z: (N, d) latent states
            U: (N, m) control inputs
            dZdt: (N, d) latent derivatives
        """
        device = Z.device
        N, d = Z.shape
        
        # Check for invalid data
        valid_mask = ~(torch.isnan(Z).any(dim=1) | torch.isnan(dZdt).any(dim=1) | 
                      torch.isinf(Z).any(dim=1) | torch.isinf(dZdt).any(dim=1))
        
        if valid_mask.sum() < 10:  # Need at least 10 valid samples
            print("Warning: Insufficient valid data for dynamics fitting")
            # Set simple stable dynamics
            self.A.data = -torch.eye(d, device=device) * 0.1
            self.H.data.zero_()
            self.B.data.zero_()
            self.residual = torch.tensor(1.0, device=device)
            return
        
        # Use only valid data
        Z_valid = Z[valid_mask]
        dZdt_valid = dZdt[valid_mask]
        U_valid = U[valid_mask] if U is not None else None
        
        # Build data matrix with quadratic terms
        Q = self._q(Z_valid)  # (N_valid, n_quad)
        
        # Stack features [Z, Q, U]
        if U_valid is not None and U_valid.shape[1] > 0:
            Φ = torch.cat([Z_valid, Q, U_valid], dim=1)
        else:
            Φ = torch.cat([Z_valid, Q], dim=1)
        
        # Add regularization to prevent ill-conditioning
        n_features = Φ.shape[1]
        reg_matrix = self.reg * torch.eye(n_features, device=device)
        
        try:
            # Regularized least squares: (Φ'Φ + λI)θ = Φ'y
            ΦtΦ = Φ.T @ Φ + reg_matrix
            Φty = Φ.T @ dZdt_valid
            
            # Solve with Cholesky decomposition for stability
            L = torch.linalg.cholesky(ΦtΦ)
            θ = torch.cholesky_solve(Φty, L)
            
        except Exception as e:
            print(f"Least squares failed: {e}, using simple dynamics")
            # Fallback to simple stable dynamics
            self.A.data = -torch.eye(d, device=device) * 0.1
            self.H.data.zero_()
            self.B.data.zero_()
            self.residual = torch.tensor(1.0, device=device)
            return
        
        # Extract parameters with bounds
        self.A.copy_(torch.clamp(θ[:d].T, -10, 10))
        
        n_quad = Q.shape[1]
        self.H.copy_(torch.clamp(θ[d:d+n_quad].T, -1, 1))
        
        if U_valid is not None and U_valid.shape[1] > 0:
            self.B.copy_(torch.clamp(θ[d+n_quad:].T, -10, 10))
        
        # Compute residual on valid data
        predictions = Φ @ θ
        residuals = (dZdt_valid - predictions).norm(dim=1)
        # Use 95th percentile instead of max to avoid outliers
        self.residual = torch.quantile(residuals, 0.95)
        
        # Check if dynamics are stable (eigenvalues of A)
        try:
            eigvals = torch.linalg.eigvals(self.A).real
            if eigvals.max() > 0.1:  # System might be unstable
                print("Warning: Fitted dynamics may be unstable, adding damping")
                self.A.data = self.A.data - 0.1 * torch.eye(d, device=device)
        except:
            pass
        
    def to(self, device):
        """Move module to device"""
        super().to(device)
        return self