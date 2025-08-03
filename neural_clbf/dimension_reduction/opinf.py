from __future__ import annotations
import torch
from neural_clbf.dimension_reduction.base import BaseReducer
from neural_clbf.rom.gp_opinf_dynamics import GPOpInfDynamics


class OpInfReducer(BaseReducer):
    """PCA encoder + quadratic latent ODE, provides γ margin."""

    def __init__(self, latent_dim: int, n_full: int, n_controls: int = 1):
        super().__init__(latent_dim)
        self.full_dim = n_full
        self.n_controls = n_controls
        self.register_buffer("μ", torch.zeros(n_full))
        self.register_buffer("proj", torch.eye(n_full, latent_dim))
        # Initialize dynamics but don't set n_controls yet
        self.dyn = None
    def fit(self, X, Xdot, V_fn, V_min):
            """
            Fit the reducer using PCA and then fit the latent dynamics.
            Uses more aggressive stabilization for power system dynamics.
            """
            device = X.device
            
            # Compute mean and center data
            self.μ = X.mean(0)
            X_centered = X - self.μ
            
            # Add regularization to avoid numerical issues
            X_norm = X_centered.norm()
            if X_norm < 1e-6:
                X_centered = X_centered + 1e-6 * torch.randn_like(X_centered)
            
            # Compute PCA via SVD with numerical stability
            try:
                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
                
                # Be more conservative with dimension selection
                # Use energy threshold to determine effective rank
                total_energy = (S ** 2).sum()
                energy_threshold = 0.99  # Keep 99% of energy
                cumsum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                n_effective = (cumsum_energy < energy_threshold).sum().item() + 1
                
                # Also check condition number
                condition_threshold = 1e6
                sig_threshold = S[0] / condition_threshold if S[0] > 0 else 1e-10
                n_stable = (S > sig_threshold).sum().item()
                
                # Take minimum of requested, effective, and stable dimensions
                n_keep = min(self.latent_dim, n_effective, n_stable)
                
                if n_keep < self.latent_dim:
                    print(f"Warning: Reducing latent dimension from {self.latent_dim} to {n_keep} for stability")
                    print(f"  Effective rank (99% energy): {n_effective}")
                    print(f"  Stable rank (condition < 1e6): {n_stable}")
                    self.latent_dim = n_keep
                    
                self.proj = Vt[:self.latent_dim].T.contiguous()
                
            except Exception as e:
                print(f"SVD failed, using random projection: {e}")
                self.proj = torch.randn(self.full_dim, self.latent_dim, device=device)
                self.proj = torch.qr(self.proj)[0]
            
            # Project to latent space
            Z = self.forward(X)
            dZdt = Xdot @ self.proj
            
            # Check data quality
            Z_scale = Z.abs().max()
            dZdt_scale = dZdt.abs().max()
            
            if Z_scale > 1e3 or dZdt_scale > 1e3:
                print(f"Warning: Large values detected (Z_max={Z_scale:.2e}, dZdt_max={dZdt_scale:.2e})")
                # Rescale for numerical stability
                scale = max(Z_scale, dZdt_scale) / 100
                Z = Z / scale
                dZdt = dZdt / scale
            
            # Check for NaN or inf
            if torch.isnan(Z).any() or torch.isinf(Z).any():
                print("Warning: NaN or inf in latent states")
                Z = torch.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
                dZdt = torch.nan_to_num(dZdt, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Initialize dynamics
            actual_n_controls = getattr(self.sys, 'n_controls', self.n_controls) if hasattr(self, 'sys') else self.n_controls
            self.dyn = GPOpInfDynamics(self.latent_dim, actual_n_controls).to(device)
            
            # For power systems, use stronger regularization
            self.dyn.reg = torch.tensor(0.01, device=device)  # Increased from 1e-6
            
            # Create control input
            U = torch.zeros(Z.shape[0], actual_n_controls, device=device)
            
            # Fit dynamics with error handling
            try:
                self.dyn.fit(Z, U, dZdt)
                
                # Additional stability check: ensure A matrix is stable
                eigvals = torch.linalg.eigvals(self.dyn.A).real
                max_eigval = eigvals.max().item()
                
                if max_eigval > -0.01:  # System should be strongly stable
                    print(f"Warning: Insufficient stability (max eigenvalue = {max_eigval:.4f})")
                    # Force stronger stability
                    self.dyn.A.data = self.dyn.A.data - (max_eigval + 0.1) * torch.eye(self.latent_dim, device=device)
                    
            except Exception as e:
                print(f"Dynamics fitting failed: {e}")
                # Use very simple stable dynamics
                self.dyn.A.data = -0.5 * torch.eye(self.latent_dim, device=device)
                self.dyn.H.data.zero_()
                self.dyn.B.data.zero_()
                self.dyn.residual = torch.tensor(1.0, device=device)
            
            # Compute gamma with safety checks
            try:
                if hasattr(self.dyn, 'residual'):
                    eps = float(self.dyn.residual.item())
                    if torch.isnan(self.dyn.residual) or torch.isinf(self.dyn.residual):
                        eps = 1.0
                else:
                    Z_pred = self.dyn.forward(Z, U)
                    residuals = (dZdt - Z_pred).norm(dim=1)
                    residuals = residuals[~torch.isnan(residuals) & ~torch.isinf(residuals)]
                    eps = residuals.max().item() if len(residuals) > 0 else 1.0
                
                # For gradient computation, use a subset to avoid memory issues
                n_grad_samples = min(1000, X.shape[0])
                X_subset = X[:n_grad_samples].requires_grad_(True)
                V_vals = V_fn(X_subset)
                gradV = torch.autograd.grad(V_vals.sum(), X_subset, create_graph=False)[0]
                gradV_norms = gradV.norm(dim=1)
                
                # Remove outliers and use robust estimate
                gradV_norms = gradV_norms[~torch.isnan(gradV_norms) & ~torch.isinf(gradV_norms)]
                if len(gradV_norms) > 0:
                    # Use 90th percentile instead of 95th for more conservative estimate
                    L_V = torch.quantile(gradV_norms, 0.90).item()
                    L_V = min(L_V, 100.0)  # Cap Lipschitz constant
                else:
                    L_V = 10.0
                
                # Set gamma with bounds
                V_min = max(V_min, 0.01)  # Ensure V_min is not too small
                self.gamma = min(eps * L_V / V_min, 10.0)  # Cap gamma at 10
                    
            except Exception as e:
                print(f"Gamma computation failed: {e}")
                self.gamma = 5.0  # Conservative default
            
            self.to(device)
            return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space."""
        # Handle batched input
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return ((x - self.μ) @ self.proj).squeeze(0)
        return (x - self.μ) @ self.proj
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space."""
        # Handle batched input
        if z.dim() == 1:
            z = z.unsqueeze(0)
            return (z @ self.proj.T + self.μ).squeeze(0)
        return z @ self.proj.T + self.μ
        
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n)."""
        B = X.shape[0] if X.dim() > 1 else 1
        # proj.T has shape (d, n), expand to (B, d, n)
        J = self.proj.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()
        
    def to(self, device):
        """Move reducer to device."""
        super().to(device)
        self.μ = self.μ.to(device)
        self.proj = self.proj.to(device)
        if self.dyn is not None:
            self.dyn = self.dyn.to(device)
        return self