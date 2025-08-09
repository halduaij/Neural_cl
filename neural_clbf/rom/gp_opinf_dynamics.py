import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GPOpInfDynamics(nn.Module):
    """
    Robust Operator Inference Dynamics with optional quadratic terms.
    Learns z_dot = A z + B u + H (z⊗z) + C.
    """
    def __init__(self, d, m, include_quadratic: bool = False, reg: float = 0.0):
        super().__init__()
        self.d = d
        self.m = m
        self.include_quadratic = include_quadratic
        
        # Initialize A with stability bias
        self.A = nn.Parameter(torch.zeros(d, d))
        self.B = nn.Parameter(torch.zeros(d, m)) if m > 0 else None
        self.C = nn.Parameter(torch.zeros(d))
        
        # Quadratic term H: d x (d*(d+1)/2) for symmetric z⊗z
        if include_quadratic:
            n_quad = d * (d + 1) // 2
            self.H = nn.Parameter(torch.zeros(d, n_quad))
        else:
            self.H = None
        
        self.register_buffer("reg", torch.tensor(reg))
        self.residual = torch.tensor(float('inf'))
        self.model_order = "linear"  # Will be updated based on fitting

    def forward(self, z, u=None):
        """Forward pass with comprehensive shape handling."""
        debug = getattr(self, '_debug', False)
        
        # Ensure z is 2D
        if z.dim() == 1: 
            z = z.unsqueeze(0)
        
        batch_size = z.shape[0]
        
        if debug:
            print(f"[Forward] Input z shape: {z.shape}")
        
        # Validate z dimension
        if z.shape[1] != self.d:
            raise ValueError(f"State dimension mismatch: expected {self.d}, got {z.shape[1]}")
        
        # Linear dynamics: z @ A.T
        z_dot = torch.matmul(z, self.A.t())
        
        # Add constant term C
        z_dot = z_dot + self.C
        
        # Control input
        if self.B is not None and self.m > 0:
            if u is None:
                u = torch.zeros(batch_size, self.m, device=z.device, dtype=z.dtype)
            else:
                if u.dim() == 1:
                    u = u.unsqueeze(0)
                if u.shape[0] != batch_size:
                    u = u.expand(batch_size, -1)
                if u.shape[1] != self.m:
                    raise ValueError(f"Control dimension mismatch: expected {self.m}, got {u.shape[1]}")
            
            control_contribution = torch.matmul(u, self.B.t())
            z_dot = z_dot + control_contribution

        # >>> BUG FIX: The original 'forward' method was missing this block entirely. <<<
        # This part makes the quadratic model functional.
        if self.include_quadratic and self.H is not None:
            z_kron = self._compute_symmetric_kronecker(z)
            quadratic_contribution = torch.matmul(z_kron, self.H.t())
            z_dot = z_dot + quadratic_contribution
            if debug:
                print(f"[Forward] After quadratic: z_dot shape = {z_dot.shape}")
        
        return z_dot

    def _compute_symmetric_kronecker(self, z):
        """Compute symmetric Kronecker product [z_i*z_j for i<=j]"""
        batch_size = z.shape[0]
        d = z.shape[1]
        
        idx = 0
        n_quad = d * (d + 1) // 2
        z_kron = torch.zeros(batch_size, n_quad, device=z.device, dtype=z.dtype)

        # >>> CONVENTION CHANGE: This version removes the factor of 2.0 from off-diagonal terms. <<<
        # This is a valid modeling choice, as long as it's consistent between fit and forward.
        for i in range(d):
            for j in range(i, d):
                z_kron[:, idx] = z[:, i] * z[:, j]
                idx += 1
        
        return z_kron

    def fit(self, Z, U, dZdt):
        """
        Fit dynamics using robust Ridge Regression with automatic fallback.
        """
        logger.info(f"Fitting OpInf dynamics (d={self.d}, quadratic={self.include_quadratic})")
        device = Z.device
        N = Z.shape[0]

        # 1. Construct Data Matrix D = [Z, U, Z⊗Z, 1]
        D_list = [Z]
        if self.m > 0 and U is not None:
            D_list.append(U)
        
        n_params_quad = self.d + (self.m if self.m > 0 else 0) + 1
        if self.include_quadratic:
            n_params_quad += self.d * (self.d + 1) // 2
            
            if N < 3 * n_params_quad: # data sufficiency check
                logger.warning(f"  Insufficient data for quadratic model: {N} samples < 3*{n_params_quad} params. Falling back to linear.")
                self.include_quadratic = False
                self.H = None
        
        if self.include_quadratic and self.H is not None:
            Z_kron = self._compute_symmetric_kronecker(Z)
            D_list.append(Z_kron)
            self.model_order = "quadratic"
        
        D_list.append(torch.ones(N, 1, device=device))
        D = torch.cat(D_list, dim=1)
        
        # 2. Preprocessing & Regression
        valid_mask = torch.isfinite(D).all(dim=1) & torch.isfinite(dZdt).all(dim=1)
        D_clean = D[valid_mask]
        Y_clean = dZdt[valid_mask]
        
        if D_clean.shape[0] < D_clean.shape[1]:
            logger.error(f"  Insufficient valid data for regression. Using fallback.")
            self._set_fallback_dynamics(device)
            return

        try:
            lambda_reg = self.reg.item()
            if lambda_reg > 0:
                # Use Ridge Regression
                DTD = D_clean.T @ D_clean
                DTY = D_clean.T @ Y_clean
                reg_matrix = lambda_reg * torch.eye(DTD.shape[0], device=device)
                X_T = torch.linalg.solve(DTD + reg_matrix, DTY)
            else:
                # Use standard least squares
                X_T = torch.linalg.lstsq(D_clean, Y_clean).solution
            
            if not torch.isfinite(X_T).all():
                raise ValueError("NaN/Inf detected in regression solution")
            
            X = X_T.T
        except Exception as e:
            logger.error(f"  CRITICAL: Regression failed: {e}. Using fallback.")
            self._set_fallback_dynamics(device)
            return

        # 3. Assign parameters
        self._assign_parameters(X)
        
        # 4. Calculate residual
        Y_pred = D_clean @ X_T
        residuals = (Y_clean - Y_pred).norm(dim=1)
        self.residual = torch.quantile(residuals[torch.isfinite(residuals)], 0.95) if residuals.numel() > 0 else torch.tensor(float('inf'))
        logger.info(f"  Fit complete. 95th percentile residual: {self.residual:.4e}")


    def _assign_parameters(self, X):
        """Extract parameters from regression solution matrix X."""
        col_idx = 0
        
        self.A.data = X[:, col_idx : col_idx + self.d].clone()
        col_idx += self.d
        
        if self.B is not None:
            self.B.data = X[:, col_idx : col_idx + self.m].clone()
            col_idx += self.m
        
        if self.H is not None and self.include_quadratic:
            n_quad = self.d * (self.d + 1) // 2
            self.H.data = X[:, col_idx : col_idx + n_quad].clone()
            col_idx += n_quad
        
        self.C.data = X[:, col_idx].clone()

    def _set_fallback_dynamics(self, device):
        """Set physically reasonable fallback dynamics."""
        self.A.data = -0.1 * torch.eye(self.d, device=device)
        if self.B is not None: self.B.data.zero_()
        if self.H is not None: self.H.data.zero_()
        self.C.data.zero_()
        self.residual = torch.tensor(float('inf'))
        self.model_order = "fallback"
        logger.warning("  Using fallback dynamics: A = -0.1*I")