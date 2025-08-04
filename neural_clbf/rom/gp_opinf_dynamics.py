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
        
        # Initialize A with stability bias (critical damping ≈ 0.1 s⁻¹)
        self.A = nn.Parameter(torch.eye(d) * -0.1) 
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
        if z.dim() == 1: 
            z = z.unsqueeze(0)
        
        # Linear dynamics
        z_dot = z @ self.A.T + self.C
        
        # Control input
        if self.B is not None:
            if u is None:
                u = torch.zeros(z.shape[0], self.m, device=z.device)
            elif u.dim() == 1:
                u = u.unsqueeze(0)
            if u.shape[1] == self.m:
                z_dot += u @ self.B.T
        
        # Quadratic dynamics
        if self.H is not None and self.include_quadratic:
            z_kron = self._compute_symmetric_kronecker(z)
            z_dot += z_kron @ self.H.T
        
        return z_dot

    def _compute_symmetric_kronecker(self, z):
        """Compute symmetric Kronecker product [z_i*z_j for i<=j]"""
        batch_size = z.shape[0]
        d = z.shape[1]
        
        # Create symmetric product efficiently
        idx = 0
        z_kron = torch.zeros(batch_size, d * (d + 1) // 2, device=z.device)
        for i in range(d):
            for j in range(i, d):
                if i == j:
                    z_kron[:, idx] = z[:, i] * z[:, j]
                else:
                    z_kron[:, idx] = 2.0 * z[:, i] * z[:, j]  # Factor of 2 for off-diagonal
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
        
        # Decide whether to include quadratic terms
        n_params_linear = self.d + (self.m if self.m > 0 else 0) + 1
        n_params_quad = n_params_linear
        
        if self.include_quadratic:
            n_params_quad += self.d * (self.d + 1) // 2
            
            # Data sufficiency check
            if N < 5 * n_params_quad:
                logger.warning(f"  Insufficient data for quadratic model: {N} samples < 5*{n_params_quad} params")
                logger.warning(f"  Falling back to linear model only")
                self.include_quadratic = False
                self.H = None
        
        # Add quadratic terms if still enabled
        if self.include_quadratic:
            Z_kron = self._compute_symmetric_kronecker(Z)
            D_list.append(Z_kron)
            self.model_order = "quadratic"
        
        # Add constant term
        D_list.append(torch.ones(N, 1, device=device))
        D = torch.cat(D_list, dim=1)
        
        # 2. Preprocessing - Remove NaNs/Infs
        valid_mask = torch.isfinite(D).all(dim=1) & torch.isfinite(dZdt).all(dim=1)
        D_clean = D[valid_mask]
        Y_clean = dZdt[valid_mask]
        N_valid = D_clean.shape[0]

        logger.info(f"  Valid samples: {N_valid}/{N} ({(N_valid/N if N>0 else 0)*100:.1f}%)")

        if N_valid < D_clean.shape[1] + 1:
            logger.error(f"  Insufficient valid data: {N_valid} < {D_clean.shape[1] + 1}")
            self._set_fallback_dynamics(device)
            return

        # 3. Center the data (but don't normalize)
        D_mean = D_clean.mean(dim=0, keepdim=True)
        D_centered = D_clean - D_mean
        Y_mean = Y_clean.mean(dim=0, keepdim=True) 
        Y_centered = Y_clean - Y_mean

        # 4. Compute adaptive regularization
        sigma_D = D_centered.std(dim=0).mean()
        lambda_reg = max(self.reg.item(), 1e-3 * sigma_D.item())
        logger.info(f"  Data scale σ_D = {sigma_D:.2e}, regularization λ = {lambda_reg:.2e}")

        # 5. Ridge Regression
        DTD = D_centered.T @ D_centered
        DTY = D_centered.T @ Y_centered
        
        # Apply scaled regularization
        reg_matrix = lambda_reg * torch.eye(DTD.shape[0], device=device)
        A_ridge = DTD + reg_matrix

        # Check condition number
        try:
            cond_number = torch.linalg.cond(A_ridge).item()
            logger.info(f"  Condition number (Ridge): {cond_number:.2e}")
        except:
            cond_number = float('inf')

        try:
            # Attempt direct solve
            X_T_centered = torch.linalg.solve(A_ridge, DTY)
            
            # Account for centering: X_full includes the offset for constant term
            X_T = X_T_centered.clone()
            # The constant term needs adjustment for centering
            offset_correction = Y_mean.T - (X_T_centered.T @ D_mean.unsqueeze(-1)).squeeze(-1)
            
            if not torch.isfinite(X_T).all() or not torch.isfinite(offset_correction).all():
                raise ValueError("NaN/Inf detected in Ridge solution")

            X = X_T.T 
            logger.info(f"  Regression successful: Ridge (direct solve)")

        except Exception as e:
            # Fallback to SVD-based solver
            logger.warning(f"  Direct solve failed: {e}. Using SVD-based lstsq")
            try:
                # Use original (uncentered) data for lstsq
                solution = torch.linalg.lstsq(D_clean, Y_clean, rcond=1e-6, driver='gelsd')
                X_T = solution.solution
                
                if not torch.isfinite(X_T).all():
                    raise ValueError("NaN/Inf in lstsq solution")
                    
                X = X_T.T
                offset_correction = None  # lstsq handles offset internally
                logger.info(f"  Regression successful: Least squares (SVD)")
                
            except Exception as e_svd:
                logger.error(f"  CRITICAL: All regression methods failed: {e_svd}")
                self._set_fallback_dynamics(device)
                return

        # 6. Extract and assign parameters
        self._assign_parameters(X, offset_correction)
        
        # 7. Calculate residual on clean data
        Y_pred = D_clean @ X_T
        residuals = (Y_clean - Y_pred).norm(dim=1)
        residuals_finite = residuals[torch.isfinite(residuals)]
        
        if residuals_finite.numel() > 0:
            self.residual = torch.quantile(residuals_finite, 0.95)
            mean_res = residuals_finite.mean().item()
            logger.info(f"  Mean residual: {mean_res:.4e}, 95th percentile: {self.residual:.4e}")
        else:
            self.residual = torch.tensor(float('inf'))
            logger.warning(f"  Could not compute residuals")

        # 8. Final stability check on learned A
        try:
            eigvals_A = torch.linalg.eigvals(self.A)
            max_real = eigvals_A.real.max().item()
            logger.info(f"  Learned A: max Re(λ) = {max_real:.3f}")
        except:
            logger.warning(f"  Could not verify eigenvalues of learned A")

    def _assign_parameters(self, X, offset_correction=None):
        """Extract parameters from regression solution matrix X"""
        col_idx = 0
        
        # Linear term A
        self.A.data = X[:, col_idx:col_idx + self.d]
        col_idx += self.d
        
        # Control term B
        if self.B is not None:
            self.B.data = X[:, col_idx:col_idx + self.m]
            col_idx += self.m
        
        # Quadratic term H
        if self.H is not None and self.include_quadratic:
            n_quad = self.d * (self.d + 1) // 2
            self.H.data = X[:, col_idx:col_idx + n_quad]
            col_idx += n_quad
        
        # Constant term C
        if offset_correction is not None:
            self.C.data = offset_correction.squeeze()
        else:
            self.C.data = X[:, col_idx].squeeze()

    def _set_fallback_dynamics(self, device):
        """Set physically reasonable fallback dynamics."""
        # Use light damping appropriate for power systems
        zeta = 0.05  # 0.05 s⁻¹ damping (reduced from 0.1)
        self.A.data = -zeta * torch.eye(self.d, device=device)
        
        if self.B is not None:
            self.B.data.zero_()
        if self.H is not None:
            self.H.data.zero_()
        self.C.data.zero_()
        
        self.residual = torch.tensor(float('inf'))
        self.model_order = "fallback"
        logger.warning(f"  Using fallback dynamics: A = -{zeta}*I")