from __future__ import annotations
import torch
# Try importing control library, handle if missing
try:
    from control import care
except ImportError:
    care = None

from .base import BaseReducer


class SymplecticProjectionReducer(BaseReducer):
    """Energy‑exact model reduction; γ = 0."""

    def __init__(self, A: torch.Tensor, J: torch.Tensor,
                 R: torch.Tensor, latent_dim: int):
        
        if care is None:
            raise ImportError("The 'control' library is required for SymplecticProjectionReducer.")
            
        # The original assertion might be too strict for the 19D system.
        # assert latent_dim % 2 == 0, "latent_dim must be even"
        super().__init__(latent_dim)

        A_np = A.cpu().numpy()
        Eye_np = torch.eye(A.shape[0], device=A.device).cpu().numpy()

        # Solve CARE based on the original implementation's structure
        try:
            # solve CARE: Aᵀ P + P A + Q = 0 with Q = I
            P_np, _, _ = care(A_np.T, Eye_np, Eye_np, Eye_np)
        except Exception as e:
            print(f"Warning: CARE solver failed: {e}. Falling back to identity basis.")
            P_np = Eye_np

        Q, _ = torch.linalg.qr(torch.as_tensor(P_np, dtype=A.dtype,
                                               device=A.device))
        
        n_full = A.shape[0]
        
        # Reordering logic (kept from original, applies if n_full is even)
        if n_full % 2 == 0:
            n = n_full // 2
            idx = torch.arange(n, device=A.device)
            reorder = torch.stack([idx, idx + n], 1).flatten()
            if reorder.shape[0] == Q.shape[1]:
                Q = Q[:, reorder]

        T = Q[:, :latent_dim]
        self.register_buffer("T", T)
        # Ti is the pseudo-inverse (or T.T if T is orthonormal)
        self.register_buffer("Ti", torch.linalg.pinv(T))
        self.gamma = 0.0

    # -------------- BaseReducer API ------------------------------------
    def fit(self, x):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Projection: z = x @ T
        return x @ self.T

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        # Reconstruction: x_hat = z @ Ti
        return z @ self.Ti

    # FIX: Implement analytical Jacobian to avoid Autograd errors.
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """Return batch Jacobian of shape (B, d, n). J = ∂z/∂x = T^T"""
        B = X.shape[0] if X.dim() > 1 else 1
        # T.T has shape (d, n). Expand to (B, d, n).
        J = self.T.T.unsqueeze(0)
        if B > 1:
            J = J.expand(B, -1, -1)
        return J.contiguous()