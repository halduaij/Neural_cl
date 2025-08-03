from __future__ import annotations
import torch
from control import care
from .base import BaseReducer


class SymplecticProjectionReducer(BaseReducer):
    """Energy‑exact model reduction; γ = 0."""

    def __init__(self, A: torch.Tensor, J: torch.Tensor,
                 R: torch.Tensor, latent_dim: int):
        assert latent_dim % 2 == 0, "latent_dim must be even"
        super().__init__(latent_dim)

        # solve CARE: Aᵀ P + P A + Q = 0 with Q = I
        P_np, _, _ = care(A.cpu().numpy().T,
                          torch.eye(A.shape[0]).numpy(),
                          torch.eye(A.shape[0]).numpy(),
                          torch.eye(A.shape[0]).numpy())
        Q, _ = torch.linalg.qr(torch.as_tensor(P_np, dtype=A.dtype,
                                               device=A.device))
        n = A.shape[0] // 2
        idx = torch.arange(n, device=A.device)
        reorder = torch.stack([idx, idx + n], 1).flatten()
        Q = Q[:, reorder]

        T = Q[:, :latent_dim]
        self.register_buffer("T", T)
        self.register_buffer("Ti", torch.linalg.pinv(T))
        self.gamma = 0.0

    # -------------- BaseReducer API ------------------------------------
    fit = lambda self, x: self
    forward = lambda self, x: x @ self.T
    inverse = lambda self, z: z @ self.Ti
