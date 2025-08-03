"""
BaseReducer
===========

• Provides a **common interface** for every dimension‑reduction component
  (projection, auto‑encoder, operator‑inference, coherency, …).

• Handles:
    – storing ``latent_dim`` (d) and, optionally, ``full_dim`` (n),
    – a default γ attribute (Lyapunov robust‑margin, 0 if exact),
    – an autograd‑based fallback for the batch Jacobian.

Sub‑classes **must** implement

    forward(x)   :  (B,n) → (B,d)
    inverse(z)   :  (B,d) → (B,n)

and optionally override ``jacobian`` and ``fit``.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class BaseReducer(nn.Module):
    """
    Abstract parent for all reducers.

    Parameters
    ----------
    latent_dim : int
        Dimension d of the reduced / latent state.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim: int = latent_dim
        # Will be filled by the manager (n of the original system)
        self.full_dim: Optional[int] = None
        # Lyapunov‑robustness margin γ; subclasses overwrite if inexact
        self.gamma: float = 0.0

    # ------------------------------------------------------------------ #
    # Mandatory API for subclasses
    # ------------------------------------------------------------------ #
    def forward(self, X: torch.Tensor) -> torch.Tensor:          # x → z
        raise NotImplementedError("Reducer must implement forward()")

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:          # z → x̂
        raise NotImplementedError("Reducer must implement inverse()")
        
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Make reducer callable - just calls forward"""
        return self.forward(X)

    # Optional: fit is a no‑op by default (analytic reducers)
    def fit(self, X: torch.Tensor, *args, **kwargs):
        """
        Optionally learn internal parameters from snapshot data.

        Return self so that `.fit()` can be chained like scikit‑learn.
        """
        return self

    # ------------------------------------------------------------------ #
    # Jacobian helper
    # ------------------------------------------------------------------ #
    def jacobian(self, X: torch.Tensor) -> torch.Tensor:
        """
        Batch Jacobian  J = ∂z/∂x  of shape  (B, d, n).

        The default implementation uses autograd; override with an
        analytic expression if you have one (e.g., PCA, linear projection).
        """
        X_req_grad = X.requires_grad
        if not X_req_grad:
            X = X.clone().requires_grad_(True)

        Z = self.forward(X)                    # (B,d)
        batch, d = Z.shape
        n = X.shape[1]
        J = []

        for k in range(d):
            grad_k = torch.autograd.grad(
                Z[:, k].sum(), X, retain_graph=(k < d - 1), create_graph=False
            )[0]                               # (B,n)
            J.append(grad_k.unsqueeze(2))      # (B,n,1)

        J = torch.cat(J, dim=2)                # (B,n,d)
        J = J.permute(0, 2, 1).contiguous()    # (B,d,n)

        if not X_req_grad:
            X.requires_grad_(False)
        return J