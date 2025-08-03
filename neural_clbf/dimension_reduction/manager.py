from __future__ import annotations
import torch, math
from .symplectic_projection import SymplecticProjectionReducer as SPR
from .opinf import OpInfReducer
from .lyap_coherency import LyapCoherencyReducer as LCR


def _score(r, w):
    return r.gamma + w * r.latent_dim / r.full_dim


def select_reducer(sys, X, Xdot, d_max=12, w_dim=0.02):
    n = X.shape[1]; V = sys.energy_function; V_min = V(X).min()
    if not X.requires_grad:
        X = X.clone().requires_grad_(True)

    candidates = []

    if hasattr(sys, "linearise"):
        try:
            A,J,R = sys.linearise(return_JR=True)
            for d in range(2, min(d_max, n), 2):
                spr = SPR(A,J,R,d); spr.full_dim = n; candidates.append(spr)
        except Exception:
            pass

    for k in (2,3,4):
        lcr = LCR(sys, k, X); lcr.full_dim = n
        lcr.gamma = lcr.gamma(float(V_min)); candidates.append(lcr)

    for d in range(4, min(d_max, n)):
        opinf = OpInfReducer(d, n).fit(X, Xdot, V, V_min)
        opinf.full_dim = n; candidates.append(opinf)

    return min(candidates, key=lambda r: _score(r, w_dim))
